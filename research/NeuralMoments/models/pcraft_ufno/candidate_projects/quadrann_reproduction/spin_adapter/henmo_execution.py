"""Padded, deterministic-reduction execution of boundary-response frontiers.

Only execution is batched here: every stencil retains the same finite-cut
witnesses, local material context and physical bandwidth as serial evaluation.
The pointwise MLP/attention and its first tangent run once per bounded wave.
Moment reduction uses FP64 batched matrix products, never CUDA index_add.
"""
from __future__ import annotations

import numpy as np
import torch

from .compact_pino_v21 import PackedPrediction


class StencilBatchPredictions(list):
    """List-compatible per-stencil predictions with aggregate execution data."""
    def __init__(self, predictions=(), *, diagnostics):
        super().__init__(predictions)
        self.diagnostics = diagnostics


def _basis(points):
    x,y,z=points.unbind(-1)
    return torch.stack((torch.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y),dim=-1)


def _pad_rows(value,count,fill=0.):
    if not count:
        return value
    return torch.cat((value,value.new_full((count,)+tuple(value.shape[1:]),fill)),dim=0)


def _motion_array(velocity,surface):
    if isinstance(velocity,torch.Tensor):
        velocity=velocity.detach().cpu().numpy()
    value=np.asarray(velocity,dtype=np.float64)
    if (value.ndim!=3 or value.shape[:2]!=(len(surface.vertices),3)
            or value.shape[2]<1 or not np.isfinite(value).all()):
        raise ValueError('vertex_velocity must match the same finite operand [V,3,D]')
    return value


def _waves(segments,budget):
    wave=[]; maximum=0
    for segment in segments:
        length=segment[2]-segment[1]
        next_max=max(maximum,length)
        if wave and (len(wave)+1)*next_max>budget:
            yield wave
            wave=[]; maximum=0
        wave.append(segment); maximum=max(maximum,length)
    if wave:
        yield wave


def evaluate_stencil_batch(stencils,contexts,model,cut_tokens,cut_motion=None,
                           vertex_velocity=None,operation='intersection',
                           point_origin=(0.,0.,0.),point_scale=1.,*,
                           max_padded_atoms=65536):
    """Evaluate same-width stencils using one neural call per padded wave.

    ``contexts[i]`` is [active_i, C]; it stays connected to its encoder graph.
    All stencils must refer to the identical prepared finite surface. Ambiguous
    active nearest features withhold J for that stencil only. Their M remains a
    proposal, as in the serial evaluator. No inference result is cached here.

    A stencil exceeding ``max_padded_atoms`` is segmented and accumulated before
    return. A fully inactive group performs no neural forward/tangent call.
    ``result.diagnostics`` describes actual wave/padding counts, not throughput.
    """
    stencils=list(stencils)
    if operation not in ('intersection','difference'):
        raise ValueError('operation must be intersection or difference')
    if type(max_padded_atoms) is not int or max_padded_atoms<1:
        raise ValueError('max_padded_atoms must be a positive integer')
    if contexts is None:
        if model.context_dim:
            raise ValueError('one context tensor per stencil is required')
        contexts=[None]*len(stencils)
    else:
        contexts=list(contexts)
    if len(contexts)!=len(stencils):
        raise ValueError('contexts must align one-to-one with stencils')
    summary=dict(stencils=len(stencils),active_stencils=0,active_atoms=0,
        padded_atoms=0,padding_atoms=0,network_calls=0,waves=[],
        max_padded_atoms=max_padded_atoms,contraction='float64_batched_matmul',
        index_add_used=False,derivative='same_response_complete_first_tangent')
    if not stencils:
        return StencilBatchPredictions(diagnostics=summary)
    parameter=next(model.parameters())
    device,dtype=parameter.device,parameter.dtype
    surface=stencils[0].surface
    width=float(stencils[0].bandwidth)
    if not np.isfinite(width) or width<=0:
        raise ValueError('positive finite stencil bandwidth required')
    if any(st.surface is not surface or float(st.bandwidth)!=width for st in stencils):
        raise ValueError('batch must share the exact finite surface and bandwidth')
    surface.check_tensor_source()
    motion=None if vertex_velocity is None else _motion_array(vertex_velocity,surface)
    dimension=None if motion is None else motion.shape[2]
    if motion is not None and cut_motion is None:
        raise ValueError('cut_motion is required for the complete conditioning derivative')
    if motion is None and cut_motion is not None:
        raise ValueError('cut_motion requires vertex_velocity')
    if cut_tokens is None:
        raise ValueError('finite-cut descriptor tokens are required')

    def tensor(value,kind=torch.float64):
        if isinstance(value,torch.Tensor):
            return value.to(dtype=kind,device=device)
        return torch.tensor(np.array(value,copy=True),dtype=kind,device=device)

    cut=tensor(cut_tokens,dtype)
    tangent_cut=None if cut_motion is None else tensor(cut_motion,dtype)
    if cut.ndim!=2 or cut.shape[1]!=model.cut_feature_dim or not len(cut):
        raise ValueError('cut_tokens must match model cut feature width')
    if tangent_cut is not None and tuple(tangent_cut.shape)!=tuple(cut.shape)+(dimension,):
        raise ValueError('cut_motion must match cut_tokens and the supplied chart')
    if not torch.isfinite(cut).all() or tangent_cut is not None and not torch.isfinite(tangent_cut).all():
        raise ValueError('finite descriptor values and tangents required')
    active_lengths=[]; prepared=[]; values=[]; jacobians=[]; flags=[]
    for stencil,context in zip(stencils,contexts):
        count=len(stencil.nodes)
        active_lengths.append(count)
        if (np.asarray(stencil.nodes).shape!=(count,3)
                or np.asarray(stencil.weights).shape!=(count,)
                or np.asarray(stencil.signed_distance).shape!=(count,)
                or not np.isfinite(stencil.nodes).all() or not np.isfinite(stencil.weights).all()
                or np.any(stencil.weights<=0) or not np.isfinite(stencil.signed_distance).all()
                or np.any(np.abs(stencil.signed_distance)>=width)):
            raise ValueError('stencil must contain positive, finite, strictly active material atoms')
        if context is None:
            ctx=torch.empty((count,0),dtype=dtype,device=device)
        else:
            ctx=tensor(context,dtype)
        if ctx.shape!=(count,model.context_dim):
            raise ValueError('context must match active stencil nodes and model width')
        points=tensor(stencil.nodes)
        weight=tensor(stencil.weights)
        signed=tensor(stencil.signed_distance)
        ambiguous=bool(stencil.ambiguous_nodes)
        # Independent AD uses the same local witness velocity lift as serial
        # execution; it does not freeze a selected face through topology events.
        live=getattr(surface,'tensor_vertices',None)
        if live is not None and not ambiguous and count:
            face=torch.tensor(surface.faces[stencil.face_indices].copy(),dtype=torch.long,device=device)
            displacement=(live-live.detach()).to(dtype=torch.float64,device=device)
            bary=tensor(stencil.barycentric)
            local_displacement=sum(bary[:,i:i+1]*displacement[face[:,i]] for i in range(3))
            signed=signed+(tensor(stencil.motion_normal)*local_displacement).sum(1)
        ds=None
        if motion is not None:
            ds=(torch.zeros((count,dimension),dtype=torch.float64,device=device) if ambiguous
                else tensor(stencil.distance_motion(motion)))
        retained=stencil.interior_moments if operation=='intersection' else stencil.exterior_moments
        if np.asarray(retained).shape!=(10,) or not np.isfinite(retained).all():
            raise ValueError('ten finite saturated moments required')
        base=tensor(retained)
        values.append(base)
        jacobians.append(None if dimension is None or ambiguous else base.new_zeros((10,dimension)))
        flags.append(ambiguous)
        prepared.append((points,weight,signed,ctx,ds))
    segments=[(i,begin,min(begin+max_padded_atoms,count)) for i,count in enumerate(active_lengths)
              for begin in range(0,count,max_padded_atoms)]
    summary['active_stencils']=sum(count>0 for count in active_lengths)
    summary['active_atoms']=sum(active_lengths)
    for wave in _waves(segments,max_padded_atoms):
        maximum=max(end-begin for _,begin,end in wave)
        padded_points=[]; padded_weights=[]; padded_distances=[]; padded_contexts=[]; padded_motion=[]
        for index,begin,end in wave:
            points,weight,signed,context,ds=prepared[index]
            padding=maximum-(end-begin)
            padded_points.append(_pad_rows(points[begin:end],padding))
            padded_weights.append(_pad_rows(weight[begin:end],padding))
            padded_distances.append(_pad_rows(signed[begin:end],padding,2*width))
            padded_contexts.append(_pad_rows(context[begin:end],padding))
            if dimension is not None:
                padded_motion.append(_pad_rows(ds[begin:end],padding))
        xyz=torch.stack(padded_points)
        weight=torch.stack(padded_weights)
        signed=torch.stack(padded_distances)
        context=torch.stack(padded_contexts)
        kwargs=dict(bandwidth=width,context=context.reshape(len(wave)*maximum,model.context_dim),cut_tokens=cut,
                    operation=operation,point_origin=point_origin,point_scale=point_scale)
        if dimension is None:
            retention=model(xyz.reshape(-1,3),signed.reshape(-1),**kwargs)
            response=retention.reshape(len(wave),maximum,1).to(torch.float64)
        else:
            ds=torch.stack(padded_motion)
            retention,derivative=model.retention_and_jacobian(xyz.reshape(-1,3),signed.reshape(-1),
                distance_motion=ds.reshape(-1,dimension),cut_motion=tangent_cut,**kwargs)
            response=torch.cat((retention[:,None],derivative),dim=1).reshape(
                len(wave),maximum,dimension+1).to(torch.float64)
        weighted=(_basis(xyz)*weight[:,:,None]).transpose(1,2)
        reduced=torch.bmm(weighted,response)
        for position,(index,_,_) in enumerate(wave):
            values[index]=values[index]+reduced[position,:,0]
            if jacobians[index] is not None:
                jacobians[index]=jacobians[index]+reduced[position,:,1:]
        count=len(wave)*maximum
        summary['padded_atoms']+=count
        summary['network_calls']+=1
        summary['waves'].append(dict(stencil_indices=[i for i,_,_ in wave],
            active_counts=[end-begin for _,begin,end in wave],padded_atoms=count,width=maximum))
    summary['padding_atoms']=summary['padded_atoms']-summary['active_atoms']
    if any(count==0 for count in active_lengths) and torch.is_grad_enabled():
        zero=sum((param.sum()*0. for param in model.parameters()),values[0].new_zeros(()))
        for index,count in enumerate(active_lengths):
            if count==0:
                values[index]=values[index]+zero
                if jacobians[index] is not None:
                    jacobians[index]=jacobians[index]+zero
    output=[]
    for i,stencil in enumerate(stencils):
        diagnostics=dict(total_nodes=stencil.total_nodes,active_nodes=active_lengths[i],
            ambiguous_nodes=int(stencil.ambiguous_nodes),batch_stencils=len(stencils),
            batch_active_atoms=summary['active_atoms'],batch_padded_atoms=summary['padded_atoms'],
            batch_network_calls=summary['network_calls'],
            derivative='same_response_complete_first_tangent',hard_cut_equivalence=False,
            solver_status='NATIVE_REQUIRED_DERIVATIVE' if flags[i] and dimension is not None
                          else 'PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE')
        output.append(PackedPrediction(values[i],jacobians[i],diagnostics))
    return StencilBatchPredictions(output,diagnostics=summary)
