"""Hierarchical boundary neural moments with event-aware first derivatives.

Three independently comparable paths share the material and finite operand:
legacy E16 response, a new cut-conditioned response, and deterministic cubature.
Adaptive refinement blends by a Moment discrepancy and includes its complete
first derivative. Jacobian discrepancies qualify a proposal; they never enter
a hidden discontinuous gate or require a neural input Hessian.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import numpy as np
import torch
from torch import nn

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import NativePCRAFTConfig, PcraftNativeMomentOperator
from .compact_pino_v21 import PackedPrediction, native_proposal_status
from .henmo_config import HENMOConfig
from .henmo_hierarchy import CoarseCubatureUnavailable
from .henmo_execution import evaluate_stencil_batch
from .henmo_geometry import (GraphLifetime, HENMOGeometry, MaterialHandle,
                             encode_table, geometry_versions, make_hierarchy)
from .henmo_response import HENMOBoundaryResponse, finite_cut_tokens, integrate_boundary_response
from .v21_response import LocalRetention, prepare_measure, prepare_stencil, evaluate_stencil, prepare_surface
from .v23_surface_runtime import SurfaceExecutionContext


def _stage(trace, name):
    return nullcontext() if trace is None else trace.stage(name)


def blend_moments(coarse, fine, probe, scale, *, tolerance, start=1., end=4.):
    """C1 blend using the coarse-to-one-level-probe discrepancy.

``fine`` may itself be recursively refined. The gate uses ``probe``, not a
detached score or a J-dependent decision. All values retain their training graph.
The discrepancy is an empirical refinement indicator, not a rigorous error bound.
"""
    scale = torch.as_tensor(scale, dtype=coarse.moments.dtype, device=coarse.moments.device)
    delta = probe.moments-coarse.moments
    u = ((delta/scale)**2).sum()/tolerance**2
    t = ((u-start)/(end-start)).clamp(0.,1.)
    alpha = t*t*(3.-2.*t)
    moment = (1.-alpha)*coarse.moments+alpha*fine.moments
    jacobian = None
    if coarse.jacobian is not None and fine.jacobian is not None and probe.jacobian is not None:
        du = (2.*delta[:,None]*(probe.jacobian-coarse.jacobian)/scale[:,None]**2).sum(0)/tolerance**2
        slope = 6.*t*(1.-t)/(end-start)
        slope = slope*((u > start)&(u < end)).to(slope.dtype)
        dalpha = slope*du
        jacobian = ((1.-alpha)*coarse.jacobian+alpha*fine.jacobian
                    +(fine.moments-coarse.moments)[:,None]*dalpha[None,:])
    return PackedPrediction(moment,jacobian,dict(blend_alpha=float(alpha.detach()),
                normalized_squared_discrepancy=float(u.detach()),
                indicator_kind='empirical_parent_child_moment_discrepancy'))


class HENMOOperator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = HENMOConfig() if config is None else config
        if not isinstance(self.config,HENMOConfig):
            raise ValueError('HENMOConfig required')
        c = self.config
        native = NativePCRAFTConfig(grid_size=c.context_grids[-1],
            pino_width=c.pino_width,pino_modes=c.pino_modes,pino_layers=c.pino_layers,
            interval_width=c.geometry_width,interval_depth=c.interval_depth,
            min_native_resolution=c.native_material_resolution,
            multiscale_enabled=True,multiscale_grids=c.context_grids,
            gno_enabled=False,transolver_enabled=False,checkpoint_intervals=True)
        self.shared_encoder = PcraftNativeMomentOperator(native,encoder_only=True)
        self.geometry_attention = nn.MultiheadAttention(c.geometry_width,4,batch_first=True)
        self.local_geometry = (nn.Sequential(nn.Linear(84,c.geometry_width),nn.SiLU(),
                                nn.Linear(c.geometry_width,c.geometry_width))
                               if c.response_path == 'neural' else None)
        self.response = (LocalRetention(c.geometry_width,c.response_hidden,
                            response_type=c.legacy_response_type,zero_init=c.response_zero_init)
                         if c.response_path == 'legacy' else
                         HENMOBoundaryResponse(c.geometry_width,c.response_hidden,
                             use_cut_attention=c.cut_attention,zero_init=c.response_zero_init,
                             profile=c.response_profile))
        self.surface_execution = SurfaceExecutionContext(max_workers=c.surface_workers)
        self._tree_entry = self._feature_entry = None
        self._closed = False
        self._counts = dict(encoder_calls=0,feature_hits=0,hierarchy_builds=0,queries=0)
        self.set_training_scope(c.training_scope)

    @property
    def training_scope(self):
        return self.config.training_scope

    def set_training_scope(self, scope):
        if scope not in ('all','response'):
            raise ValueError('scope must be all or response')
        self.config = replace(self.config,training_scope=scope)
        for module in (self.shared_encoder,self.geometry_attention,self.local_geometry):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(scope == 'all' and self.config.response_path != 'deterministic')
                p.grad = None
        for p in self.response.parameters():
            p.requires_grad_(self.config.response_path != 'deterministic')
            p.grad = None
        self.invalidate_caches()
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self,'config',None) and self.training_scope == 'response':
            self.shared_encoder.eval(); self.geometry_attention.eval()
            if self.local_geometry is not None:
                self.local_geometry.eval()
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn,recurse=recurse)
        self.invalidate_caches()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict,strict=strict,assign=assign)
        self.invalidate_caches()
        return result

    def invalidate_caches(self, *, force=False):
        # Geometry-only hierarchy and positive atoms survive parameter updates.
        if self._feature_entry is not None:
            entry = self._feature_entry[1]
            if force or entry.versions != geometry_versions(self) or entry.graph_lifetime is not None:
                self._feature_entry = None

    def clear_efficiency_caches(self):
        self.invalidate_caches(force=True)
        if self._tree_entry is not None:
            self._tree_entry[1].clear_cache()
        self._tree_entry = None

    def close(self):
        if not self._closed:
            self.clear_efficiency_caches()
            self.surface_execution.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def efficiency_statistics(self):
        return dict(self._counts,architecture='HENMO_V1_PLUS',config_fingerprint=self.config.fingerprint(),
                    native_resolution=self.config.native_material_resolution,
                    feature_cache_bytes=0 if self._feature_entry is None else self._feature_entry[1].nbytes,
                    hierarchy=None if self._tree_entry is None else self._tree_entry[1].diagnostics,
                    surface_execution=self.surface_execution.snapshot(),closed=self._closed)

    def encode_geometry(self, source, fixed_moments, density=1., *, inference=False):
        if self._closed:
            raise RuntimeError('HENMO operator is closed')
        if inference and self.training:
            raise ValueError('inference geometry requires eval mode')
        if not np.isfinite(density) or density <= 0:
            raise ValueError('positive finite material density required')
        fixed = np.array(fixed_moments,dtype=np.float64,copy=True)
        if (fixed.shape != (10,) or not np.isfinite(fixed).all() or fixed[0] < 0
                or fixed[0] == 0 and np.any(fixed != 0)):
            raise ValueError('ten physically valid fixed moments required')
        fixed.flags.writeable = False
        if self._tree_entry is not None and self._tree_entry[0] is source:
            tree = self._tree_entry[1]
            tree.assert_source_unchanged()
        else:
            tree = make_hierarchy(source,self.config)
            self._tree_entry = (source,tree)
            self._counts['hierarchy_builds'] += 1
        if tree.material is not None and float(tree.material.density) != float(density):
            raise ValueError('encoder density differs from registered material density')
        versions = geometry_versions(self)
        key = (id(tree),tree.identity,fixed.tobytes(),float(density),versions,self.config.feature_grid)
        frozen = self.training_scope == 'response' or self.config.response_path == 'deterministic'
        reusable = frozen or inference or not torch.is_grad_enabled()
        inference_only = inference or not torch.is_grad_enabled()
        if reusable and self._feature_entry is not None and self._feature_entry[0] == key:
            entry = self._feature_entry[1]
            self._counts['feature_hits'] += 1
            return replace(entry,inference_only=inference_only)
        with torch.no_grad() if reusable else nullcontext():
            if self.config.response_path == 'deterministic':
                p = next(self.parameters())
                table = p.new_zeros((1,self.config.geometry_width))
            else:
                table = encode_table(self,tree,fixed,float(density))
                self._counts['encoder_calls'] += 1
            if table.is_inference():
                with torch.inference_mode(False),torch.no_grad():
                    table = table.clone()
        life = None
        if table.requires_grad and not reusable:
            life = GraphLifetime(); table.register_hook(life.backward)
        handle = MaterialHandle(tree.material_identity,tree.side,tree.pitch,tree.origin)
        result = HENMOGeometry(tree,handle,fixed,table,self.config.feature_grid,id(self),versions,
                              table._version,fixed.tobytes(),inference_only,life)
        if reusable and result.nbytes <= self.config.feature_cache_bytes:
            self._feature_entry = (key,result)
        return result

    def _validate_geometry(self, cache):
        if self._closed:
            raise RuntimeError('HENMO operator is closed')
        if not isinstance(cache,HENMOGeometry) or cache.owner != id(self) or cache.versions != geometry_versions(self):
            raise ValueError('foreign or stale encoded geometry; encode again after encoder update')
        if cache.table._version != cache.table_version or cache.fixed_moments.tobytes() != cache.fixed_bytes:
            raise ValueError('encoded feature or fixed material payload was modified')
        if cache.inference_only and self.training:
            raise ValueError('inference-only geometry cannot be used for training')
        if cache.graph_lifetime is not None and cache.graph_lifetime.consumed:
            raise ValueError('training geometry graph consumed; encode for next update')
        cache.hierarchy.assert_source_unchanged()

    def evaluate_operand(self, cache, operand, *, with_jacobian=True, bandwidth=None,
                         chunk_size=None, trace=None, parameter_scales=None,coarse_grid=None):
        if type(with_jacobian) is not bool:
            raise ValueError('with_jacobian must be Boolean')
        if with_jacobian and operand.vertex_velocity is None:
            raise ValueError('qualified native vertex motion is required for a Jacobian')
        with _stage(trace,'surface_prepare'):
            surface = prepare_surface(operand.vertices,operand.faces,execution_context=self.surface_execution)
        return self.evaluate_surface(cache,surface,operation=operand.operation,
            vertex_velocity=operand.vertex_velocity if with_jacobian else None,
            bandwidth=bandwidth,chunk_size=chunk_size,trace=trace,parameter_scales=parameter_scales,
            coarse_grid=coarse_grid)

    def evaluate_surface(self, cache, surface, *, operation='intersection', vertex_velocity=None,
                         bandwidth=None, chunk_size=None, trace=None, parameter_scales=None,coarse_grid=None):
        """Also accepts a live tensor FiniteSurface for independent local AD checks."""
        self._validate_geometry(cache)
        c, tree = self.config, cache.hierarchy
        execution = c.training_execution if self.training else c.execution
        query_grid = c.coarse_grid if coarse_grid is None else coarse_grid
        if type(query_grid) is not int or query_grid < 1 or query_grid & (query_grid-1) or query_grid > tree.leaf_grid:
            raise ValueError('coarse_grid must be a valid power-of-two hierarchy level')
        width = c.bandwidth_voxels*tree.pitch if bandwidth is None else float(bandwidth)
        if operation not in ('intersection','difference') or not np.isfinite(width) or width <= 0:
            raise ValueError('registered operation and positive bandwidth required')
        chunk = c.chunk_size if chunk_size is None else chunk_size
        if type(chunk) is not int or chunk < 1:
            raise ValueError('positive chunk_size required')
        motion = vertex_velocity
        dimension = None if motion is None else motion.shape[-1]
        if motion is not None and (tuple(motion.shape[:2]) != tuple(surface.vertices.shape)
                or dimension not in (3,8,10) or not np.isfinite(np.asarray(motion)).all()):
            raise ValueError('motion must follow the finite operand and valid 3/8/10 chart')
        p = next(self.response.parameters())
        def tensor(a):
            return torch.tensor(np.array(a,copy=True),dtype=torch.float64,device=p.device)
        parameter_scale = np.ones(dimension) if dimension is not None else None
        if parameter_scales is not None:
            parameter_scale = np.asarray(parameter_scales,dtype=float)
            if dimension is None or parameter_scale.shape != (dimension,) or not np.isfinite(parameter_scale).all() or np.any(parameter_scale <= 0):
                raise ValueError('positive scales for the actual derivative chart required')
        with _stage(trace,'hierarchy_classification'):
            stop = tree.leaf_grid if execution == 'leaf' or c.response_path == 'deterministic' else query_grid
            coarse_support = max(width,np.sqrt(3.)*tree.side*tree.pitch/stop*c.coarse_support_fraction)
            selection_width = width if execution == 'leaf' or c.response_path == 'deterministic' else coarse_support
            selected = tree.select(surface,selection_width,operation,stop_grid=stop)
        self._counts['queries'] += 1
        diagnostics = dict(architecture='HENMO_V1_PLUS',response_path=c.response_path,
            execution=execution,training_execution=self.training,native_resolution=tree.side,leaf_grid=tree.leaf_grid,
            coarse_grid=stop,coarse_support_fraction=c.coarse_support_fraction,
            selected_coarse_blocks=len(selected.boundary_nodes),visited_nodes=selected.visited_nodes,
            classification_distance_queries=selected.distance_queries,
            saturated_nodes=len(selected.retained_nodes),discarded_nodes=len(selected.discarded_nodes),
            evaluated_atoms=0,active_atoms=0,ambiguous_atoms=0,coarse_evaluations=0,
            deterministic_leaf_evaluations=0,refined_nodes=0,blend_nodes=0,
            coarse_compression_fallbacks=0,
            neural_network_calls=0,boundary_batch_groups=0,padded_response_atoms=0,
            max_parent_child_moment_error=0.,max_parent_child_jacobian_error=0.,
            max_output_fine_jacobian_discrepancy=0.,
            derivative_discrepancy_exceeded=False,error_indicator_is_certificate=False,
            bandwidth=width,operation=operation,requires_native_acceptance=True,
            parameter_scale_supplied=parameter_scales is not None)
        if c.response_path == 'neural':
            with _stage(trace,'finite_cut_descriptors'):
                cut = finite_cut_tokens(surface,motion,origin=tree.origin,
                    length_scale=tree.side*tree.pitch,max_tokens=c.cut_token_count,
                    device=p.device,dtype=p.dtype)
        else:
            cut = None
        memo = {}
        child_selections = {}

        def constant(values):
            return PackedPrediction(tensor(values),None if dimension is None else tensor(np.zeros((10,dimension))),{})

        def add(items):
            value = tensor(np.zeros(10))
            jac = None if dimension is None else tensor(np.zeros((10,dimension)))
            for result in items:
                value = value+result.moments
                if jac is not None:
                    jac = None if result.jacobian is None else jac+result.jacobian
            return PackedPrediction(value,jac,{})

        def local_context(node_id,points):
            saved = cache.node_contexts.get(node_id)
            if saved is not None:
                local,version = saved
                if local._version != version:
                    raise ValueError('cached local material feature was modified')
            else:
                node = tree.node(node_id)
                frame = (node.center-tree.origin)/(tree.side*tree.pitch)
                raw = np.r_[tree.spatial_features(node_id,grid=2).reshape(80),frame,1./node.grid]
                with torch.no_grad() if self.training_scope == 'response' or cache.inference_only else nullcontext():
                    local = self.local_geometry(torch.tensor(raw.copy(),device=p.device,dtype=p.dtype))
                if local.is_inference():
                    with torch.inference_mode(False),torch.no_grad():
                        local = local.clone()
                size = local.numel()*local.element_size()
                if size+cache.nbytes <= c.feature_cache_bytes:
                    cache.node_contexts[node_id] = (local,local._version)
                    cache.node_context_bytes['bytes'] += size
            return cache.context(points)+local[None,:]

        def response_chunk(points,weights,model_path,node_id,response_width=None):
            support = width if response_width is None else response_width
            measure = prepare_measure(points,weights)
            with _stage(trace,'distance_stencil'):
                stencil = prepare_stencil(measure,surface,bandwidth=support,chunk_size=chunk)
            diagnostics['evaluated_atoms'] += stencil.total_nodes
            diagnostics['active_atoms'] += len(stencil.nodes)
            diagnostics['ambiguous_atoms'] += stencil.ambiguous_nodes
            with _stage(trace,'response_moment_jacobian'):
                if model_path != 'neural':
                    result = evaluate_stencil(stencil,operation=operation,vertex_velocity=motion,
                        model=self.response if model_path == 'legacy' else None,
                        context=cache.context if model_path == 'legacy' else None,
                        chunk_size=chunk,fused_contraction=True)
                    return PackedPrediction(result.moments.to(p.device),
                        None if result.jacobian is None else result.jacobian.to(p.device),result.diagnostics)
                retained = stencil.interior_moments if operation == 'intersection' else stencil.exterior_moments
                base = constant(retained)
                if stencil.ambiguous_nodes:
                    base = replace(base,jacobian=None)
                if not len(stencil.nodes):
                    return base
                signed = tensor(stencil.signed_distance)
                live = surface.tensor_vertices
                if live is not None and not stencil.ambiguous_nodes:
                    face = torch.as_tensor(surface.faces[stencil.face_indices],device=p.device)
                    displacement = (live-live.detach()).to(dtype=torch.float64,device=p.device)
                    delta = sum(tensor(stencil.barycentric[:,i:i+1])*displacement[face[:,i]] for i in range(3))
                    signed = signed+(tensor(stencil.motion_normal)*delta).sum(1)
                ds = stencil.distance_motion(motion) if motion is not None and not stencil.ambiguous_nodes else None
                point_tensor = torch.tensor(stencil.nodes.copy(),device=p.device,dtype=p.dtype)
                diagnostics['neural_network_calls'] += 1
                result = integrate_boundary_response(stencil.nodes,stencil.weights,signed,
                    bandwidth=support,model=self.response,context=local_context(node_id,point_tensor),
                    cut_tokens=cut.features,distance_motion=ds,
                    cut_motion=cut.motion if ds is not None else None,operation=operation,
                    point_origin=tree.origin,point_scale=tree.side*tree.pitch)
                return add((base,PackedPrediction(result.moments,result.jacobian,result.diagnostics)))

        def prime_coarse(node_ids):
            """Run one neural response per bounded same-level frontier wave.

            Tree traversal and finite-surface witnesses still run on CPU. This
            batches the actual neural tangent/reduction, not just a Python API.
            """
            if c.response_path != 'neural':
                return
            groups = {}
            for node_id in dict.fromkeys(node_ids):
                if (node_id,True) in memo:
                    continue
                try:
                    with _stage(trace,'coarse_material_rule'):
                        measure = tree.measure(node_id,max_atoms=c.max_coarse_atoms,chunk_size=chunk)
                except CoarseCubatureUnavailable:
                    diagnostics['coarse_compression_fallbacks'] += 1
                    with _stage(trace,'material_rule'):
                        memo[(node_id,True)] = add(response_chunk(x,w,'deterministic',node_id)
                            for x,w in tree.chunks(node_id,max_nodes=chunk,full_cell_rule=c.leaf_rule))
                    continue
                support = max(width,2*tree.node(node_id).radius*c.coarse_support_fraction)
                with _stage(trace,'distance_stencil'):
                    stencil = prepare_stencil(prepare_measure(measure.nodes,measure.weights),
                        surface,bandwidth=support,chunk_size=chunk)
                diagnostics['coarse_evaluations'] += 1
                diagnostics['evaluated_atoms'] += stencil.total_nodes
                diagnostics['active_atoms'] += len(stencil.nodes)
                diagnostics['ambiguous_atoms'] += stencil.ambiguous_nodes
                points = torch.tensor(stencil.nodes.copy(),device=p.device,dtype=p.dtype)
                context = (local_context(node_id,points) if len(points)
                           else p.new_empty((0,c.geometry_width)))
                groups.setdefault(support,[]).append((node_id,stencil,context))
            for group in groups.values():
                with _stage(trace,'response_moment_jacobian'):
                    results = evaluate_stencil_batch([row[1] for row in group],
                        [row[2] for row in group],self.response,cut.features,
                        cut_motion=cut.motion if motion is not None else None,
                        vertex_velocity=motion,operation=operation,
                        point_origin=tree.origin,point_scale=tree.side*tree.pitch,
                        max_padded_atoms=chunk)
                diagnostics['boundary_batch_groups'] += 1
                diagnostics['neural_network_calls'] += results.diagnostics['network_calls']
                diagnostics['padded_response_atoms'] += results.diagnostics['padded_atoms']
                for (node_id,_,_),value in zip(group,results):
                    memo[(node_id,True)] = value

        def evaluate_node(node_id, *, coarse):
            key = (node_id,coarse)
            if key in memo:
                return memo[key]
            if coarse and c.response_path == 'neural':
                prime_coarse([node_id])
                return memo[key]
            if coarse:
                try:
                    with _stage(trace,'coarse_material_rule'):
                        measure = tree.measure(node_id,max_atoms=c.max_coarse_atoms,chunk_size=chunk)
                except CoarseCubatureUnavailable:
                    # The decision depends only on static material. No material
                    # is discarded and no query-dependent discontinuous switch
                    # is added to the response.
                    diagnostics['coarse_compression_fallbacks'] += 1
                    with _stage(trace,'material_rule'):
                        value = add(response_chunk(x,w,'deterministic',node_id) for x,w in
                            tree.chunks(node_id,max_nodes=chunk,full_cell_rule=c.leaf_rule))
                else:
                    diagnostics['coarse_evaluations'] += 1
                    support = max(width,2*tree.node(node_id).radius*c.coarse_support_fraction)
                    value = response_chunk(measure.nodes,measure.weights,c.response_path,node_id,support)
            else:
                # Adaptive leaves are deterministic high-order references. The
                # explicit leaf arm retains learned response for E16 equivalence.
                path = c.response_path if execution == 'leaf' else 'deterministic'
                diagnostics['deterministic_leaf_evaluations'] += int(path == 'deterministic')
                with _stage(trace,'material_rule'):
                    parts = tree.chunks(node_id,max_nodes=chunk,full_cell_rule=c.leaf_rule)
                    value = add(response_chunk(x,w,path,node_id) for x,w in parts)
            memo[key] = value
            return value

        def prime_children(parent_ids):
            # The usual 32 -> 64 probe is one frontier, not thousands of
            # separate eight-child GPU launches. Deeper refinement remains lazy.
            groups = {}
            for parent_id in dict.fromkeys(parent_ids):
                if parent_id in child_selections:
                    continue
                ids = tree.children(parent_id)
                child_selections[parent_id] = []
                if ids:
                    support = max(width,2*tree.node(ids[0]).radius*c.coarse_support_fraction)
                    groups.setdefault(support,[]).extend((parent_id,child) for child in ids)
            boundary = []
            for support,group in groups.items():
                ids = [child for _,child in group]
                with _stage(trace,'hierarchy_classification'):
                    status = tree.classify_nodes(surface,ids,support,operation)
                diagnostics['visited_nodes'] += len(ids)
                diagnostics['classification_distance_queries'] += len(ids)
                for (parent_id,child),kind in zip(group,status):
                    child_selections[parent_id].append((child,int(kind)))
                    if kind == 0:
                        boundary.append(child)
            prime_coarse(boundary)

        def child_values(node_id):
            prime_children([node_id])
            values = []
            for child,kind in child_selections[node_id]:
                if kind == 1:
                    value = constant(tree.node(child).moments)
                elif kind == -1:
                    value = constant(np.zeros(10))
                else:
                    # The cheap probe must not unconditionally pay for the full
                    # fine deterministic integral before deciding to refine.
                    value = evaluate_node(child,coarse=True)
                values.append((child,int(kind),value))
            return values

        def refine(node_id, initial=None):
            if not tree.children(node_id):
                return evaluate_node(node_id,coarse=False)
            coarse = evaluate_node(node_id,coarse=True) if initial is None else initial
            children = child_values(node_id)
            probe = add(value for _,_,value in children)
            span = tree.side*tree.pitch
            scale = tree.node(node_id).moments[0]*span**np.array([0,1,1,1,2,2,2,2,2,2])
            delta = (probe.moments-coarse.moments)/tensor(scale)
            error = float(torch.linalg.vector_norm(delta).detach())
            diagnostics['max_parent_child_moment_error'] = max(diagnostics['max_parent_child_moment_error'],error)
            j_error = None
            if dimension is not None and coarse.jacobian is not None and probe.jacobian is not None:
                j_error = float(((probe.jacobian-coarse.jacobian)*tensor(parameter_scale)[None,:]/tensor(scale)[:,None]).abs().max().detach())
                diagnostics['max_parent_child_jacobian_error'] = max(diagnostics['max_parent_child_jacobian_error'],j_error)
            u = error**2/c.moment_tolerance**2
            if u <= c.blend_start:
                # This branch's gate is exactly zero with zero gate slope.
                if j_error is not None and j_error > c.jacobian_tolerance:
                    diagnostics['derivative_discrepancy_exceeded'] = True
                return coarse
            diagnostics['refined_nodes'] += 1
            fine = add(refine(child,value) if kind == 0 else value for child,kind,value in children)
            diagnostics['blend_nodes'] += int(u < c.blend_end)
            result = blend_moments(coarse,fine,probe,scale,tolerance=c.moment_tolerance,
                                    start=c.blend_start,end=c.blend_end)
            if dimension is not None and result.jacobian is not None and fine.jacobian is not None:
                discrepancy = float(((result.jacobian-fine.jacobian)*tensor(parameter_scale)[None,:]
                                     /tensor(scale)[:,None]).abs().max().detach())
                diagnostics['max_output_fine_jacobian_discrepancy'] = max(
                    diagnostics['max_output_fine_jacobian_discrepancy'],discrepancy)
                if discrepancy > c.jacobian_tolerance:
                    diagnostics['derivative_discrepancy_exceeded'] = True
            return result

        with torch.no_grad() if cache.inference_only else nullcontext():
            if execution != 'leaf':
                prime_coarse(selected.boundary_nodes)
            if execution == 'adaptive' and c.response_path != 'deterministic':
                prime_children(selected.boundary_nodes)
            partials = []
            for node_id in selected.boundary_nodes:
                if execution == 'adaptive' and c.response_path != 'deterministic':
                    partials.append(refine(node_id))
                else:
                    partials.append(evaluate_node(node_id,coarse=execution == 'coarse' and c.response_path != 'deterministic'))
            result = add((constant(cache.fixed_moments+selected.retained_moments),*partials))
            if self.training and c.response_path != 'deterministic' and torch.is_grad_enabled():
                zero = next(self.response.parameters()).reshape(-1)[0]*0.
                result = replace(result,moments=result.moments+zero,
                    jacobian=None if result.jacobian is None else result.jacobian+zero)
        diagnostics['solver_status'] = native_proposal_status(result,cache.fixed_moments)
        if dimension is None and diagnostics['solver_status'] == 'NATIVE_REQUIRED_DERIVATIVE':
            diagnostics['solver_status'] = 'MOMENT_ONLY_REQUIRES_NATIVE_ACCEPTANCE'
        if diagnostics['derivative_discrepancy_exceeded']:
            diagnostics['solver_status'] = 'NATIVE_REQUIRED_REFINEMENT'
        if dimension is not None and result.jacobian is None:
            diagnostics['solver_status'] = 'NATIVE_REQUIRED_DERIVATIVE'
        diagnostics['jacobian_source'] = 'same_moment_complete_first_tangent_chain'
        return PackedPrediction(result.moments,result.jacobian,diagnostics)

    def evaluate_operands(self, cache, operands, **kwargs):
        """Share one encoding across ragged queries; no unmeasured GPU batching claim."""
        return [self.evaluate_operand(cache,operand,**kwargs) for operand in operands]
