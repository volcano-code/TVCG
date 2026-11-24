"""Read the frozen panels and their actual conditioning grids on the archive machine."""
import json
import hashlib
import sys
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ADAPTER = (Path(__file__).resolve().parents[3] /
           "research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter")
sys.path.insert(0, str(ADAPTER))
from v21_native import RawChart


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"archive digest mismatch at {path}")
    return path


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stage:
        temporary = Path(stage.name)
        try:
            json.dump(data, stage, default=plain, ensure_ascii=False, allow_nan=False, indent=1)
            stage.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_npz(path, **arrays):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stage:
        temporary = Path(stage.name)
        try:
            np.savez_compressed(stage, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


@dataclass
class Probe:
    id: str
    query: np.ndarray
    stratum: str
    origin: dict
    chart: RawChart | None = None
    reference: dict | None = None
    opposite: bool = False


@dataclass
class Geometry:
    id: str
    request: object
    normalization: dict
    physical: np.ndarray
    rle: Path
    split: str
    hashes: dict
    probes: list = field(default_factory=list)


def label_ref(path, index, labels):
    return dict(file=str(path), sha256=sha(path), index=int(index), query_id=str(labels['query_id'][index]),
                historical_masks={k:int(labels[k][index]) for k in
                            ('moment_valid','derivative_valid','restricted_valid','restricted_dimension',
                             'ambient_event_valid','restricted_event_valid','source_restricted_dimension',
                             'source_restricted_valid') if k in labels})


class Registry:
    def __init__(self, alignment, corpus, physics):
        self.alignment = Path(alignment)
        self.corpus = Path(corpus)
        self.physics = Path(physics)
        self.solver = self.alignment.parent / "phase3_1b/solver"
        pinned = read(Path(__file__).with_name('host_inputs.json'))
        for group, root in [('alignment',self.alignment),('solver',self.solver),('physics',self.physics),
                            ('deployment_absolute',Path('/'))]:
            for name, digest in pinned[group].items():
                verified(dict(path=root/name,sha256=digest))
        sys.path.insert(0, str(self.alignment))
        import panel
        self.panel = panel
        self.strict_release = self.corpus / "releases/four_split_delivery_392/manifest.json"
        self.strict_manifest = verified(read(self.strict_release)["views"]["development"])
        self.strict = {r["geometry_id"]: r for r in read(self.strict_manifest)["records"]}
        self.train_manifest = self.corpus / "manifests/train_manifest.json"
        self.strict_train_manifest = verified(read(self.strict_release)["views"]["train"])
        train_records = read(self.strict_train_manifest)["records"]
        # First publisher-ordered S and N geometry. These are training examples only.
        self.train_examples = {s["geometry_id"]:s for s in (
            next(r for r in train_records if r['geometry_id'].startswith('S')),
            next(r for r in train_records if r['geometry_id'].startswith('N')))}

    def noodles(self):
        req, norm, physical = self.panel.paper_input("noodles")
        trajectory = self.alignment / 'reference/trajectory_M_J.json'
        original = self.alignment.parent / 'noodles256_cut_audit/trajectory_M_J.json'
        solver_journal = self.solver / 'results/noodles/native/seed-00.progress.jsonl'
        truth = read(trajectory)
        grid_record = read(self.solver / 'grids/noodles/identity.json')['levels']['256']
        grid = verified(grid_record)
        if (self.panel.meshhash(req.design_vertices, req.design_faces) != truth['registered_mesh_sha256']
                or sha(trajectory) != sha(original) or sha(grid) != truth['rle_sha256']
                or sha(solver_journal) != truth['journal_sha256']):
            raise ValueError('frozen noodles trajectory/mesh/grid/journal integrity error')
        geom = Geometry("noodles", req, norm, physical, grid, "fixed_benchmark",
                        {str(p):sha(p) for p in [grid,trajectory,solver_journal]})
        for row in truth['rows']:
            ref = row['native']
            geom.probes.append(Probe(f'noodles_{row["call"]:02d}',np.array(row['query']),"trajectory",
                                     dict(file=str(trajectory),sha256=sha(trajectory),call=row['call']),
                                     reference=ref))
        return geom

    def registered(self, key):
        panel = self.panel
        if key in self.strict or key in self.train_examples:
            record = (self.strict if key in self.strict else self.train_examples)[key]
            repfile = self.corpus / record['split'] / key / "representation.json"
            rep = read(repfile)
            raw = verified(rep['mesh'])
            if sha(raw) != record['raw_sha256'] or rep['frame'] != record['coordinate_convention']:
                raise ValueError(f'design mesh or frame mismatch for {key}')
            import trimesh
            mesh = trimesh.load(raw, process=False, maintain_order=True)
            physical = np.array(mesh.vertices)
            frame = record['coordinate_convention']
            norm = dict(origin=frame['center'], L=frame['length_scale'])
            mesh.vertices = (physical - np.asarray(norm['origin'])) / norm['L']
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError(f'open or inconsistent signed mesh for {key}')
            v, f = mesh.vertices.tolist(), mesh.faces.tolist()
            req = panel.spin.SpinItFasterRequest(v,f,v,f,fixed_shell_moments=tuple(record['fixed_moments']),
                            material_density=record['density'],original_units_per_normalized_unit=norm['L'])
            grid = verified(record['geometry'])
            if grid != verified(rep['rle256']):
                raise ValueError('strict grid registration mismatch')
            files = [repfile, self.strict_release, self.strict_manifest if key in self.strict else self.strict_train_manifest, raw, grid]
        else:
            record = panel.by_id[key]
            if record['split'] not in ('train', 'development'):
                raise ValueError(f'geometry outside native replay registration: {key}')
            req, norm, physical = panel.registered(record)
            grid = verified(dict(path=record['geometry_file'],sha256=record['geometry_sha256']))
            files = [record['raw_file'],grid]
            if key == 'T2':
                entry = next(r for r in read(self.train_manifest)['records'] if r['geometry_id'] == key)
                if entry['raw_sha256'] != record['raw_sha256']:
                    raise ValueError('T2 parent mismatch')
                grid = verified(entry['rle256'])
                files += [self.train_manifest,grid]
                record = entry
        labs = verified(dict(path=record['labels_file'],sha256=record['labels_sha256']))
        files.append(labs)
        geom = Geometry(key,req,norm,physical,grid,record['split'],{str(p):sha(p) for p in files})
        return geom, record, labs

    def example(self, key):
        geom, record, labs = self.registered(key)
        with np.load(labs, allow_pickle=False) as z:
            if key in self.strict:
                # The compact view withholds the basis but preserves original query IDs and source rank.
                orig_path = verified(record["original_labels"])
                geom.hashes[str(orig_path)] = sha(orig_path)
                with np.load(orig_path, allow_pickle=False) as orig:
                    ix_by_id = {str(s):i for i,s in enumerate(orig['query_id'])}
                    for dim, surface, stratum in [(3,'PLANE','plane'),(8,'PARABOLOID','parabolic')]:
                        selected = ((z['stratum']==stratum) & (z['source_restricted_dimension']==dim)
                                    & z['source_restricted_valid'])
                        for i in np.flatnonzero(selected):
                            oid = ix_by_id[str(z['query_id'][i])]
                            if not np.array_equal(z['queries'][i],orig['queries'][oid]):
                                raise ValueError('preserved source chart query mismatch')
                            basis = orig['restricted_basis'][oid,:,:dim]
                            origin = label_ref(labs,i,z)
                            origin['source_chart'] = label_ref(orig_path,oid,orig)
                            geom.probes.append(Probe(f'{key}_R{dim}_{i:03d}',z['queries'][i].copy(),stratum,
                                                     origin, RawChart(surface,basis)))
                if key == 'combB':
                    audit = read(self.physics / 'formal_dev392.json')
                    i = int(audit['bank_index'])
                    if not np.array_equal(audit['query'],z['queries'][i]):
                        raise ValueError('combB registered query mismatch')
                    self.add_pair(geom, 'combB_hyperbolic', z['queries'][i], 'hyperbolic',
                                  label_ref(labs,i,z), audit)
                return geom
            if key in self.train_examples:
                for role in ('ellipsoid','hyperbolic','mixed','plane','parabolic'):
                    indices = np.flatnonzero((z['stratum']==role) & z['moment_valid'])
                    if not len(indices):
                        continue
                    i = int(indices[0])
                    chart = None
                    if z['restricted_valid'][i] and role in ('plane','parabolic'):
                        d = 3 if role == 'plane' else 8
                        chart = RawChart('PLANE' if d==3 else 'PARABOLOID',z['restricted_basis'][i,:,:d])
                    geom.probes.append(Probe(f'{key}_registered_{role}',z['queries'][i].copy(),role,label_ref(labs,i,z),chart))
                mass = z['moments'][:,0] / z['full_moments'][0]
                for i in np.flatnonzero(z['moment_valid'] & z['derivative_valid'] & (mass>0) & (mass<.01)):
                    geom.probes.append(Probe(f'{key}_registered_near_zero_{i:03d}',z['queries'][i].copy(),
                                             str(z['stratum'][i]),label_ref(labs,i,z)))
                return geom
            if key == 'T2':
                audit = read(self.physics / 'data.json')
                i = int(audit['index'])
                if not np.array_equal(audit['probes']['plus']['q_raw'],z['queries'][i]):
                    raise ValueError('T2 registered query mismatch')
                self.add_pair(geom, 'T2_ellipsoid', z['queries'][i], 'ellipsoid', label_ref(labs,i,z), audit)
                p = self.alignment / 'supplement/T2/middle_HYPERBOLOID.json'
                self.add_pair(geom, 'T2_hyperbolic', np.asarray(read(p)['label']['query']), 'hyperbolic',
                              dict(file=str(p),sha256=sha(p),role='historical train supplement'),
                              read(self.physics / 'hyperboloid.json'))
                geom.probes.append(Probe("T2_plane",z['queries'][0].copy(),"plane",label_ref(labs,0,z),
                                         RawChart('PLANE',z['restricted_basis'][0,:,:3])))
                # Frozen, valid in-volume and rejected endpoint tests include a registered host mesh.
                extremes = self.physics / 'extreme.json'
                for name, row in read(extremes)['probes'].items():
                    geom.probes.append(Probe(f"T2_endpoint_{name}",np.asarray(row['q_raw']),"endpoint",
                                             dict(file=str(extremes),sha256=sha(extremes),key=name),reference=row))
                for p in sorted((self.alignment / 'supplement/T2').glob('*.json')):
                    if p.name == 'middle_HYPERBOLOID.json':
                        continue   # This exact query is already T2_hyperbolic_plus.
                    old = read(p)
                    ref = old['label']
                    geom.probes.append(Probe(f'T2_{p.stem}',np.array(ref['query']),old['native_family'],
                                             dict(file=str(p),sha256=sha(p),band=old['band']),
                                             reference=dict(defined=True,M=ref['moments'],J=ref['jacobian'])))
                return geom
            if key in self.panel.cross_ids[:2]:
                for role in ('ellipsoid','hyperbolic','mixed','plane','parabolic'):
                    i = int(np.flatnonzero(z['stratum'] == role)[0])
                    chart = None
                    if role in ('plane','parabolic'):
                        dim = 3 if role == 'plane' else 8
                        chart = RawChart('PLANE' if dim == 3 else 'PARABOLOID',z['restricted_basis'][i,:,:dim])
                    ref_file = self.alignment / 'cross' / key / (role+'.json')
                    ref = read(ref_file)
                    if not np.array_equal(z['queries'][i], ref['target_query']):
                        raise ValueError('development chart/query frame mismatch')
                    reference = (dict(defined=True,M=ref['reference_M'],J=ref['reference_J'])
                                 if ref['native_defined'] else dict(defined=False,diagnostic=ref['native_rejection']['reason']))
                    geom.probes.append(Probe(f'{key}_{role}',z['queries'][i].copy(),role,
                                             label_ref(labs,i,z),chart,reference))
                return geom
        # Every sample is a frozen independently measured train-only supplement.
        for p in sorted((self.alignment / 'supplement' / key).glob('*.json')):
            old = read(p)
            if not old['accepted'] or old['split'] != 'train' or old['geometry_id'] != key:
                raise ValueError('supplement scope mismatch')
            ref = old['label']
            geom.probes.append(Probe(f'{key}_{p.stem}',np.array(ref['query']),old['native_family'],
                                     dict(file=str(p),sha256=sha(p),band=old['band']),
                                     reference=dict(defined=True,M=ref['moments'],J=ref['jacobian'])))
        return geom

    @staticmethod
    def add_pair(geom, prefix, query, stratum, origin, audit):
        for suffix, scale in [('plus',1.),('minus',-1.),('scale2',2.)]:
            row = audit['probes'][suffix]
            q = scale * np.array(query)
            if not np.array_equal(q,row['q_raw']):
                raise ValueError('archived query differs from registered sign/scale control')
            geom.probes.append(Probe(f'{prefix}_{suffix}',q,stratum,dict(origin,transform=scale),
                                     reference=row,opposite=suffix=='plus'))
