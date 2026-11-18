"""R3.1.5.4 topology/provenance audit for an externally produced tetrahedral solid.

This audit checks tetrahedral complex combinatorics and signed volume. It does NOT
prove tetrahedron interiors do not overlap, nor prove equivalence to a native
triangle boundary. Such equivalence requires an explicit reviewed mapping
certificate binding the exact source/native/independent assets; the report keeps
that evidence claim separate from the computed topology checks.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

SCHEMA="tvcg.r3.independent-geometry-integrity.v1"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def audit_tetrahedral_complex(vertices, tetrahedra, *, volume_tolerance=1e-14):
    v=np.asarray(vertices,dtype=np.float64); t=np.asarray(tetrahedra)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
        raise ValueError("vertices must be finite [N,3]")
    if t.ndim!=2 or t.shape[1]!=4 or len(t)==0 or t.dtype.kind not in "iu":
        raise ValueError("tetrahedra must be integer [T,4]")
    if (t<0).any() or (t>=len(v)).any(): raise ValueError("tetrahedron vertex index out of range")
    if any(len(set(map(int,row)))!=4 for row in t): raise ValueError("tetrahedron repeats a vertex")
    canonical=[tuple(sorted(map(int,row))) for row in t]
    if len(set(canonical))!=len(canonical): raise ValueError("duplicate tetrahedron detected")
    xyz=v[t]
    det=np.einsum("ij,ij->i",xyz[:,1]-xyz[:,0],
                  np.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,0]))
    signed=det/6.0
    if not np.isfinite(signed).all(): raise ValueError("non-finite tetrahedron volume")
    if np.any(np.abs(signed)<=volume_tolerance): raise ValueError("degenerate tetrahedron detected")
    if np.any(signed<0): raise ValueError("negative tetrahedron orientation; consistent positive orientation required")

    faces=defaultdict(list); edges=defaultdict(int); cell_neighbors=[set() for _ in range(len(t))]
    local_faces=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
    for ci,row in enumerate(t):
        for lf in local_faces:
            oriented=tuple(int(row[k]) for k in lf)
            key=tuple(sorted(oriented))
            faces[key].append((ci,oriented))
        for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            edges[tuple(sorted((int(row[a]),int(row[b]))))]+=1
    over=[k for k,x in faces.items() if len(x)>2]
    if over: raise ValueError(f"non-manifold face incidence >2 ({len(over)} faces)")
    boundary=[entries[0][1] for entries in faces.values() if len(entries)==1]
    for entries in faces.values():
        if len(entries)==2:
            (a,fa),(b,fb)=entries
            if fa != tuple(reversed(fb)) and not all(fa[i]==fb[(3-i)%3] for i in range(3)):
                # Check orientation by parity of permutations instead of relying on one reverse ordering.
                def parity(seq):
                    inv=sum(seq[i]>seq[j] for i in range(3) for j in range(i+1,3))
                    return inv%2
                if parity(fa)==parity(fb):
                    raise ValueError("interior face orientation is inconsistent")
            cell_neighbors[a].add(b);cell_neighbors[b].add(a)
    boundary_edges=defaultdict(int)
    for face in boundary:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            boundary_edges[tuple(sorted((a,b)))]+=1
    bad_boundary_edges=[e for e,n in boundary_edges.items() if n!=2]
    if bad_boundary_edges: raise ValueError(f"boundary is not a closed 2-manifold ({len(bad_boundary_edges)} bad edges)")
    seen=set();components=[]
    for start in range(len(t)):
        if start in seen: continue
        q=[start];seen.add(start);count=0
        while q:
            x=q.pop();count+=1
            for y in cell_neighbors[x]:
                if y not in seen:seen.add(y);q.append(y)
        components.append(count)
    used=np.unique(t)
    total_volume=float(signed.sum())
    return {"tetrahedron_count":int(len(t)),"used_vertex_count":int(len(used)),
      "unused_vertex_count":int(len(v)-len(used)),"boundary_face_count":int(len(boundary)),
      "interior_face_count":int(sum(len(x)==2 for x in faces.values())),
      "component_tetrahedron_counts":components,"component_count":len(components),
      "signed_volume_sum":total_volume,"minimum_tetrahedron_volume":float(signed.min()),
      "maximum_tetrahedron_volume":float(signed.max()),
      "topology_checks":{"positive_orientation":True,"no_duplicate_tetrahedra":True,
        "no_nonmanifold_faces":True,"closed_boundary_edges":True},
      "limitations":["does not detect overlapping tetrahedron interiors",
        "does not prove equivalence to a native triangle boundary",
        "does not establish shell/design/cavity semantics"]}

def audit_manifest(manifest_path):
    p=Path(manifest_path);root=p.parent;m=json.loads(p.read_text())
    if m.get("schema")!=SCHEMA:raise ValueError("unsupported geometry integrity manifest schema")
    required=("case_id","task_hash","geometry_hash","coordinate_frame","length_unit",
      "meters_per_unit","tetrahedra_file","tetrahedra_sha256","vertices_key","tetrahedra_key",
      "native_final_boundary_file","native_final_boundary_sha256",
      "source_outer_file","source_outer_sha256","fixed_shell_file","fixed_shell_sha256",
      "design_domain_file","design_domain_sha256","cavity_file","cavity_sha256",
      "mapping_certificate_file","mapping_certificate_sha256")
    missing=[k for k in required if k not in m]
    if missing:raise ValueError("manifest missing: "+", ".join(missing))
    assets={}
    for path_key,hash_key in (("tetrahedra_file","tetrahedra_sha256"),
      ("native_final_boundary_file","native_final_boundary_sha256"),
      ("source_outer_file","source_outer_sha256"),("fixed_shell_file","fixed_shell_sha256"),
      ("design_domain_file","design_domain_sha256"),("cavity_file","cavity_sha256"),
      ("mapping_certificate_file","mapping_certificate_sha256")):
        fp=root/m[path_key]
        if not fp.is_file():raise FileNotFoundError(fp)
        digest=sha256_file(fp)
        if digest!=str(m[hash_key]).lower():raise ValueError(f"{hash_key} mismatch")
        assets[path_key]={"path":str(fp),"sha256":digest}
    with np.load(root/m["tetrahedra_file"],allow_pickle=False) as z:
        if m["vertices_key"] not in z.files or m["tetrahedra_key"] not in z.files:
            raise KeyError("vertices/tetrahedra array key missing")
        vertices=np.array(z[m["vertices_key"]],copy=True)
        tetra=np.array(z[m["tetrahedra_key"]],copy=True)
    topology=audit_tetrahedral_complex(vertices,tetra)
    cert=json.loads((root/m["mapping_certificate_file"]).read_text())
    if cert.get("schema")!="tvcg.r3.native-independent-geometry-mapping.v1":
        raise ValueError("mapping certificate schema mismatch")
    cert_required=("case_id","task_hash","geometry_hash","native_final_boundary_sha256",
      "tetrahedra_sha256","source_outer_sha256","fixed_shell_sha256","design_domain_sha256",
      "cavity_sha256","coordinate_frame","length_unit","reviewer","review_source",
      "equivalence_method","equivalence_result")
    missing_cert=[k for k in cert_required if not cert.get(k)]
    if missing_cert:raise ValueError("mapping certificate missing: "+", ".join(missing_cert))
    for k in ("case_id","task_hash","geometry_hash","coordinate_frame","length_unit"):
        if cert[k]!=m[k]:raise ValueError(f"mapping certificate {k} mismatch")
    bindings={"native_final_boundary_sha256":"native_final_boundary_sha256",
      "tetrahedra_sha256":"tetrahedra_sha256","source_outer_sha256":"source_outer_sha256",
      "fixed_shell_sha256":"fixed_shell_sha256","design_domain_sha256":"design_domain_sha256",
      "cavity_sha256":"cavity_sha256"}
    for ck,mk in bindings.items():
        if cert[ck]!=m[mk]:raise ValueError(f"mapping certificate {ck} mismatch")
    if cert["equivalence_result"]!="reviewed_equivalent":
        raise ValueError("native/independent geometry equivalence not explicitly reviewed")
    if not isinstance(cert["reviewer"],str) or not cert["reviewer"].strip() or not cert["review_source"].strip():
        raise ValueError("mapping certificate reviewer/source missing")
    return {"schema":"tvcg.r3.geometry-integrity-report.v1","case_id":m["case_id"],
      "task_hash":m["task_hash"],"geometry_hash":m["geometry_hash"],
      "coordinate_frame":m["coordinate_frame"],"length_unit":m["length_unit"],
      "asset_hashes":{k:v["sha256"] for k,v in assets.items()},
      "topology_audit":topology,"native_independent_equivalence":{
        "certificate_sha256":assets["mapping_certificate_file"]["sha256"],
        "reviewer":cert["reviewer"],"review_source":cert["review_source"],
        "equivalence_method":cert["equivalence_method"],
        "reviewed_equivalent":True},
      "geometry_integrity_pass":True,
      "scope":"combinatorial tetrahedral topology and hash-bound reviewed equivalence certificate; no automatic geometric intersection proof"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();r=audit_manifest(a.manifest)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"case_id":r["case_id"],"geometry_integrity_pass":r["geometry_integrity_pass"],
      "topology_audit":r["topology_audit"],"output":str(out)},indent=2))
if __name__=="__main__":main()
