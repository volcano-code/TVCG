import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from research.NeuralMoments.r3_independent_geometry_integrity_v1 import (
    SCHEMA, audit_tetrahedral_complex, audit_manifest
)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class R3GeometryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        self.t=np.array([[0,1,2,3]],np.int64)

    def test_single_tetra_is_closed_and_connected(self):
        r=audit_tetrahedral_complex(self.v,self.t)
        self.assertEqual(r["component_count"],1)
        self.assertEqual(r["boundary_face_count"],4)
        self.assertTrue(all(r["topology_checks"].values()))
        self.assertAlmostEqual(r["signed_volume_sum"],1/6)

    def test_duplicate_tetra_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate tetrahedron"):
            audit_tetrahedral_complex(self.v,np.vstack([self.t,self.t]))

    def test_negative_orientation_rejected(self):
        t=self.t.copy();t[0,[1,2]]=t[0,[2,1]]
        with self.assertRaisesRegex(ValueError,"negative tetrahedron orientation"):
            audit_tetrahedral_complex(self.v,t)

    def test_degenerate_tetra_rejected(self):
        v=self.v.copy();v[3]=[1,1,0]
        with self.assertRaisesRegex(ValueError,"degenerate"):
            audit_tetrahedral_complex(v,self.t)

    def test_nonfinite_vertex_rejected(self):
        v=self.v.copy();v[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            audit_tetrahedral_complex(v,self.t)

    def test_out_of_range_index_rejected(self):
        t=self.t.copy();t[0,3]=9
        with self.assertRaisesRegex(ValueError,"out of range"):
            audit_tetrahedral_complex(self.v,t)

    def test_manifest_requires_reviewed_asset_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mesh=root/"tetra.npz";np.savez(mesh,vertices=self.v,tetrahedra=self.t)
            assets={}
            for name,data in (("native.obj",b"native"),("outer.obj",b"outer"),
                ("shell.npz",b"shell"),("design.npz",b"design"),("cavity.obj",b"cavity")):
                p=root/name;p.write_bytes(data);assets[name]=sha(p)
            cert={"schema":"tvcg.r3.native-independent-geometry-mapping.v1",
              "case_id":"case","task_hash":"task","geometry_hash":"geom",
              "native_final_boundary_sha256":assets["native.obj"],
              "tetrahedra_sha256":sha(mesh),"source_outer_sha256":assets["outer.obj"],
              "fixed_shell_sha256":assets["shell.npz"],"design_domain_sha256":assets["design.npz"],
              "cavity_sha256":assets["cavity.obj"],"coordinate_frame":"world","length_unit":"m",
              "reviewer":"reviewer","review_source":"CAD mapping review",
              "equivalence_method":"independent surface/volume correspondence audit",
              "equivalence_result":"reviewed_equivalent"}
            cp=root/"mapping.json";cp.write_text(json.dumps(cert))
            m={"schema":SCHEMA,"case_id":"case","task_hash":"task","geometry_hash":"geom",
              "coordinate_frame":"world","length_unit":"m","meters_per_unit":1.0,
              "tetrahedra_file":"tetra.npz","tetrahedra_sha256":sha(mesh),
              "vertices_key":"vertices","tetrahedra_key":"tetrahedra",
              "native_final_boundary_file":"native.obj","native_final_boundary_sha256":assets["native.obj"],
              "source_outer_file":"outer.obj","source_outer_sha256":assets["outer.obj"],
              "fixed_shell_file":"shell.npz","fixed_shell_sha256":assets["shell.npz"],
              "design_domain_file":"design.npz","design_domain_sha256":assets["design.npz"],
              "cavity_file":"cavity.obj","cavity_sha256":assets["cavity.obj"],
              "mapping_certificate_file":"mapping.json","mapping_certificate_sha256":sha(cp)}
            mp=root/"manifest.json";mp.write_text(json.dumps(m))
            r=audit_manifest(mp)
            self.assertTrue(r["geometry_integrity_pass"])
            cert["equivalence_result"]="unreviewed"
            cp.write_text(json.dumps(cert));m["mapping_certificate_sha256"]=sha(cp);mp.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"not explicitly reviewed"):
                audit_manifest(mp)

if __name__=="__main__":unittest.main()
