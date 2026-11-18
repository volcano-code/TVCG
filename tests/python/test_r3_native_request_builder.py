import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from research.NeuralMoments.r3_native_request_builder_v1 import SCHEMA, build_request

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

class R3NativeRequestBuilderTests(unittest.TestCase):
    def test_builds_explicit_request_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mesh=root/"mesh.npz"
            v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]],float)
            f=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],np.int64)
            np.savez(mesh,vertices=v,faces=f)
            task=root/"task.json"; task.write_text(json.dumps({"task_hash":"task-1"}))
            pin=root/"pin.obj"; pin.write_bytes(b"pin-support")
            manifest={"schema":SCHEMA,"case_id":"case-1","geometry_hash":"geom-1","task_hash":"task-1",
              "task_record":"task.json","task_record_sha256":sha(task),
              "outer_mesh_file":"mesh.npz","outer_mesh_sha256":sha(mesh),
              "outer_vertices_key":"vertices","outer_faces_key":"faces",
              "design_mesh_file":"mesh.npz","design_mesh_sha256":sha(mesh),
              "design_vertices_key":"vertices","design_faces_key":"faces",
              "pin_support_file":"pin.obj","pin_support_sha256":sha(pin),
              "moment_metadata":{"coordinate_frame":"world","length_unit":"m","meters_per_unit":1.0,"density":1.0,"moment_order":["M","LX","LY","LZ","QXX","QYY","QZZ","QYZ","QXZ","QXY"]},"request_fields":{"fixed_shell_moments":[0.0]*10,"target_origin":[0,0,0],
                "target_axes":[[1,0,0],[0,1,0],[0,0,1]],"mode":"SPINNING_TOP",
                "material_density":1.0,"original_units_per_normalized_unit":1.0,
                "minimum_wall_thickness":0.0,"spinning_top_objective":"INERTIA_AND_LOWERING",
                "settings":{"equation_profile":"DUAL_MONOMIAL_RAW_V1","capture_evaluation_payload":True}},
              "semantic_review":{"status":"reviewed","reviewer":"human","source_reference":"source CAD"}}
            p=root/"manifest.json";p.write_text(json.dumps(manifest))
            out=build_request(p)
            self.assertEqual(out["request_status"],"assembled_unexecuted")
            self.assertFalse(out["native_execution"])
            self.assertEqual(len(out["request"]["outer_vertices"]),4)

    def test_missing_request_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task=root/"task.json";task.write_text('{"task_hash":"t"}')
            pin=root/"pin.obj";pin.write_bytes(b"pin")
            mesh=root/"mesh.npz";np.savez(mesh,vertices=np.ones((4,3)),faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]))
            m={"schema":SCHEMA,"case_id":"c","geometry_hash":"g","task_hash":"t",
              "task_record":"task.json","task_record_sha256":sha(task),
              "outer_mesh_file":"mesh.npz","outer_mesh_sha256":sha(mesh),"outer_vertices_key":"vertices","outer_faces_key":"faces",
              "design_mesh_file":"mesh.npz","design_mesh_sha256":sha(mesh),"design_vertices_key":"vertices","design_faces_key":"faces",
              "pin_support_file":"pin.obj","pin_support_sha256":sha(pin),
              "moment_metadata":{"coordinate_frame":"world","length_unit":"m","meters_per_unit":1.0,"density":1.0,"moment_order":["M","LX","LY","LZ","QXX","QYY","QZZ","QYZ","QXZ","QXY"]},"request_fields":{},"semantic_review":{"status":"reviewed","reviewer":"h","source_reference":"r"}}
            p=root/"m.json";p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"request_fields missing"):build_request(p)

if __name__=="__main__":unittest.main()
