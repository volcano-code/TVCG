import hashlib, json, tempfile, unittest
from pathlib import Path
from research.NeuralMoments.r3_real_geometry_admission_v1 import SCHEMA, admit_real_geometry

def put(root,name,data):
    p=root/name; p.write_bytes(data); return hashlib.sha256(data).hexdigest()

class R3RealGeometryAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.gm_hash=put(self.root,"geometry.json",json.dumps({"schema":"tvcg.r3.geometry-adapter.v1"}).encode())
        task={"task_hash":"task-123","axis":[0,0,1],"axis_origin":[0,0,0],"contact":[0,0,0]}
        self.task_hash=put(self.root,"task.json",json.dumps(task).encode())
        self.pin_hash=put(self.root,"pin.obj",b"source pin geometry")

    def tearDown(self): self.tmp.cleanup()

    def manifest(self):
        return {"schema":SCHEMA,"case_id":"trellis-case-01",
          "geometry_manifest":"geometry.json","geometry_manifest_sha256":self.gm_hash,
          "task_record":"task.json","task_record_sha256":self.task_hash,"task_hash":"task-123",
          "pin_support_file":"pin.obj","pin_support_sha256":self.pin_hash,
          "axis":[0,0,1],"axis_origin":[0,0,0],"contact_point":[0,0,0],
          "coordinate_frame":"source_mesh_frame","length_unit":"mm","meters_per_unit":0.001,
          "grid_to_world_linear":[[1,0,0],[0,1,0],[0,0,1]],
          "grid_to_world_translation":[0,0,0],
          "initial_material_policy":"full material in declared design domain",
          "fixed_shell_semantics":"source CAD shell annotation",
          "design_domain_semantics":"source CAD interior design volume",
          "pin_support_semantics":"separate pin component from CAD source",
          "semantic_review":{"status":"reviewed","reviewer":"reviewer-id","source_reference":"CAD source drawing rev A"}}

    def test_reviewed_metadata_admitted_but_physical_geometry_not_claimed(self):
        p=self.root/"admission.json"; p.write_text(json.dumps(self.manifest()))
        r=admit_real_geometry(p)
        self.assertTrue(r["metadata_admission"])
        self.assertFalse(r["physical_mesh_admission"])
        self.assertFalse(r["native_exact_geometry_checks_passed"])

    def test_contact_must_lie_on_axis(self):
        m=self.manifest(); m["contact_point"]=[1,0,0]
        p=self.root/"admission.json"; p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"does not lie"): admit_real_geometry(p)

    def test_task_frame_must_match(self):
        m=self.manifest(); m["axis"]=[1,0,0]
        p=self.root/"admission.json"; p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"differ from canonical TaskSpec"): admit_real_geometry(p)

    def test_missing_semantic_review_rejected(self):
        m=self.manifest(); m["semantic_review"]={"status":"pending"}
        p=self.root/"admission.json"; p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"explicitly reviewed"): admit_real_geometry(p)

    def test_pin_hash_mismatch_rejected(self):
        m=self.manifest(); m["pin_support_sha256"]="0"*64
        p=self.root/"admission.json"; p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"pin_support_sha256 mismatch"): admit_real_geometry(p)

if __name__=="__main__": unittest.main()
