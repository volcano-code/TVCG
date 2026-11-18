import hashlib, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from research.NeuralMoments.r3_replay_admission_guard_v1 import validate_replay_admission

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

class R3ReplayAdmissionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.assets={}
        for name,data in (("evaluator.json",b'{"settings":{}}'),("outer.npz",b"outer"),
                          ("design.npz",b"design"),("pin.obj",b"pin"),("task.json",b'{"task_hash":"task"}')):
            p=self.root/name;p.write_bytes(data);self.assets[name]={"path":str(p),"sha256":digest(p)}
        self.evaluator=self.root/"evaluator.json"
        self.init=SimpleNamespace(task_hash="task",geometry_hash="geom")
        self.request={"case_id":"case","geometry_hash":"geom","task_hash":"task"}
        self.envelope={"schema":"tvcg.r3.native-request.v1","request_status":"assembled_unexecuted",
          "request":self.request,"source_assets":{
            "outer_mesh_file":self.assets["outer.npz"],"design_mesh_file":self.assets["design.npz"],
            "pin_support_file":self.assets["pin.obj"],"task_record":self.assets["task.json"]}}
        self.admission={"schema":"tvcg.r3.physical-admission.v1","status":"physically_admitted",
          "physical_mesh_admission":True,"case_id":"case","geometry_hash":"geom","task_hash":"task",
          "task_record_sha256":self.assets["task.json"]["sha256"],
          "outer_mesh_sha256":self.assets["outer.npz"]["sha256"],
          "design_mesh_sha256":self.assets["design.npz"]["sha256"],
          "pin_support_sha256":self.assets["pin.obj"]["sha256"],
          "native_geometry_evidence_sha256":"native-evidence","reviewer":"reviewer",
          "review_source":"CAD review","evaluator_config_sha256":self.assets["evaluator.json"]["sha256"]}

    def tearDown(self): self.tmp.cleanup()
    def test_valid_physical_admission_authorizes(self):
        self.assertTrue(validate_replay_admission(self.envelope,self.init,self.admission,self.evaluator)["authorized"])
    def test_metadata_only_admission_rejected(self):
        a=dict(self.admission);a["physical_mesh_admission"]=False
        with self.assertRaisesRegex(ValueError,"metadata-only"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)
    def test_case_or_geometry_mismatch_rejected(self):
        a=dict(self.admission);a["geometry_hash"]="other"
        with self.assertRaisesRegex(ValueError,"geometry hash mismatch"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)
    def test_task_mismatch_rejected(self):
        a=dict(self.admission);a["task_hash"]="other"
        with self.assertRaisesRegex(ValueError,"TaskSpec hash mismatch"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)
    def test_source_mesh_hash_mismatch_rejected(self):
        a=dict(self.admission);a["outer_mesh_sha256"]="other"
        with self.assertRaisesRegex(ValueError,"outer_mesh_file hash"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)
    def test_asset_modified_after_assembly_rejected(self):
        Path(self.assets["outer.npz"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"content changed"):validate_replay_admission(self.envelope,self.init,self.admission,self.evaluator)
    def test_evaluator_hash_mismatch_rejected(self):
        a=dict(self.admission);a["evaluator_config_sha256"]="other"
        with self.assertRaisesRegex(ValueError,"evaluator config hash"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)

if __name__=="__main__":unittest.main()
