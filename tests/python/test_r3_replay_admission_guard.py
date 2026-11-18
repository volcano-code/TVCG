import hashlib, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from research.NeuralMoments.r3_replay_admission_guard_v1 import validate_replay_admission

class R3ReplayAdmissionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.evaluator=self.root/"evaluator.json"; self.evaluator.write_text('{"settings":{}}')
        self.ehash=hashlib.sha256(self.evaluator.read_bytes()).hexdigest()
        self.init=SimpleNamespace(task_hash="task",geometry_hash="geom")
        self.request={"case_id":"case","geometry_hash":"geom","task_hash":"task"}
        self.envelope={"schema":"tvcg.r3.native-request.v1","request_status":"assembled_unexecuted",
          "request":self.request,"source_assets":{
            "outer_mesh_file":{"sha256":"outer"},
            "design_mesh_file":{"sha256":"design"}}}
        self.admission={"schema":"tvcg.r3.physical-admission.v1","status":"physically_admitted",
          "physical_mesh_admission":True,"case_id":"case","geometry_hash":"geom","task_hash":"task",
          "outer_mesh_sha256":"outer","design_mesh_sha256":"design","pin_support_sha256":"pin",
          "native_geometry_evidence_sha256":"native-evidence","reviewer":"reviewer",
          "review_source":"CAD review","evaluator_config_sha256":self.ehash}

    def tearDown(self): self.tmp.cleanup()

    def test_valid_physical_admission_authorizes(self):
        r=validate_replay_admission(self.envelope,self.init,self.admission,self.evaluator)
        self.assertTrue(r["authorized"])

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

    def test_evaluator_hash_mismatch_rejected(self):
        a=dict(self.admission);a["evaluator_config_sha256"]="other"
        with self.assertRaisesRegex(ValueError,"evaluator config hash"):validate_replay_admission(self.envelope,self.init,a,self.evaluator)

if __name__=="__main__":unittest.main()
