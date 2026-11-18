import json, tempfile, unittest, hashlib
from pathlib import Path
from research.NeuralMoments.r3_m69_corpus_admission_v1 import audit_corpus

class R3M69CorpusAdmissionTests(unittest.TestCase):
    def test_assets_do_not_imply_surface_mesh_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); files=root/"files"; files.mkdir()
            for name,data in (("case.npz",b"case"),("geom.npz",b"geometry"),
                              ("labels.npz",b"labels"),("audit.json",b"{}")):
                (files/name).write_bytes(data)
            def digest(name): return hashlib.sha256((files/name).read_bytes()).hexdigest()
            rec={"id":"m69-test","status":"qualified","design_qualified":True,
                 "case_file":"files/case.npz","case_sha256":digest("case.npz"),
                 "geometry_file":"files/geom.npz","geometry_sha256":digest("geom.npz"),
                 "labels_file":"files/labels.npz","labels_sha256":digest("labels.npz"),
                 "label_audit_file":"files/audit.json","label_audit_sha256":digest("audit.json")}
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps({"schema":"m69.corpus.v1","records":[rec]}))
            report=audit_corpus(manifest,root)
            self.assertEqual(report["qualified_design_count"],1)
            self.assertFalse(report["records"][0]["surface_mesh_present"])
            self.assertFalse(report["records"][0]["physical_admission"])
            self.assertEqual(report["admission_summary"]["physical_cases_admitted"],0)

    def test_missing_assets_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=root/"manifest.json"
            manifest.write_text(json.dumps({"schema":"m69.corpus.v1","records":[{
                "id":"missing","status":"qualified","design_qualified":True,
                "case_file":"files/no.npz","geometry_file":"files/no.npz",
                "labels_file":"files/no.npz","label_audit_file":"files/no.json"}]}))
            report=audit_corpus(manifest,root)
            self.assertEqual(len(report["records"][0]["missing_assets"]),4)

if __name__=="__main__": unittest.main()
