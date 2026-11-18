import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from research.NeuralMoments.r3_geometry_adapter_v1 import SCHEMA, load_geometry_bundle

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

class R3GeometryAdapterTests(unittest.TestCase):
    def _case(self, root, mask_dtype=bool):
        mesh=root/"mesh.npz"
        np.savez(mesh,vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]],float),
                 faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],dtype=np.int64))
        shell=root/"shell.npz"; s=np.zeros((4,4,4),dtype=mask_dtype); s[0]=True; np.savez(shell,mask=s)
        design=root/"design.npz"; d=np.zeros((4,4,4),bool); d[1:]=True; np.savez(design,mask=d)
        m={"schema":SCHEMA,"mesh_file":"mesh.npz","vertices_key":"vertices","faces_key":"faces",
           "shell_mask_file":"shell.npz","shell_mask_key":"mask",
           "design_mask_file":"design.npz","design_mask_key":"mask",
           "grid_shape":[4,4,4],"grid_origin":[0,0,0],"unit_token":"source_unit_unscaled",
           "mesh_sha256":digest(mesh),"shell_mask_sha256":digest(shell),"design_mask_sha256":digest(design)}
        p=root/"manifest.json"; p.write_text(json.dumps(m)); return p, m

    def test_explicit_assets_load_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,_=self._case(Path(tmp)); b=load_geometry_bundle(p)
            self.assertEqual(b.vertices.shape,(4,3)); self.assertTrue(b.validate_partition())
            self.assertEqual(b.unit_token,"source_unit_unscaled")

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,m=self._case(Path(tmp)); m["mesh_sha256"]="0"*64; p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"mesh_sha256 mismatch"): load_geometry_bundle(p)

    def test_non_boolean_mask_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,_=self._case(Path(tmp),mask_dtype=np.uint8)
            with self.assertRaisesRegex(ValueError,"must be bool"): load_geometry_bundle(p)

if __name__=="__main__": unittest.main()
