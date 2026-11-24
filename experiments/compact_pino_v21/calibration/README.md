# Training-side native Newton sensitivity

The proposed 1% moment and 5% directional Jacobian limits remain empirical screening criteria. On the genuine S001 starting state, a 2.0994% change to a single Jacobian row preserved the starting moments exactly and made the original compiled Newton solver terminate after one evaluation. Its bordered matrix rank fell from 13 to 12 and remained 12 after reversible equilibration. The original evaluator accepted the first trial at that state. The bounded smooth perturbation and both actual matrices are in the receipts.

Eight seed-0 prefixes used registered T2, N002 and S001 training meshes, their existing static maximum-mass request, and the original strict spherical Newton implementation. All 21 scored trials were accepted. Their smallest residual decrease was 0.0036271. Every fixed-trial decision survived all independent componentwise 1% moment errors. None of the prefixes produced a scored backtrack or a converged solve. The published training registrations lack a spinning contact pose. The trajectory catalogue records no admitted P4 source for T2, so these observations establish local static sensitivity only.

Eight additional T2 proposals used smooth (M,J) perturbations within the stated budgets and freshly reconstructed Cherchi geometry at the resulting coordinates. All eight decreased the true residual. For the second original Newton iteration, the decrease changed from 0.09874 to between 0.02790 and 0.05475. Centered native finite differences of the first raw tangent had scaled errors of 3.68 × 10⁻⁷ and 9.20 × 10⁻⁸ at steps 10⁻⁴ and 5 × 10⁻⁵.

`evidence/threshold_sensitivity.json` gives all source hashes, starts, terminal states and conclusions. `evidence/native_training_receipts.tar.gz` contains the 12 named JSON members, including the complete original solver event payloads, physical evaluations and the independent decision reconstruction. The summary authenticates each member. Problem inputs are the registered physical train meshes and native profile on the archive host.

To reproduce after the V2.1 native exporter has produced the three training geometry manifests and registered probe rows, run from the repository root:

```bash
bash experiments/compact_pino_v21/calibration/run.sh /tmp/tvcg-v21-calibration-replay /tmp/tvcg-v21-native
```

`CALIBRATION_CPUS` defaults to `8-11,24-27` on the archive host. The CLI writes the fresh receipts and summary into the first argument's `evidence` subdirectory. Timings and paths are recorded, so compare scientific trace hashes, status, scores and ranks across runs.
