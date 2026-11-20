# Native Spin-It initial-state export — independent Python verification

The native `spin-it-2014-initial-material-state-v1` diagnostics payload is independently checked by:

`research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/spin_initial_material_state.py`

The reader validates:
- schema and scope;
- grid dimensions, bounds, spacing and x-fast indexing;
- binary source/carveable/fixed/design masks and exact partition relations;
- orthonormal model-to-target transform;
- independent recomputation of all ten unit-density geometric moments in model and target frames;
- explicit non-SI semantics.

Test:

```bash
python -m unittest research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_spin_initial_material_state -v
```

The synthetic fixture is a 2x2x2 unit-density box. It checks total volume 8 and the analytic second moments. It does not substitute for running the native C++ Spin-It API regression on an actual artifact.
