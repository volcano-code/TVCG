# R3.1.1 — Native build integration preflight

The native Python facade calls the pybind extension make_it_stand._make_it_stand, with the full-rank Spin-It Faster function solve_spin_it_faster_fullrank_quadric.

The preflight checks extension import and required symbols. If a CMake build directory is supplied, it checks MAKE_IT_STAND_BUILD_PYTHON=ON and MAKE_IT_STAND_BUILD_SPIN_IT_FASTER=ON and records available compiler/build metadata from CMakeCache.txt. This is an environment preflight, not a native geometry run. GitHub CI uses fake modules and does not claim the extension was built in CI.

## Local procedure

    cd /data/xvxinyao/TVCG
    export PYTHONPATH="$PWD/python:$PWD/research/NeuralMoments"
    export TVCG_SOURCE_COMMIT="$(git rev-parse HEAD)"
    python research/NeuralMoments/r3_native_build_preflight_v1.py --build-dir build --expected-commit "$TVCG_SOURCE_COMMIT" --output reports/r3/native-build-preflight.json

If the CMake build directory differs from build, supply its path. The extension must be importable in the selected Python environment.

## Expected CMake configuration

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMAKE_IT_STAND_BUILD_PYTHON=ON -DMAKE_IT_STAND_BUILD_SPIN_IT_FASTER=ON
    cmake --build build --target _make_it_stand --parallel 8

Check cmake --build build --target help if the configured pybind target has a different spelling.

## Remaining gap

A successful preflight proves only that the intended native API is importable and CMake options are enabled. R3.1.2 still requires a real admitted geometry and source-backed task/init records. TRELLIS data generation remains owned by Work.
