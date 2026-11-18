# R3.1.1 local native build recovery — pinned libigl

## Diagnosis

The CMake error is caused by an uninitialized Git submodule, not by the C++ compiler or the R3 changes.

The repository records:
- submodule path: libigl
- URL: https://github.com/libigl/libigl.git
- pinned gitlink: f06fb13014aadff7eb1d7423ac018620441a3657

CMake explicitly requires libigl/include/igl/read_triangle_mesh.h. Do not clone the latest libigl manually or change the pinned revision.

## Safe recovery

First inspect the working tree and submodule registration:

    cd /data/xvxinyao/TVCG
    git status --short
    git branch --show-current
    git submodule status

Initialize only the required submodule (QuadrANN is not needed for this CMake configure):

    git submodule sync -- libigl
    git submodule update --init --recursive libigl

Verify the checked-out revision and required header:

    test -f libigl/include/igl/read_triangle_mesh.h
    git -C libigl rev-parse HEAD
    git ls-tree HEAD libigl

Expected libigl revision:
f06fb13014aadff7eb1d7423ac018620441a3657

If the submodule update fails, preserve the output and stop; do not delete libigl or change its URL/revision.

## Configure in a clean, separate build directory

This avoids reusing the incomplete build directory created by the failed configure:

    cmake -S . -B build-r3-native \
      -DCMAKE_BUILD_TYPE=Release \
      -DMAKE_IT_STAND_BUILD_VIEWER=OFF \
      -DMAKE_IT_STAND_BUILD_PYTHON=ON \
      -DMAKE_IT_STAND_BUILD_SPIN_IT_FASTER=ON

The project defines the pybind target as _make_it_stand:

    cmake --build build-r3-native --target _make_it_stand --parallel 8

## Verify the extension

The extension is emitted under build-r3-native/python. Use the same Python interpreter that CMake selected:

    export PYTHONPATH="$PWD/python:$PWD/build-r3-native/python:$PWD/research/NeuralMoments"
    python -c 'import _make_it_stand; print(_make_it_stand.__file__); print(hasattr(_make_it_stand, "solve_spin_it_faster_fullrank_quadric"))'

Then run:

    export TVCG_SOURCE_COMMIT="$(git rev-parse HEAD)"
    python research/NeuralMoments/r3_native_build_preflight_v1.py \
      --build-dir build-r3-native \
      --expected-commit "$TVCG_SOURCE_COMMIT" \
      --output reports/r3/native-build-preflight.json

## Important distinction

The CMake build enables the canonical full-rank dual-monomial solver. This is the repository's reconstructed/strict profile and must not be described as bitwise equivalent to the unreleased author Matlab implementation. The legacy formula-reproduction entry is separate.

## What success means

A successful build/preflight proves the pinned dependency and native API are available locally. It does not prove real TRELLIS geometry admission, a Faster exact replay, or physical feasibility. Those remain R3.1.2 onward.
