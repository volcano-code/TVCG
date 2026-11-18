# R3.0.2 — Real M69 corpus admission audit

## Known source state

The local M69 kit is registered at:

- root: /data/xvxinyao/Spin_M68_Trellis_Data_Kit
- source manifest: runs/m72-diag-inline-l5rblne3/source-manifest.json
- schema: m69.corpus.v1
- records: 104
- statuses: 67 qualified, 36 invalid_mesh, 1 pending_labels
- prior split counts: 80 train, 12 validation, 12 test

The M69 and M72 manifests previously cross-checked with 104 common record IDs. A prior local report found some source-manifest paths unresolved under the M72 diagnostic files directory while shared near-duplicate evidence files existed. Asset presence must therefore be rechecked against the authoritative combined70-v2 corpus root.

## New auditor

The R3.0.2 auditor inventories the corpus manifest and verifies case, geometry, labels, and label-audit artifacts by SHA-256.

It intentionally does not promote geometry_file (native voxel/interval cache) to an original surface mesh. It also does not infer CAD pin/support geometry, fixed shell mask, editable design-domain mask, cavity policy, or Faster quadric initialization.

## Local command

From the local kit:

    cd /data/xvxinyao/Spin_M68_Trellis_Data_Kit
    export TVCG=/data/xvxinyao/TVCG
    python "$TVCG/research/NeuralMoments/r3_m69_corpus_admission_v1.py" \
      --manifest "$PWD/runs/m72-diag-inline-l5rblne3/source-manifest.json" \
      --corpus-root "$PWD/data/combined70-v2" \
      --output "$TVCG/reports/r3/m69-corpus-admission.json"

If TVCG is stored elsewhere, change only TVCG. This command is read-only with respect to the source corpus.

## Evidence boundary / distance

This audit proves inventory and artifact hashes only. It cannot establish physical M69 admission. Current admitted counts remain zero for original surface meshes, pin/support, shell masks, design masks, Faster beta initialization, and complete physical cases.

Next: use the audit report to identify candidate records, inspect source task/mesh provenance, and bind one case only when every physical and numerical input has source-backed evidence.
