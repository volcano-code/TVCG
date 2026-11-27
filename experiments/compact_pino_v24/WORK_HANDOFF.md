# Compact PINO V2.4 — Work A / Work B handoff

Updated: 2026-11-27 (Asia/Shanghai). Branch: research/compact-pino-v24-cut-conditioned.

Use the commit containing this file as the delivered revision; record git rev-parse HEAD after checkout. The reviewed starting revision was 558a24df380b7e7120a6072c678885b218e1eb7b. Production E16 / V2.3 remains the baseline.

## Status

The branch contains the standalone active-token cut-attention primitive, strict architecture contract and seven CPU tests. This update adds higher-order-safe math attention for Jacobian-loss training. A full V2.4 encoder / stencil / M-J integration, target-GPU training and native physical acceptance are still required. No trained V2.4 checkpoint or speedup is claimed.

## Execute these guides

1. [Work A: frozen data identities, 24-query smoke pack, legal cut inputs and targeted labels](WORK_A_ACTION_GUIDE.md)
2. [Work B: integration, numerical gates, staged pilots, training and end-to-end acceptance](WORK_B_ACTION_GUIDE.md)
3. [Proposed architecture and training configuration](TRAINING_RECOMMENDATION.yaml)
4. [Update summary, evidence, scope and goal assessment](UPDATE_SUMMARY_20261127.md)

The YAML is a proposal requiring integration into the training driver, not a current executable CLI configuration. Start admitted research training after implementation correctness and real CUDA updates pass; do not demand that an untrained model already pass final native physical acceptance. Sparse Full-J coverage limits the claims that can be made, but does not invalidate otherwise qualified M/R3/R8 training. Missing protected denominators are not PASS for the original full-protocol long-training GO decision.

The detailed guides replace earlier unconditional zero-branch identity claims across grid changes, Boolean-input allowances, and blanket interpretations of J errors as missing chain rules. Keep data qualification, implementation correctness, model accuracy and final native solve acceptance as distinct decisions.

