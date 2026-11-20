# Unified operator memory profile

Device `cuda`. Status `complete`. Wall time 138.511 s.
Peak allocated 10.125 GiB. Peak reserved 10.195 GiB. Maximum sampled RSS 4.049 GiB.

CUDA allocator peaks merge adjacent synchronized segments, including nested scopes. Stage times are inclusive, phase times exclusive. RSS is current process RSS sampled at boundaries. Repeated stages aggregate counts, durations, first entry, last exit and maxima. Named tensor observations count logical elements each time objects are observed. Fine-grained recompute regions measure host wall time and current-stream CUDA event time inside their parent phase; their times may overlap and must not be added to wall time. A missing timing region is unmeasured. The Backward stage covers the entire autograd call. Named replay regions only cover recomputation, not the surrounding gradient kernels. Adjoint spans run from first observed output gradient to last observed input or parameter gradient inside the training backward. Shared inputs/parameters can extend and overlap spans. These are inclusive gradient-arrival spans and must not be summed. I/O, optimizer update, and immutable metadata have no parameter backward.

| Phase | Exclusive seconds |
|---|---:|
| other | 3.1031 |
| cache | 1.1744 |
| forward | 51.7950 |
| backward | 36.8290 |
| optimizer | 0.0339 |
| verification | 45.5756 |

| Recompute region | Calls | CPU wall s | CUDA stream s |
|---|---:|---:|---:|
| cross_scale_recompute | 11 | 0.1018 | 0.1015 |
| attention_key_value_projection | 1151 | 0.0332 | 0.0383 |
| attention_scores_and_directions | 4604 | 1.3026 | 0.2889 |
| Moment Operator | 4604 | 13.9538 | 17.2870 |
| sqno_fusion_recompute | 1151 | 0.2754 | 0.0841 |
| sqno_gnn_layer_recompute | 2302 | 0.8223 | 0.7769 |
| sqno_gnn_initial_recompute | 1151 | 0.5172 | 0.5003 |
| transolver_deslice_recompute | 72 | 0.0308 | 0.0315 |
| transolver_pool_recompute | 72 | 0.0388 | 0.0560 |
| pcraft_gno_recompute | 2301 | 1.0144 | 0.9729 |
| pcraft_interval_recompute | 2301 | 1.9062 | 1.8653 |
| pcraft_scale_recompute | 3 | 0.0686 | 0.2644 |

| Training adjoint span | Completed / started | Parameter gradients | CPU wall s | CUDA span s |
|---|---:|---:|---:|---:|
| transolver | 1 / 1 | 22 | 0.2680 | 0.2826 |
| pcraft_native_encoding | 1 / 1 | 168 | 8.6108 | 8.6107 |
| sqno_gnn | 1 / 1 | 14 | 3.4793 | 3.4768 |
| sqno_fusion | 1 / 1 | 10 | 12.3936 | 12.3281 |
| m80_coefficient_jacobian | 1 / 1 | 6 | 4.4511 | 4.3840 |
| Moment Operator | 1 / 1 | 0 | 20.7197 | 20.7867 |
| geometry_query_jacobian | 1 / 1 | 13 | 27.6527 | 27.6548 |
| cross_scale_attention | 1 / 1 | 14 | 27.6584 | 27.6583 |
| Loss | 1 / 1 | 0 | 23.2414 | 23.3082 |

| Stage | Phase | Calls | Inclusive s | Peak allocated GiB | Peak reserved GiB | Sampled RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| rle_decode | other | 1 | 1.8491 | 0.0226 | 0.0410 | 1.2318 |
| training_batch_transfer | other | 1 | 0.0007 | 0.0227 | 0.0410 | 1.2318 |
| geometry_metadata_cache | cache | 1 | 1.1739 | 0.0227 | 0.0410 | 1.4831 |
| one_optimizer_update | forward | 1 | 88.6580 | 10.1253 | 10.1953 | 3.9984 |
| encode_geometry | forward | 1 | 29.3640 | 5.1724 | 6.4648 | 3.5865 |
| pcraft_native_encoding | forward | 1 | 19.6290 | 4.5465 | 5.4316 | 2.8937 |
| transolver | forward | 1 | 0.1649 | 4.5465 | 5.4316 | 2.8936 |
| sqno_gnn | forward | 1 | 6.9931 | 5.1505 | 6.4648 | 3.3529 |
| sqno_fusion | forward | 1 | 0.4578 | 4.8204 | 6.4648 | 3.4186 |
| sqno_quadrature | forward | 1 | 2.0297 | 5.0495 | 6.4648 | 3.5864 |
| train_step | forward | 1 | 59.2936 | 10.1253 | 10.1953 | 3.9984 |
| m80_coefficient_jacobian | forward | 1 | 4.0222 | 5.0754 | 6.4648 | 3.6088 |
| Moment Operator | forward | 1 | 17.0083 | 5.3842 | 6.4648 | 3.7573 |
| geometry_query_jacobian | forward | 1 | 0.8348 | 5.3711 | 6.4648 | 3.7925 |
| geometry_physics_attention | forward | 1 | 0.8344 | 5.3711 | 6.4648 | 3.7925 |
| attention_moments | forward | 1 | 0.8312 | 5.3711 | 6.4648 | 3.7925 |
| cross_scale_attention | forward | 1 | 0.1188 | 5.3924 | 6.4648 | 3.7989 |
| Loss | forward | 1 | 0.0058 | 5.3920 | 6.4648 | 3.7991 |
| Backward | backward | 1 | 36.8290 | 10.1253 | 10.1953 | 3.9206 |
| attention_moments | backward | 1 | 2.4783 | 6.2489 | 6.7578 | 3.8710 |
| Optimizer Update | optimizer | 1 | 0.0339 | 1.5505 | 10.1953 | 3.9984 |
| cache_invalidation_check | cache | 1 | 0.0006 | 1.5312 | 10.1953 | 4.0011 |
| checkpoint_save | other | 1 | 0.7846 | 0.0965 | 10.1953 | 4.0011 |
| checkpoint_resume | other | 1 | 0.1686 | 0.1578 | 10.1953 | 4.0012 |
| checkpoint_prediction_comparison | verification | 1 | 45.5756 | 4.5636 | 10.1953 | 4.0490 |
| pcraft_native_encoding | verification | 2 | 7.0715 | 4.5636 | 10.1953 | 4.0489 |
| transolver | verification | 2 | 0.2124 | 4.5636 | 10.1953 | 4.0489 |
| sqno_gnn | verification | 2 | 2.1957 | 2.5972 | 10.1953 | 4.0489 |
| sqno_fusion | verification | 2 | 0.1790 | 1.9687 | 10.1953 | 4.0489 |
| sqno_quadrature | verification | 2 | 0.0527 | 2.1978 | 10.1953 | 4.0489 |
| m80_coefficient_jacobian | verification | 2 | 0.0154 | 1.5940 | 10.1953 | 4.0489 |
| Moment Operator | verification | 2 | 34.1443 | 1.9019 | 10.1953 | 4.0489 |
| geometry_query_jacobian | verification | 2 | 1.7668 | 1.6055 | 10.1953 | 4.0489 |
| geometry_physics_attention | verification | 2 | 1.7663 | 1.6055 | 10.1953 | 4.0489 |
| attention_moments | verification | 2 | 1.7627 | 1.6055 | 10.1953 | 4.0489 |
| cross_scale_attention | verification | 2 | 0.0178 | 1.5943 | 10.1953 | 4.0489 |

## Batch and update

```json
{
  "geometry_id": "N002",
  "manifest_sha256": "a0f914df030228b0f7fc930141b97968238250e43dd671e6f64e88ed882e42ee",
  "geometry_sha256": "bac895b5af674119bc20c02948d364710dac04445f8e0f7ebef7832146612291",
  "labels_sha256": "c226e8b2490016713d81a67546a44fd1d7ce982abf4850062b3b3d815b4d329d",
  "config": "full_v1_candidate",
  "seed": 830251,
  "model_manifest": {
    "schema": "tvcg.spin-neural-moments.v1",
    "name": "full_v1_candidate",
    "config": {
      "name": "full_v1_candidate",
      "pcraft_preset": "xl_ms_gno_transolver",
      "sqno_hidden_dim": 32,
      "sqno_final_hidden_dim": 128,
      "sqno_num_layers": 3,
      "sqno_neighbors": 16,
      "sqno_temperature": 0.04,
      "quadrature_order": 2,
      "sqno_chunk_intervals": 512,
      "sqno_chunk_nodes": 65536,
      "query_chunk": 16,
      "fusion_mode": "query_conditioned",
      "fusion_initial_weight": 0.5,
      "query_fusion_hidden": 32,
      "geometry_cross_attention": true,
      "geometry_attention_heads": 2,
      "geometry_attention_chunk": 1024,
      "geometry_attention_output_scale": 0.05,
      "cross_scale_attention": true,
      "cross_scale_heads": 2,
      "cross_scale_max_scales": 16,
      "interaction_variant": "parallel_gno_attention",
      "uq_enabled": true,
      "event_enabled": true
    },
    "config_fingerprint": "10d79334834607823243b34d393c96e3d018b583e10a1b498bbfac7507a1a63e",
    "pcraft_config": {
      "grid_size": 64,
      "pino_width": 32,
      "pino_modes": 12,
      "pino_layers": 6,
      "interval_width": 192,
      "interval_depth": 4,
      "chunk_intervals": 512,
      "min_native_resolution": 512,
      "checkpoint_intervals": true,
      "gno_enabled": true,
      "gno_neighbors": 8,
      "gno_cache_mib": 512,
      "gno_topology_filter": true,
      "transolver_enabled": true,
      "transolver_slices": 32,
      "transolver_heads": 8,
      "transolver_adaptive": false,
      "transolver_min_slices": 4,
      "transolver_adaptive_temperature": false,
      "transolver_temperature_min": 0.05,
      "transolver_temperature_max": 2.0,
      "multiscale_enabled": true,
      "multiscale_grids": [
        32,
        64,
        128
      ],
      "convex_potential_enabled": false,
      "convex_potential_components": 4,
      "convex_amplitude_mode": "sigmoid",
      "event_validity_head_enabled": true,
      "uq_enabled": true,
      "uq_hidden": 64,
      "uq_min_log_std": -7.0,
      "uq_max_log_std": 4.0
    },
    "modules": {
      "shared_pcraft": true,
      "m80": true,
      "sqno": true,
      "gno": true,
      "transolver_physics_slicing": true,
      "geometry_physics_cross_attention": true,
      "cross_scale_attention": true,
      "query_conditioned_fusion": true,
      "moment_jacobian_uq": true,
      "event_validity": true
    },
    "scientific_status": "candidate_configuration_not_performance_validated"
  },
  "batch": {
    "geometry_id": "N002",
    "source_group_id": "trellis/imagegen-sheet/e60739dc2685affa5f70a6e2fd5ede8e7be3a433341d0fe98e0a763e177ed198",
    "geometry_sha256": "bac895b5af674119bc20c02948d364710dac04445f8e0f7ebef7832146612291",
    "labels_sha256": "c226e8b2490016713d81a67546a44fd1d7ce982abf4850062b3b3d815b4d329d",
    "queries": 64,
    "moment_labels": 64,
    "full_jacobian_labels": 38,
    "restricted_jacobian_labels": 26,
    "restricted_by_dimension": {
      "3": 13,
      "8": 13
    },
    "native_resolution": 1024,
    "native_intervals": 1177764,
    "full_batch": true
  },
  "update": {
    "loss": 0.22954777548053193,
    "moment_loss": 0.15204767026361898,
    "jacobian_loss": 3.0306081671777093e-05,
    "full_jacobian_loss": 3.318655567307962e-05,
    "restricted_jacobian_loss": 2.6096158131411857e-05,
    "gradient_norm": 406.6504312339405,
    "parameter_delta_l2": 0.23342499864866031,
    "changed_parameter_tensors": 226,
    "elapsed_seconds": 88.65807101898827,
    "peak_memory_bytes": 10871928320,
    "peak_reserved_bytes": 10947133440,
    "optimizer_steps": 1,
    "consumed_moment_queries": 64,
    "consumed_full_jacobians": 38,
    "consumed_restricted_jacobians": 26,
    "consumed_restricted_directions": 143
  },
  "cache_invalidated": true,
  "checkpoint_resume_roundtrip": {
    "predictions": {
      "moments": {
        "shape": [
          64,
          10
        ],
        "max_abs": 5.464938918064988e-10,
        "max_rel": 1.2350500458591528e-07
      },
      "jacobian_coefficients": {
        "shape": [
          64,
          10,
          10
        ],
        "max_abs": 3.907219429943876e-09,
        "max_rel": 0.00048305655358790966
      }
    },
    "prediction_rtol": 1e-05,
    "prediction_atol": 1e-08,
    "relative_error_denominator": "max(abs(original_output), dtype_epsilon)",
    "restored_parameters": 245,
    "global_step": 1,
    "optimizer_state_entries": 230,
    "scheduler_last_epoch": 1,
    "rng_restored": true
  }
}
```
