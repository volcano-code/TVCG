import unittest
import numpy as np
from research.NeuralMoments.phase2_2_controlled_distillation import run_controlled_distillation

class ControlledDistillationSmokeTests(unittest.TestCase):
    def test_all_three_architectures_train_and_report_finite_metrics(self):
        report=run_controlled_distillation(seeds=(22101,),steps=10)
        self.assertEqual(report["protocol"]["kind"],"synthetic_teacher_distillation_not_physical_oracle")
        self.assertEqual(len(report["results"]),3)
        self.assertEqual(set(report["summary"]),{"film","conditional_norm","cross_attention"})
        for row in report["results"]:
            self.assertTrue(np.isfinite(row["best_validation_moment_rmse"]))
            self.assertTrue(np.isfinite(row["final_validation_moment_rmse"]))
            self.assertTrue(np.isfinite(row["validation_jacobian_rmse"]))
            self.assertGreater(row["trainable_parameters"],0)

if __name__=="__main__": unittest.main()
