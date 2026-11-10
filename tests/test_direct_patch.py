import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('direct_patch',Path(__file__).resolve().parents[1]/'experiments/head-algebra/direct_patch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DirectPatchTests(unittest.TestCase):
    def test_target_and_control_invariants(self):
        report=module.demo()
        self.assertAlmostEqual(report['target_margin_change'],1.5)
        self.assertEqual(report['construction_control_changes'],[0,0])
    def test_degenerate_target_rejected(self):
        with self.assertRaises(ValueError):module.direction_for_target([1,2],[[1,0],[0,1]],1)
    def test_large_edit_rejected(self):
        with self.assertRaises(ValueError):module.direction_for_target([1,2,1e-4],[[1,0,0],[0,1,0]],1,max_update_norm=1)
