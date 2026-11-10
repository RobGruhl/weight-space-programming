import unittest
import torch
from run import construct_control

class ControlTests(unittest.TestCase):
    def test_frozen_repeatable_support(self):
        base = torch.zeros((64, 8), dtype=torch.bfloat16)
        edit = base.clone()
        edit[:, [2, 5]] = torch.arange(128).reshape(64, 2).to(torch.bfloat16) / 128
        a, report = construct_control(base, edit, torch.tensor([2, 5]), 20261107)
        b, _ = construct_control(base, edit, torch.tensor([2, 5]), 20261107)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, [0, 1, 3, 4, 6, 7]], base[:, [0, 1, 3, 4, 6, 7]]))
        self.assertLess(abs(report['relative_norm_mismatch']), .01)
        self.assertFalse(torch.equal(a, edit))

if __name__ == '__main__':
    unittest.main()
