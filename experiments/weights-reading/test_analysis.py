import unittest
import torch
import analysis


class AnalysisTests(unittest.TestCase):
    def test_m1_and_profile_on_shifted_identity(self):
        t = 64
        attn = torch.zeros(2, t, t)
        attn[0, 1:, :-1] = torch.eye(t - 1)            # head 0: all mass one position back
        attn[0, 0, 0] = 1.0
        attn[1] = torch.tril(torch.ones(t, t)) / torch.arange(1, t + 1)[:, None]   # head 1: uniform over past
        m1 = analysis.m1_from_attn(attn, 16)
        self.assertAlmostEqual(m1[0].item(), 1.0, places=6)
        self.assertLess(m1[1].item(), 0.05)
        prof = analysis.profile_from_attn(attn, 48)
        self.assertEqual(prof.shape, (2, 48))
        self.assertAlmostEqual(prof[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(prof[0].sum().item(), 1.0, places=6)
        self.assertAlmostEqual(prof[1, 0].item(), prof[1, 47].item(), places=6)

    def test_spearman_and_permutation(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertAlmostEqual(analysis.spearman(x, x), 1.0)
        self.assertAlmostEqual(analysis.spearman(x, x[::-1]), -1.0)
        self.assertAlmostEqual(analysis.spearman([1, 1, 2, 2], [1, 2, 1, 2]), 0.0)
        rho, p = analysis.permutation_p(x, [v * 2 for v in x], n=2000, seed=1)
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(p, 0.01)
        rho, p = analysis.permutation_p(x, [3, 1, 4, 1, 5, 9, 2, 6], n=2000, seed=1)
        self.assertGreater(p, 0.05)


if __name__ == '__main__':
    unittest.main()
