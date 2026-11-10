import unittest
from audit_termination import termination

class TerminationTests(unittest.TestCase):
    def test_first_marker_and_suffix(self):
        r = termination([1, 46, 46, 44], 46, [44])
        self.assertEqual(r['first_im_end_index_zero_based'], 1)
        self.assertEqual(r['tokens_through_first_im_end'], 2)
        self.assertEqual(r['tokens_after_first_im_end'], 2)
        self.assertEqual(r['suffix_after_first_im_end'], [46, 44])
        self.assertTrue(r['last_token_is_generation_eos'])

    def test_missing_marker(self):
        r = termination([1, 2], 46, [44])
        self.assertIsNone(r['first_im_end_index_zero_based'])
        self.assertFalse(r['last_token_is_generation_eos'])

if __name__ == '__main__':
    unittest.main()
