import unittest
from metrics import grade, repetition

class MetricsTests(unittest.TestCase):
    def test_censored(self):
        g = grade(r'\\boxed{45}', '45', False)
        self.assertTrue(g['exact_box_match'])
        self.assertIsNone(g['correct'])

    def test_completed(self):
        self.assertEqual(grade(r'</think>\\boxed{45}', '45', True)['grading_status'], 'completed_correct')
        self.assertEqual(grade(r'</think>\\boxed{44}', '45', True)['grading_status'], 'completed_wrong')
        self.assertEqual(grade('unfinished answer', '45', True)['grading_status'], 'completed_missing_answer')

    def test_last_box(self):
        self.assertFalse(grade(r'</think>\\boxed{45} then \\boxed{44}', '45', True)['correct'])
        self.assertIsNone(grade(r'\\boxed{45}</think>No final answer', '45', True)['correct'])

    def test_repetition_all_alignments(self):
        block = list(range(16))
        self.assertFalse(repetition(block * 3)['repeated_16_block_four_times'])
        self.assertEqual(repetition([99] + block * 4)['first_four_block_end'], 65)
        self.assertFalse(repetition(list(range(100)))['repeated_16_block_four_times'])

if __name__ == '__main__':
    unittest.main()
