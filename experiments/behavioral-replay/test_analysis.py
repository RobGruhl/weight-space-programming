import unittest
from analyze import diagnostics

class ExploratoryTests(unittest.TestCase):
    def test_period_seven_and_offsets(self):
        d = diagnostics([99, 99] + list(range(7)) * 5, '', False)
        self.assertEqual(d['maximum_exact_adjacent_repeats_by_period']['7'], 5)
        self.assertIn(7, d['periods_with_four_adjacent_repeats'])
        self.assertEqual(d['eos_format_category'], 'censored')

    def test_eos_box_without_think(self):
        self.assertEqual(diagnostics([1], r'\\boxed{4}', True)['eos_format_category'], 'eos_box_present_without_closing_think')

    def test_compression_and_empty(self):
        a = diagnostics([42] * 512, '', False)
        b = diagnostics(list(range(512)), '', False)
        self.assertLess(a['compression']['compressed_over_raw'], b['compression']['compressed_over_raw'])
        self.assertIsNone(diagnostics([], '', False)['compression']['compressed_over_raw'])

if __name__ == '__main__':
    unittest.main()
