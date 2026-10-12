import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[1] / 'scripts/evaluate_alignment_uniformity.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class GeometryTests(unittest.TestCase):
    def test_antipodal_vectors(self):
        z = np.array([[1., 0.], [-1., 0.]])
        result = evaluation.metrics(z, np.array([0, 1]), np.array([[0, 1]]),
                                    np.array(['left', 'right']), 'stratified', 42, 5)
        self.assertEqual(result['alignment'], 4.)
        self.assertAlmostEqual(result['uniformity'], -8.)
        self.assertEqual(result['uniformity_distances'], 1)

    def test_shared_events_deduplicate_article_pairs(self):
        rows = [{'event_ids': ['a', 'b']}, {'event_ids': ['a', 'b']}, {'event_ids': ['a']}]
        pairs, _ = evaluation.positive_pairs(rows, np.array(['left', 'right', 'left']), 'event_ids')
        np.testing.assert_array_equal(pairs, [[0, 1], [1, 2]])

    def test_sampling_repeats_without_replacement(self):
        labels = np.array(['left'] * 20 + ['right'] * 20)
        a, mode = evaluation.select_articles(labels, 10, 42)
        b, _ = evaluation.select_articles(labels, 10, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(mode, 'stratified')
        self.assertEqual(len(set(a)), 10)
        self.assertEqual(sum(labels[a] == 'left'), 5)


if __name__ == '__main__':
    unittest.main()
