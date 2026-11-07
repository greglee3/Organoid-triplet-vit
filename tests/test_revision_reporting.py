"""Synthetic, CPU-only tests. Run: python -m unittest discover -s tests -v."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_revision.py"
SPEC = importlib.util.spec_from_file_location("revision_reporting", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def synthetic_rows():
    rows = []
    labels = np.repeat(np.arange(3), 2)
    for seed in (1, 2):
        ce = np.full((6, 3), .1)
        ce[np.arange(6), labels] = .8
        triplet = ce.copy()
        efficient = ce.copy()
        triplet[[0, 3] if seed == 1 else [4]] = [.1, .8, .1]
        efficient[1] = [.1, .1, .8]
        probabilities = (ce, triplet, efficient, .5 * triplet + .5 * efficient)
        for model, probability in zip(reporter.MODELS, probabilities):
            for index, label in enumerate(labels):
                rows.append(dict(zip(reporter.COLUMNS, (seed, model, f"sample-{index}", reporter.CLASSES[label], *probability[index]))))
    return rows


class RevisionReportingTests(unittest.TestCase):
    def setUp(self):
        # Use the ignored workspace output directory, also in restricted sandboxes.
        scratch = SCRIPT.parents[1] / "generated"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="revision-tests-", dir=scratch)
        self.root = Path(self.temporary.name)
        self.path = self.root / "predictions.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, rows=None):
        rows = synthetic_rows() if rows is None else rows
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=reporter.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return reporter.load_predictions(self.path, expected_seeds=(1, 2), expected_class_count=2)

    def test_metrics_mean_sample_sd_and_mean_confusion(self):
        data = self.load()
        report = reporter.summarize(data)
        self.assertEqual(len(report["rows"]), 8)
        self.assertEqual(report["summary"]["CE-only ViT"]["accuracy"], {"mean": 1.0, "sd": 0.0})
        accuracies = [row["metrics"]["accuracy"] for row in report["rows"] if row["model"] == "Triplet-ViT"]
        self.assertAlmostEqual(report["summary"]["Triplet-ViT"]["accuracy"]["sd"], np.std(accuracies, ddof=1))
        np.testing.assert_allclose(np.sum(report["mean_confusion_matrices"]["50:50 ensemble"], axis=1), [2, 2, 2])
        for row in report["rows"]:
            self.assertTrue(all(0 <= value <= 1 for value in row["metrics"]["average_precision_ovr"].values()))

    def test_reordered_rows_align_by_identity(self):
        expected = self.load()["probabilities"].copy()
        rows = synthetic_rows()
        rows[6:12] = reversed(rows[6:12])
        np.testing.assert_array_equal(self.load(rows)["probabilities"], expected)

    def test_duplicate_identity_rejected(self):
        rows = synthetic_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(rows + [rows[0]])

    def test_identity_and_label_mismatch_rejected(self):
        rows = synthetic_rows()
        rows[6]["sample"] = "replacement"
        with self.assertRaisesRegex(ValueError, "identities"):
            self.load(rows)
        rows = synthetic_rows()
        rows[6]["true_label"] = "B"
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            self.load(rows)

    def test_invalid_probability_rejected(self):
        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            rows = synthetic_rows()
            rows[0]["prob_A"] = invalid
            with self.assertRaisesRegex(ValueError, "finite"):
                self.load(rows)
        rows = synthetic_rows()
        rows[0]["prob_A"] = .3
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.load(rows)

    def test_ensemble_arithmetic_rejected(self):
        rows = synthetic_rows()
        ensemble = next(row for row in rows if row["model"] == "50:50 ensemble")
        ensemble["prob_A"], ensemble["prob_B"] = .7, .2
        ensemble["prob_C"] = .1
        with self.assertRaisesRegex(ValueError, "ensemble probabilities"):
            self.load(rows)

    def test_missing_seed_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.load(synthetic_rows()[:-6])

    def test_reference_verification(self):
        report = reporter.summarize(self.load())
        reference = {"rows": json.loads(json.dumps(report["rows"])), "summary": {
            model: {metric: {stat: 100 * values[stat] for stat in ("mean", "sd")}
                    for metric, values in summary.items() if metric in reporter.METRICS}
            for model, summary in report["summary"].items()}}
        reporter.verify_reference(report, reference)
        reference["rows"][0]["metrics"]["accuracy"] -= .01
        with self.assertRaisesRegex(ValueError, "Reference mismatch"):
            reporter.verify_reference(report, reference)

    def test_paired_bootstrap_is_deterministic_and_matches_original(self):
        data = self.load()
        first = reporter.paired_uncertainty(data, replicates=31, rng_seed=20261102)
        self.assertEqual(first, reporter.paired_uncertainty(data, replicates=31, rng_seed=20261102))
        # Independent transcription of the original image-resampling calculation.
        rng = np.random.default_rng(20261102)
        y = data["labels"]
        weights = np.zeros((31, 6), dtype=np.int32)
        for label in range(3):
            ids = np.flatnonzero(y == label)
            weights[:, ids] = rng.multinomial(2, np.full(2, .5), size=31)
        predictions = data["probabilities"][:, :2].argmax(-1)
        differences = np.empty((31, 2))
        for seed in range(2):
            correct_ce = predictions[seed, 0] == y
            correct_triplet = predictions[seed, 1] == y
            differences[:, seed] = 100 * (weights[:, correct_triplet].sum(1) - weights[:, correct_ce].sum(1)) / 6
        np.testing.assert_allclose(first["image_bootstrap_summary"]["accuracy"]["paired_image_bootstrap_ci95_pp"],
                                   np.quantile(differences.mean(1), [.025, .975]))
        self.assertEqual(first["per_seed"][0]["ce_only_correct"], 1)
        self.assertEqual(first["per_seed"][0]["triplet_only_correct"], 0)

    def test_identical_models_zero_uncertainty(self):
        data = self.load()
        data["probabilities"][:, 1] = data["probabilities"][:, 0]
        result = reporter.paired_uncertainty(data, replicates=11)
        for metric in result["image_bootstrap_summary"].values():
            self.assertEqual(metric["delta_pp"], 0)
            self.assertEqual(metric["paired_image_bootstrap_ci95_pp"], [0, 0])
        self.assertTrue(all(row["mcnemar_exact_two_sided_p"] == 1 for row in result["per_seed"]))
        np.testing.assert_allclose(reporter.holm([.03, .01, .04]), [.06, .03, .06])

    def test_roc_endpoints_and_output_no_overwrite(self):
        data = self.load()
        report = reporter.summarize(data)
        curves = reporter.mean_roc(data, points=11)
        self.assertEqual(len(curves), 4 * 3 * 11)
        self.assertTrue(all(row["mean_tpr"] == row["fpr"] for row in curves if row["fpr"] in (0, 1)))
        output = self.root / "report"
        reporter.write_outputs(output, report, curves, figures=False)
        self.assertTrue((output / "revision_report.json").is_file())
        with self.assertRaises(FileExistsError):
            reporter.write_outputs(output, report, curves, figures=False)
        marker = output / "unrelated.txt"
        marker.write_text("keep", encoding="utf-8")
        reporter.write_outputs(output, report, curves, overwrite=True, figures=False)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
