import json
from pathlib import Path
import unittest


ASSIGNMENT_DIR = Path(__file__).parent


class AssignmentTests(unittest.TestCase):
    def test_dataset_has_expected_records(self):
        with (ASSIGNMENT_DIR / "instruction-data.json").open(encoding="utf-8") as file:
            data = json.load(file)

        self.assertEqual(len(data), 1142)

    def test_train_source_completes_assignment_requirements(self):
        source = (ASSIGNMENT_DIR / "train.py").read_text(encoding="utf-8")

        self.assertNotIn("# TODO:", source)
        self.assertIn('CHOOSE_MODEL = "gpt2-medium (355M)"', source)
        self.assertIn("num_epochs=2", source)

    def test_inference_asks_world_cup_question(self):
        source = (ASSIGNMENT_DIR / "inference.py").read_text(encoding="utf-8")

        self.assertIn("2026 FIFA World Cup", source)
        self.assertIn("Round of 32", source)


if __name__ == "__main__":
    unittest.main()
