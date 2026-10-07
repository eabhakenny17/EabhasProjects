import tempfile
import unittest
from pathlib import Path

from llm_as_a_jury.case_file import load_case
from llm_as_a_jury.errors import CaseFileError


class LoadCaseTests(unittest.TestCase):
    def test_reads_utf8_case_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.txt"
            path.write_text("Fictional case: café.", encoding="utf-8")

            self.assertEqual(load_case(path), "Fictional case: café.")

    def test_rejects_empty_case_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text(" \n", encoding="utf-8")

            with self.assertRaisesRegex(CaseFileError, "is empty"):
                load_case(path)

    def test_reports_missing_case_file(self) -> None:
        path = Path("missing-case-file-for-test.txt")

        with self.assertRaisesRegex(CaseFileError, "Could not read case file"):
            load_case(path)


if __name__ == "__main__":
    unittest.main()
