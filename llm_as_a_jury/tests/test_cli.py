import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_as_a_jury.cli import main
from llm_as_a_jury.errors import ProviderError
from llm_as_a_jury.jury import JuryResult, JurorVote, Verdict, Vote


class CliTests(unittest.TestCase):
    @patch("llm_as_a_jury.cli.run_jury")
    @patch("llm_as_a_jury.cli.OpenAICompatibleProvider.from_environment")
    def test_prints_tally_and_educational_notice(
        self, from_environment: MagicMock, run_jury_mock: MagicMock
    ) -> None:
        from_environment.return_value = object()
        run_jury_mock.return_value = JuryResult(
            verdict=Verdict.HUNG,
            votes=(
                JurorVote(1, Vote.GUILTY, "Facts support the rule."),
                JurorVote(2, Vote.NOT_GUILTY, "Facts are unclear."),
            ),
            guilty_votes=1,
            not_guilty_votes=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.txt"
            case_path.write_text("Fictional case.", encoding="utf-8")
            output = io.StringIO()

            with (
                redirect_stdout(output),
                patch("llm_as_a_jury.cli.load_dotenv") as load_dotenv,
            ):
                exit_code = main([str(case_path), "--jurors", "2"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Verdict: HUNG", output.getvalue())
        self.assertIn("Educational simulation only", output.getvalue())
        load_dotenv.assert_called_once_with(
            dotenv_path=Path(__file__).resolve().parents[1] / ".env",
            override=False,
        )

    @patch("llm_as_a_jury.cli.OpenAICompatibleProvider.from_environment")
    def test_reports_provider_error(self, from_environment: MagicMock) -> None:
        from_environment.side_effect = ProviderError("Missing API configuration.")
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.txt"
            case_path.write_text("Fictional case.", encoding="utf-8")
            error_output = io.StringIO()

            with redirect_stderr(error_output):
                exit_code = main([str(case_path)])

        self.assertEqual(exit_code, 2)
        self.assertIn("Missing API configuration.", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
