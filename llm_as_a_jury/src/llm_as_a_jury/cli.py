import argparse
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .case_file import load_case
from .errors import JuryError
from .jury import JuryResult, run_jury
from .provider import OpenAICompatibleProvider


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mock-jury",
        description="Run an educational mock jury against a fictional case text file.",
        epilog="Educational simulation only; never use for real cases or legal decisions.",
    )
    parser.add_argument("case_file", type=Path, help="UTF-8 case description text file")
    parser.add_argument(
        "--jurors",
        type=_positive_integer,
        default=6,
        help="number of independent simulated jurors (default: 6)",
    )
    return parser


def _print_result(result: JuryResult) -> None:
    print(
        f"Verdict: {result.verdict.value.upper()} "
        f"(guilty: {result.guilty_votes}, "
        f"not guilty: {result.not_guilty_votes})"
    )
    for vote in result.votes:
        print(
            f"Juror {vote.juror_id} [{vote.verdict.value}]: {vote.rationale}"
        )
    print(
        "\nEducational simulation only. This output is not a real verdict "
        "or legal advice."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    try:
        case_text = load_case(args.case_file)
        provider = OpenAICompatibleProvider.from_environment()
        result = run_jury(case_text, juror_count=args.jurors, provider=provider)
    except JuryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_result(result)
    return 0
