import unittest

from llm_as_a_jury.errors import InvalidJurorResponse
from llm_as_a_jury.jury import (
    JurorVote,
    Verdict,
    Vote,
    aggregate_votes,
    parse_vote_response,
    run_jury,
)


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self.responses[len(self.prompts) - 1]


class ParseVoteTests(unittest.TestCase):
    def test_parses_guilty_vote(self) -> None:
        vote = parse_vote_response(
            '{"verdict":"guilty","rationale":"The stated facts support the rule."}',
            2,
        )

        self.assertEqual(vote, JurorVote(2, Vote.GUILTY, "The stated facts support the rule."))

    def test_rejects_non_json_response(self) -> None:
        with self.assertRaisesRegex(InvalidJurorResponse, "not valid JSON"):
            parse_vote_response("not json", 1)

    def test_rejects_unsupported_verdict(self) -> None:
        response = '{"verdict":"innocent","rationale":"No evidence."}'

        with self.assertRaisesRegex(InvalidJurorResponse, "unsupported verdict"):
            parse_vote_response(response, 1)

    def test_rejects_missing_rationale(self) -> None:
        with self.assertRaisesRegex(InvalidJurorResponse, "rationale"):
            parse_vote_response('{"verdict":"not_guilty","rationale":" "}', 1)


class AggregateVotesTests(unittest.TestCase):
    def test_guilty_majority_wins(self) -> None:
        result = aggregate_votes(
            [
                JurorVote(1, Vote.GUILTY, "A"),
                JurorVote(2, Vote.GUILTY, "B"),
                JurorVote(3, Vote.NOT_GUILTY, "C"),
            ]
        )

        self.assertEqual(result.verdict, Verdict.GUILTY)
        self.assertEqual((result.guilty_votes, result.not_guilty_votes), (2, 1))

    def test_not_guilty_majority_wins(self) -> None:
        result = aggregate_votes(
            [
                JurorVote(1, Vote.GUILTY, "A"),
                JurorVote(2, Vote.NOT_GUILTY, "B"),
                JurorVote(3, Vote.NOT_GUILTY, "C"),
            ]
        )

        self.assertEqual(result.verdict, Verdict.NOT_GUILTY)

    def test_tied_vote_is_hung(self) -> None:
        result = aggregate_votes(
            [
                JurorVote(1, Vote.GUILTY, "A"),
                JurorVote(2, Vote.NOT_GUILTY, "B"),
            ]
        )

        self.assertEqual(result.verdict, Verdict.HUNG)

    def test_empty_or_duplicate_votes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            aggregate_votes([])
        with self.assertRaisesRegex(ValueError, "unique ID"):
            aggregate_votes(
                [
                    JurorVote(1, Vote.GUILTY, "A"),
                    JurorVote(1, Vote.NOT_GUILTY, "B"),
                ]
            )


class RunJuryTests(unittest.TestCase):
    def test_calls_provider_once_per_juror_and_aggregates(self) -> None:
        provider = FakeProvider(
            [
                '{"verdict":"guilty","rationale":"Evidence supports the rule."}',
                '{"verdict":"not_guilty","rationale":"Evidence is unclear."}',
                '{"verdict":"guilty","rationale":"Evidence supports the rule."}',
            ]
        )

        result = run_jury("A fictional case.", juror_count=3, provider=provider)

        self.assertEqual(len(provider.prompts), 3)
        self.assertIn("juror 1 of 3", provider.prompts[0][1])
        self.assertEqual(result.verdict, Verdict.GUILTY)

    def test_rejects_invalid_juror_count_before_calling_provider(self) -> None:
        provider = FakeProvider([])

        with self.assertRaisesRegex(ValueError, "at least one"):
            run_jury("A case.", juror_count=0, provider=provider)
        self.assertEqual(provider.prompts, [])


if __name__ == "__main__":
    unittest.main()
