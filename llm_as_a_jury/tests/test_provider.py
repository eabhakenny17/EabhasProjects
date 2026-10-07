import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from llm_as_a_jury.errors import ProviderError
from llm_as_a_jury.provider import OpenAICompatibleProvider


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = OpenAICompatibleProvider(
            base_url="https://example.test/v1/",
            api_key="test-key",
            model="test-model",
        )

    @patch("llm_as_a_jury.provider.urlopen")
    def test_sends_chat_request_and_returns_message_content(
        self, urlopen: MagicMock
    ) -> None:
        response = io.BytesIO(
            b'{"choices":[{"message":{"content":"{\\"verdict\\":\\"guilty\\",'
            b'\\"rationale\\":\\"Evidence.\\"}"}}]}'
        )
        urlopen.return_value = response

        content = self.provider.complete(
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(
            content,
            '{"verdict":"guilty","rationale":"Evidence."}',
        )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    @patch(
        "llm_as_a_jury.provider.urlopen",
        side_effect=HTTPError("https://example.test", 401, "Unauthorized", {}, None),
    )
    def test_surfaces_http_errors(self, _urlopen: MagicMock) -> None:
        with self.assertRaisesRegex(ProviderError, "HTTP 401"):
            self.provider.complete(system_prompt="system", user_prompt="user")

    def test_requires_http_provider_url(self) -> None:
        with self.assertRaisesRegex(ProviderError, "absolute HTTP"):
            OpenAICompatibleProvider(
                base_url="file:///tmp/provider",
                api_key="key",
                model="model",
            )

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "llama3.2",
        },
        clear=True,
    )
    def test_reads_openai_compatible_environment_names(self) -> None:
        provider = OpenAICompatibleProvider.from_environment()

        self.assertEqual(provider.api_key, "test-openai-key")
        self.assertEqual(
            provider.endpoint,
            "http://localhost:11434/v1/chat/completions",
        )
        self.assertEqual(provider.model, "llama3.2")


if __name__ == "__main__":
    unittest.main()
