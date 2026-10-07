import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .errors import ProviderError


class ChatProvider(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant's text response for one chat completion."""


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ProviderError("LLM_BASE_URL must be an absolute HTTP or HTTPS URL.")
        if not api_key.strip():
            raise ProviderError("LLM_API_KEY must not be empty.")
        if not model.strip():
            raise ProviderError("LLM_MODEL must not be empty.")
        if timeout_seconds <= 0:
            raise ProviderError("Provider timeout must be greater than zero.")

        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleProvider":
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("LLM_MODEL")
        if api_key is None or not api_key.strip():
            raise ProviderError("Missing required environment variable: LLM_API_KEY.")
        if model is None or not model.strip():
            raise ProviderError("Missing required environment variable: LLM_MODEL.")

        return cls(
            base_url=(
                os.environ.get("LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            api_key=api_key,
            model=model,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
        except HTTPError as exc:
            raise ProviderError(
                f"LLM provider returned HTTP {exc.code}: {exc.reason}."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderError(f"Could not reach the LLM provider: {exc}.") from exc
        try:
            raw_response = response_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderError("LLM provider response was not valid UTF-8.") from exc

        try:
            payload: object = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ProviderError("LLM provider returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ProviderError("LLM provider response must be a JSON object.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("LLM provider response did not contain any choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderError("LLM provider returned an invalid choice.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("LLM provider response did not contain a message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("LLM provider returned an empty or non-text message.")

        return content
