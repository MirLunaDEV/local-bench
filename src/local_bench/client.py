"""OpenAI-compatible chat client for local servers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from local_bench.download import USER_AGENT


class ChatError(RuntimeError):
    pass


class ChatReply:
    def __init__(self, text: str, usage: dict | None = None):
        self.text = text
        self.usage = usage


def _retry_delay(header: str | None, fallback: float) -> float:
    """Honor Retry-After seconds or an HTTP date. Cap so a bad header cannot stall a run."""
    if not header:
        return fallback
    text = header.strip()
    try:
        return min(max(float(text), 0.0), 60.0)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return fallback
    if when is None:
        return fallback
    return min(max(when.timestamp() - time.time(), 0.0), 60.0)


def normalize_base(url: str) -> str:
    text = url.strip().rstrip("/")
    if text.endswith("/chat/completions"):
        text = text[: -len("/chat/completions")]
    if not text.endswith("/v1"):
        text += "/v1"
    return text


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        content = "\n".join(parts)
    text = str(content or "").strip()
    if text:
        return text
    fallback = message.get("reasoning_content") or message.get("reasoning") or ""
    return str(fallback)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout: float = 300,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        no_think: bool = False,
        retries: int = 3,
    ):
        self.base_url = normalize_base(base_url)
        self.model = model
        self.api_key = api_key or "local"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.no_think = no_think
        self.retries = retries

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        return self.complete(messages, max_tokens=max_tokens).text

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> ChatReply:
        limit = self.max_tokens if max_tokens is None else max_tokens
        payload = self._payload(messages, limit, use_completion_tokens=False)
        try:
            return self._post(payload)
        except ChatError as exc:
            if "max_completion_tokens" in str(exc) or "max_tokens" in str(exc).lower():
                return self._post(self._payload(messages, limit, use_completion_tokens=True))
            raise

    def _payload(self, messages: list[dict], max_tokens: int, use_completion_tokens: bool) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        key = "max_completion_tokens" if use_completion_tokens else "max_tokens"
        payload[key] = max_tokens
        if self.no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(self, payload: dict) -> ChatReply:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return _read_choice(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
                last_error = ChatError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt + 1 == self.retries:
                    raise last_error from exc
                delay = 1.5 * (attempt + 1)
                if exc.code in {429, 503}:
                    header = exc.headers.get("Retry-After") if exc.headers else None
                    delay = _retry_delay(header, delay)
                time.sleep(delay)
                continue
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = ChatError(str(exc))
                if attempt + 1 == self.retries:
                    raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
        raise ChatError(str(last_error))


def _read_choice(data: dict) -> ChatReply:
    choices = data.get("choices") or []
    if not choices:
        raise ChatError(f"응답에 choices가 없습니다: {json.dumps(data)[:400]}")
    message = choices[0].get("message") or {}
    text = message_text(message)
    if not text and choices[0].get("text"):
        text = str(choices[0]["text"])
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return ChatReply(text, usage)
