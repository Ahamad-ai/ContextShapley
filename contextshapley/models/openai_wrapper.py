"""
LLM wrapper supporting:
  1. Direct OpenAI API (default)
  2. AWS Bedrock (Claude models via boto3)
  3. Optional local Azure wrapper via model_loader.py

Handles: API calls, retries, response logging, cost tracking.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import RateLimitError, APITimeoutError, APIConnectionError, BadRequestError
from dotenv import load_dotenv

# Load variables from .env when available for reproducible local setup.
load_dotenv()


@dataclass
class CallResult:
    """Result from a single LLM call."""
    response_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: str
    raw_response: dict = field(default_factory=dict)


def _load_model_loader_client():
    """Load the AzureOpenAI client from the workspace model_loader.py."""
    workspace_root = Path(__file__).resolve().parents[4]  # up to MyResearchSpace
    sys.path.insert(0, str(workspace_root))
    from model_loader import ModelLoader
    loader = ModelLoader()
    client = loader.get_llm_model()
    return client


class BedrockModel:
    """
    Wrapper around AWS Bedrock Converse API for Claude models.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-opus-4-20250514-v1:0",
        region: str = "us-east-1",
        profile_name: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        log_dir: str | None = None,
    ):
        import boto3

        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_dir = Path(log_dir) if log_dir else None

        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        self.client = session.client("bedrock-runtime", region_name=region)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        call_id: str | None = None,
    ) -> CallResult:
        last_error = None

        # Convert OpenAI format to Bedrock Converse format
        system_text = None
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        for attempt in range(self.max_retries):
            try:
                start = time.time()

                kwargs = {
                    "modelId": self.model_id,
                    "messages": bedrock_messages,
                    "inferenceConfig": {
                        "maxTokens": max_tokens,
                        "temperature": 0.0,
                    },
                }
                if system_text:
                    kwargs["system"] = [{"text": system_text}]

                response = self.client.converse(**kwargs)
                latency_ms = (time.time() - start) * 1000

                usage = response.get("usage", {})
                content = response["output"]["message"]["content"]
                response_text = content[0]["text"] if content else ""

                result = CallResult(
                    response_text=response_text,
                    model=self.model_id,
                    prompt_tokens=usage.get("inputTokens", 0),
                    completion_tokens=usage.get("outputTokens", 0),
                    total_tokens=usage.get("totalTokens", 0),
                    latency_ms=latency_ms,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    raw_response={
                        "stop_reason": response.get("stopReason", ""),
                        "model": self.model_id,
                    },
                )

                self.total_prompt_tokens += result.prompt_tokens
                self.total_completion_tokens += result.completion_tokens
                self.total_calls += 1

                if self.log_dir and call_id:
                    self._log_call(call_id, messages, result)

                return result

            except self.client.exceptions.ThrottlingException as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                time.sleep(wait)
            except Exception as e:
                if "throttl" in str(e).lower() or "rate" in str(e).lower():
                    last_error = e
                    wait = self.retry_delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_error}")

    def _log_call(self, call_id: str, messages: list[dict], result: CallResult):
        log_entry = {
            "call_id": call_id,
            "model": self.model_id,
            "messages": messages,
            "response_text": result.response_text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
        }
        log_path = self.log_dir / f"{call_id}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

    def get_cost_summary(self) -> dict:
        return {
            "model": self.model_id,
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


class OllamaModel:
    """
    Wrapper around Ollama's OpenAI-compatible API (localhost:11434).
    Supports temperature=0 for deterministic results.
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434/v1",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        log_dir: str | None = None,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_dir = Path(log_dir) if log_dir else None
        self.client = OpenAI(base_url=base_url, api_key="ollama")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        call_id: str | None = None,
    ) -> CallResult:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.time() - start) * 1000

                usage = response.usage
                p_tok = usage.prompt_tokens if usage else 0
                c_tok = usage.completion_tokens if usage else 0

                result = CallResult(
                    response_text=response.choices[0].message.content or "",
                    model=response.model or self.model_name,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency_ms,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    raw_response={
                        "model": self.model_name,
                        "finish_reason": response.choices[0].finish_reason,
                    },
                )

                self.total_prompt_tokens += p_tok
                self.total_completion_tokens += c_tok
                self.total_calls += 1

                if self.log_dir and call_id:
                    self._log_call(call_id, messages, result)

                return result

            except Exception as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                time.sleep(wait)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_error}")

    def _log_call(self, call_id: str, messages: list[dict], result: CallResult):
        log_entry = {
            "call_id": call_id,
            "model": self.model_name,
            "messages": messages,
            "response_text": result.response_text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
        }
        log_path = self.log_dir / f"{call_id}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

    def get_cost_summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


class OpenAIModel:
    """
    Wrapper around OpenAI / Azure OpenAI chat completions API.

    Supports two modes:
    - use_model_loader=False (default): Uses direct OpenAI API key
    - use_model_loader=True: Uses optional local model_loader.py wrapper
    """

    def __init__(
        self,
        model_name: str = "gpt-5",
        use_model_loader: bool = False,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        log_dir: str | None = None,
    ):
        self.model_name = model_name
        self.use_model_loader = use_model_loader
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_dir = Path(log_dir) if log_dir else None

        if use_model_loader:
            self.client = _load_model_loader_client()
        else:
            from openai import OpenAI
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it in environment or pass api_key."
                )
            self.client = OpenAI(api_key=resolved_key)

        # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        call_id: str | None = None,
    ) -> CallResult:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start = time.time()

                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                }

                if self.use_model_loader:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["temperature"] = 0.0
                    kwargs["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**kwargs)
                latency_ms = (time.time() - start) * 1000

                usage = response.usage
                result = CallResult(
                    response_text=response.choices[0].message.content or "",
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    latency_ms=latency_ms,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    raw_response={
                        "id": response.id,
                        "model": response.model,
                        "finish_reason": response.choices[0].finish_reason,
                    },
                )

                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self.total_calls += 1

                if self.log_dir and call_id:
                    self._log_call(call_id, messages, result)

                return result

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                time.sleep(wait)
            except BadRequestError as e:
                raise

        raise RuntimeError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )

    def _log_call(self, call_id: str, messages: list[dict], result: CallResult):
        log_entry = {
            "call_id": call_id,
            "model": self.model_name,
            "messages": messages,
            "response_text": result.response_text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
        }
        log_path = self.log_dir / f"{call_id}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

    def get_cost_summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
