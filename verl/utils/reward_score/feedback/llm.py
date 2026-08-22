from __future__ import annotations

import os
import random

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-32B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
REQUEST_TIMEOUT_S = 60.0

DEFAULT_JUDGE_PROMPT_TEMPLATE = """You are an expert reviewer. Your job is to read the following user prompt and response, then provide helpful, concise general feedback on the response. Focus on how well the response addresses the user's needs, tone, correctness, and completeness. If there are issues or improvements that could be made, mention them briefly and constructively.

<user_prompt>
{prompt}
</user_prompt>

<assistant_response>
{response}
</assistant_response>

Output only one or two sentences of feedback, addressed to the response writer. Do not provide scores, grades, or JSON, just the textual feedback."""  # noqa: E501

_client: OpenAI | None = None


def _base_url() -> str:
    return os.environ.get("LLM_JUDGE_BASE_URL", DEFAULT_BASE_URL)


def _api_key() -> str:
    return os.environ.get("LLM_JUDGE_API_KEY", "EMPTY")


def _model() -> str:
    return os.environ.get("LLM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = _base_url()
        try:
            # Connectivity probe via a tiny chat completion (the OpenAI-style contract
            # both local vLLM and the CSCS serving gateway honor). Do NOT probe GET
            # /models: the CSCS gateway does not expose it (404). A reachable server
            # returns *some* HTTP status (200, or 4xx/5xx while warming up / on a bad
            # key) — only a transport error means it is truly unreachable.
            httpx.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _model(),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            print(f"Error: {e}")
            raise RuntimeError(
                f"LLM judge server not reachable at {base_url}. Check "
                f"LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL / LLM_JUDGE_API_KEY.\n"
                f"For a local server: vllm serve {_model()} --host 127.0.0.1 --port 8000\n"
                f"Original error: {e}"
            ) from e
        _client = OpenAI(base_url=base_url, api_key=_api_key(), timeout=REQUEST_TIMEOUT_S)
    return _client


def _run_judge_call(prompt: str) -> str:
    try:
        resp = _get_client().chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(" Error in feedback LLM call: ", e)
        return ""


def compute_score(
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict | None = None,
    judge_prompt_template: str | None = None,
) -> dict:
    extra_info = extra_info or {}
    template = judge_prompt_template if judge_prompt_template is not None else DEFAULT_JUDGE_PROMPT_TEMPLATE
    prompt = template.format(
        prompt=extra_info.get("prompt", ""),
        response=solution_str,
        ground_truth=ground_truth or "",
    )

    if random.random() < 0.05:
        print("Response for feedback LLM: ", solution_str)
        print("Prompt for feedback LLM: ", prompt)
    if extra_info.get("prompt", "").strip() == "":
        print("Warning: empty prompt")

    feedback = _run_judge_call(prompt)

    return {
        # ! -1.0 is a DUMMY SCORE: this module is feedback-only, NOT a reward signal.
        "score": -1.0,
        "acc": -1.0,
        "pred": "",
        "incorrect_format": 0,
        "feedback": feedback,
    }
