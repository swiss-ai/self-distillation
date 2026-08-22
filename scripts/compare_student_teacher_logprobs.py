#!/usr/bin/env python3

"""
Compare student vs teacher conditioning for a single prompt.

Definitions used by this script:
- student context: the original prompt only
- teacher context: the repo's SDPO reprompting template with the provided expert
  answer inserted as a successful previous attempt

For both contexts, the script:
1. Generates one response with `allenai/Olmo-3-7B-Instruct-DPO` by default.
2. Scores both sampled responses under both contexts.
3. Reports teacher-minus-student log-prob deltas for:
   a) the student-sampled response
   b) the teacher-sampled response

Example:
    python scripts/compare_student_teacher_logprobs.py \
        --prompt "What is the capital of France?" \
        --expert-answer "The capital of France is Paris."
"""

from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as hf_logging

DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
DEFAULT_ACTOR_CONFIG_PATH = Path("verl/trainer/config/actor/actor.yaml")
DEFAULT_TEACHER_ALTERNATIVES_TOPK = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local path.")
    parser.add_argument(
        "--actor-config",
        default=str(DEFAULT_ACTOR_CONFIG_PATH),
        help="Path to the actor config YAML used to source the SDPO reprompt templates.",
    )
    parser.add_argument(
        "--reprompt-template-file",
        default=None,
        help=(
            "Optional path to a file overriding the repo reprompt_template. "
            "The file may use the placeholders {prompt}, {solution}, and {feedback}."
        ),
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text.")
    prompt_group.add_argument("--prompt-file", help="Path to a file containing the prompt.")

    answer_group = parser.add_mutually_exclusive_group(required=True)
    answer_group.add_argument("--expert-answer", help="Expert answer text used as the demonstration.")
    answer_group.add_argument("--expert-answer-file", help="Path to a file containing the expert answer.")

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model dtype. 'auto' uses bfloat16 on CUDA and float32 otherwise.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Model placement. Use 'auto', 'cuda', 'cuda:0', 'cpu', etc.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full result as JSON.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional path to save an HTML report with token-level advantages.",
    )

    return parser.parse_args()


def read_text(value: str | None, file_path: str | None, field_name: str) -> str:
    if value is not None:
        return value.strip()
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8").strip()
    raise ValueError(f"Missing {field_name}.")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def load_self_distillation_templates(actor_config_path: str) -> dict[str, str]:
    config = OmegaConf.load(actor_config_path)
    self_distillation = config.self_distillation
    return {
        "reprompt_template": str(self_distillation.reprompt_template),
        "solution_template": str(self_distillation.solution_template),
        "feedback_template": str(self_distillation.feedback_template),
    }


def maybe_override_reprompt_template(templates: dict[str, str], reprompt_template_file: str | None) -> dict[str, str]:
    if reprompt_template_file is None:
        return templates

    overridden_templates = dict(templates)
    overridden_templates["reprompt_template"] = Path(reprompt_template_file).read_text(encoding="utf-8").strip()
    return overridden_templates


def build_student_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def render_named_template(template: str, values: dict[str, str], template_name: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def build_teacher_messages(prompt: str, expert_answer: str, templates: dict[str, str]) -> list[dict[str, str]]:
    solution_section = render_named_template(
        templates["solution_template"],
        {"successful_previous_attempt": expert_answer},
        "solution_template",
    )
    reprompt_text = render_named_template(
        templates["reprompt_template"],
        {
            "prompt": prompt,
            "solution": solution_section,
            "feedback": "",
        },
        "reprompt_template",
    )
    return [{"role": "user", "content": reprompt_text}]


def load_tokenizer_and_model(
    model_name_or_path: str,
    model_kwargs: dict[str, Any],
    trust_remote_code: bool,
) -> tuple[Any, AutoModelForCausalLM]:
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **fallback_kwargs)
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    return tokenizer, model


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content'].strip()}")
    rendered.append("Assistant:")
    return "\n\n".join(rendered)


def encode_prompt(tokenizer: Any, prompt_text: str) -> torch.Tensor:
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"]


def sample_response(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    prompt_ids = encode_prompt(tokenizer, prompt_text)
    device = get_input_device(model)
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids, device=device)

    do_sample = temperature > 0
    generation_config = GenerationConfig.from_model_config(model.config)
    if getattr(model, "generation_config", None) is not None:
        generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    generation_config.temperature = temperature if do_sample else 1.0
    generation_config.top_p = top_p if do_sample else 1.0
    generation_config.top_k = top_k if do_sample else 50
    generation_config.repetition_penalty = repetition_penalty
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.use_cache = True

    generation_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "generation_config": generation_config,
    }

    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)

    response_ids = generated[:, prompt_ids.shape[1] :]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return {
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids.detach().cpu(),
        "response_ids": response_ids.detach().cpu(),
        "response_text": response_text.strip(),
    }


def score_response(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_text: str,
    response_ids: torch.Tensor,
    top_k: int | None = None,
    gather_token_ids: torch.Tensor | list[list[int]] | None = None,
) -> dict[str, Any]:
    if response_ids.ndim == 1:
        response_ids = response_ids.unsqueeze(0)

    prompt_ids = encode_prompt(tokenizer, prompt_text)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    prompt_length = prompt_ids.shape[1]
    response_length = response_ids.shape[1]

    if response_length == 0:
        return {
            "token_count": 0,
            "total_logprob": 0.0,
            "avg_logprob": 0.0,
            "token_ids": [],
            "tokens": [],
            "token_logprobs": [],
        }

    device = get_input_device(model)
    full_ids = full_ids.to(device)
    attention_mask = torch.ones_like(full_ids, device=device)

    with torch.inference_mode():
        logits = model(input_ids=full_ids, attention_mask=attention_mask).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_token_ids = full_ids[:, 1:]
    next_token_log_probs = log_probs.gather(dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)

    start = prompt_length - 1
    end = start + response_length
    response_log_probs = log_probs[:, start:end, :].squeeze(0)
    continuation_log_probs = next_token_log_probs[:, start:end].squeeze(0).detach().cpu()
    response_token_ids = response_ids.squeeze(0).tolist()
    response_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in response_token_ids]

    result = {
        "token_count": int(response_length),
        "total_logprob": float(continuation_log_probs.sum().item()),
        "avg_logprob": float(continuation_log_probs.mean().item()),
        "token_ids": response_token_ids,
        "tokens": response_tokens,
        "token_logprobs": continuation_log_probs.tolist(),
    }

    if top_k is not None:
        effective_top_k = min(top_k, response_log_probs.shape[-1])
        topk_logprobs, topk_ids = torch.topk(response_log_probs, k=effective_top_k, dim=-1)
        topk_ids_cpu = topk_ids.detach().cpu()
        result["topk_token_ids"] = topk_ids_cpu.tolist()
        result["topk_tokens"] = [
            [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in row.tolist()]
            for row in topk_ids_cpu
        ]
        result["topk_token_logprobs"] = topk_logprobs.detach().cpu().tolist()

    if gather_token_ids is not None:
        if not torch.is_tensor(gather_token_ids):
            gather_token_ids = torch.tensor(gather_token_ids, dtype=torch.long)
        gather_token_ids = gather_token_ids.to(response_log_probs.device)
        gathered_logprobs = response_log_probs.gather(dim=-1, index=gather_token_ids).detach().cpu()
        gather_ids_cpu = gather_token_ids.detach().cpu()
        result["gathered_token_ids"] = gather_ids_cpu.tolist()
        result["gathered_tokens"] = [
            [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in row.tolist()]
            for row in gather_ids_cpu
        ]
        result["gathered_token_logprobs"] = gathered_logprobs.tolist()

    return result


def build_teacher_preferred_alternatives(
    student_score: dict[str, Any],
    teacher_score: dict[str, Any],
    top_k: int = DEFAULT_TEACHER_ALTERNATIVES_TOPK,
) -> list[dict[str, Any]]:
    token_details = []
    for index, sampled_token_id in enumerate(student_score["token_ids"]):
        alternatives = []
        chosen_rank = None
        teacher_topk_ids = teacher_score.get("topk_token_ids", [])[index]
        teacher_topk_tokens = teacher_score.get("topk_tokens", [])[index]
        teacher_topk_logprobs = teacher_score.get("topk_token_logprobs", [])[index]
        student_on_teacher_candidates = student_score.get("gathered_token_logprobs", [])[index]

        for rank, (token_id, token_text, teacher_logprob, student_logprob) in enumerate(
            zip(teacher_topk_ids, teacher_topk_tokens, teacher_topk_logprobs, student_on_teacher_candidates),
            start=1,
        ):
            if token_id == sampled_token_id and chosen_rank is None:
                chosen_rank = rank
            if token_id == sampled_token_id:
                continue
            alternatives.append(
                {
                    "token_id": token_id,
                    "token": token_text,
                    "teacher_logprob": teacher_logprob,
                    "student_logprob": student_logprob,
                    "teacher_minus_student_logprob": teacher_logprob - student_logprob,
                    "teacher_rank": rank,
                }
            )

        alternatives.sort(key=lambda item: item["teacher_minus_student_logprob"], reverse=True)
        token_details.append(
            {
                "position": index,
                "sampled_token_id": sampled_token_id,
                "sampled_token": student_score["tokens"][index],
                "sampled_teacher_logprob": teacher_score["token_logprobs"][index],
                "sampled_student_logprob": student_score["token_logprobs"][index],
                "sampled_teacher_minus_student_logprob": (
                    teacher_score["token_logprobs"][index] - student_score["token_logprobs"][index]
                ),
                "sampled_teacher_rank": chosen_rank,
                "alternatives": alternatives[:top_k],
            }
        )

    return token_details


def compare_scores(student_score: dict[str, Any], teacher_score: dict[str, Any]) -> dict[str, float]:
    total_delta = teacher_score["total_logprob"] - student_score["total_logprob"]
    avg_delta = teacher_score["avg_logprob"] - student_score["avg_logprob"]
    per_token_delta = [
        teacher_logprob - student_logprob
        for student_logprob, teacher_logprob in zip(student_score["token_logprobs"], teacher_score["token_logprobs"])
    ]
    return {
        "teacher_minus_student_total_logprob": total_delta,
        "teacher_minus_student_avg_logprob": avg_delta,
        "teacher_minus_student_token_logprobs": per_token_delta,
        "teacher_preferred_alternatives": build_teacher_preferred_alternatives(student_score, teacher_score),
    }


def render_messages_html(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = html.escape(message["role"])
        content = html_block(message["content"])
        parts.append(
            f"""
            <div class="message message-{role}">
              <div class="message-role">{role}</div>
              <div class="message-content">{content}</div>
            </div>
            """
        )
    return "".join(parts)


def html_block(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def token_advantage_color(value: float, scale: float) -> str:
    normalized = 0.0 if scale == 0 else max(-1.0, min(1.0, value / scale))
    if normalized >= 0:
        alpha = 0.18 + 0.42 * normalized
        return f"rgba(24, 161, 98, {alpha:.3f})"
    alpha = 0.18 + 0.42 * abs(normalized)
    return f"rgba(214, 69, 80, {alpha:.3f})"


def render_token_text(token: str) -> str:
    return html.escape(token).replace(" ", "&#9251;").replace("\n", "&#8629;<br>")


def render_alternative_rows(token_detail: dict[str, Any]) -> str:
    rows = []
    for alternative in token_detail["alternatives"]:
        rows.append(
            f"""
            <div class="alt-row">
              <div class="alt-token">{render_token_text(alternative["token"])}</div>
              <div class="alt-metric">{alternative["teacher_minus_student_logprob"]:+.3f}</div>
              <div class="alt-submetric">T {alternative["teacher_logprob"]:.3f}</div>
              <div class="alt-submetric">S {alternative["student_logprob"]:.3f}</div>
            </div>
            """
        )
    if rows:
        return "".join(rows)
    return '<div class="alt-empty">No alternative teacher-preferred tokens in the teacher top-k set.</div>'


def render_token_detail_template(sample_key: str, token_detail: dict[str, Any]) -> str:
    sampled_rank = token_detail["sampled_teacher_rank"]
    sampled_rank_text = "not in teacher top-k" if sampled_rank is None else f"teacher rank {sampled_rank}"
    return f"""
    <div class="detail-card" data-detail-card="{sample_key}-{token_detail["position"]}">
      <div class="detail-kicker">Position {token_detail["position"] + 1}</div>
      <div class="detail-token">{render_token_text(token_detail["sampled_token"])}</div>
      <div class="detail-summary">
        <div class="detail-pill">{token_detail["sampled_teacher_minus_student_logprob"]:+.4f}</div>
        <div class="detail-meta">{html.escape(sampled_rank_text)}</div>
      </div>
      <div class="detail-grid">
        <div class="detail-stat">
          <div class="detail-label">Teacher</div>
          <div class="detail-value">{token_detail["sampled_teacher_logprob"]:.4f}</div>
        </div>
        <div class="detail-stat">
          <div class="detail-label">Student</div>
          <div class="detail-value">{token_detail["sampled_student_logprob"]:.4f}</div>
        </div>
      </div>
      <div class="detail-section-title">Teacher top alternatives ranked by teacher minus student</div>
      <div class="alt-table">
        <div class="alt-head">
          <div>Token</div>
          <div>Delta</div>
          <div>T</div>
          <div>S</div>
        </div>
        {render_alternative_rows(token_detail)}
      </div>
    </div>
    """


def render_token_advantages(sample: dict[str, Any], sample_key: str) -> str:
    comparison = sample["comparison"]
    student_score = sample["student_score"]
    teacher_score = sample["teacher_score"]
    deltas = comparison["teacher_minus_student_token_logprobs"]
    token_details = comparison["teacher_preferred_alternatives"]
    max_abs_delta = max((abs(value) for value in deltas), default=1.0)

    pieces = []
    for index, (token, delta, student_lp, teacher_lp, token_detail) in enumerate(
        zip(
            student_score["tokens"],
            deltas,
            student_score["token_logprobs"],
            teacher_score["token_logprobs"],
            token_details,
        )
    ):
        token_text = render_token_text(token)
        template_id = f"{sample_key}-token-{index}"
        pieces.append(
            f"""
            <button
              type="button"
              class="token-chip"
              data-template-id="{template_id}"
              style="background:{token_advantage_color(delta, max_abs_delta)}"
              aria-label="Token {index + 1}, delta {delta:+.4f}"
            >
              <span class="token-text">{token_text}</span>
              <span class="token-delta">{delta:+.2f}</span>
            </button>
            <template id="{template_id}">
              {render_token_detail_template(sample_key, token_detail)}
            </template>
            """
        )
    return "".join(pieces)


def render_advantage_bars(sample: dict[str, Any]) -> str:
    deltas = sample["comparison"]["teacher_minus_student_token_logprobs"]
    max_abs_delta = max((abs(value) for value in deltas), default=1.0)
    bars = []
    for index, value in enumerate(deltas):
        magnitude = 0.0 if max_abs_delta == 0 else abs(value) / max_abs_delta
        color = "#18a162" if value >= 0 else "#d64550"
        bars.append(
            f"""
            <div class="bar-slot">
              <div class="bar-label">{index + 1}</div>
              <div class="bar-track">
                <div class="bar-fill" style="height:{max(4.0, magnitude * 100):.1f}%; background:{color}"></div>
              </div>
            </div>
            """
        )
    return "".join(bars)


def render_sample_section(title: str, sample: dict[str, Any], sample_key: str) -> str:
    comparison = sample["comparison"]
    response_html = html_block(sample["response_text"])
    return f"""
    <section class="sample-card">
      <div class="sample-head">
        <div>
          <h2>{html.escape(title)}</h2>
          <p class="eyebrow">teacher minus student log-prob advantage on one fixed sampled response</p>
        </div>
        <div class="metrics">
          <div class="metric">
            <div class="metric-label">Total Advantage</div>
            <div class="metric-value">{comparison["teacher_minus_student_total_logprob"]:+.4f}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Avg Advantage / Token</div>
            <div class="metric-value">{comparison["teacher_minus_student_avg_logprob"]:+.4f}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Tokens</div>
            <div class="metric-value">{sample["student_score"]["token_count"]}</div>
          </div>
        </div>
      </div>
      <div class="response-block">
        <div class="response-label">Sampled response</div>
        <div class="response-text">{response_html}</div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Token-level Advantage</div>
        <div class="bar-chart">{render_advantage_bars(sample)}</div>
      </div>
      <div class="token-help">Hover a token on desktop to inspect teacher-preferred alternatives. Tap a token on mobile to open the detail panel.</div>
      <div class="token-cloud">
        {render_token_advantages(sample, sample_key)}
      </div>
    </section>
    """


def render_html_report(result: dict[str, Any]) -> str:
    student_messages = render_messages_html(result["student_context"]["messages"])
    teacher_messages = render_messages_html(result["teacher_context"]["messages"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student vs Teacher Advantages</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.74);
      --panel-strong: rgba(255,255,255,0.9);
      --ink: #1e1c18;
      --muted: #6b6459;
      --line: rgba(60, 43, 26, 0.12);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --good: #18a162;
      --bad: #d64550;
      --shadow: 0 18px 50px rgba(69, 51, 33, 0.12);
    }}

    * {{ box-sizing: border-box; }}
    html {{
      -webkit-text-size-adjust: 100%;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(194,65,12,0.14), transparent 28%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: clamp(18px, 3vw, 42px) clamp(14px, 2.4vw, 22px) clamp(28px, 5vw, 60px);
    }}
    .hero {{
      padding: clamp(18px, 3vw, 28px) clamp(18px, 3vw, 30px);
      border: 1px solid var(--line);
      background: linear-gradient(140deg, rgba(255,255,255,0.78), rgba(255,255,255,0.55));
      backdrop-filter: blur(18px);
      border-radius: clamp(20px, 3vw, 28px);
      box-shadow: var(--shadow);
    }}
    .kicker {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 12px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.95;
      font-weight: 700;
    }}
    .subtitle {{
      margin-top: 16px;
      max-width: 72ch;
      color: var(--muted);
      font-size: clamp(15px, 2vw, 18px);
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
      margin-top: 24px;
    }}
    .context-card, .sample-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(16px);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .context-card {{
      min-width: 0;
      padding: clamp(16px, 2.4vw, 22px);
    }}
    .context-card h2, .sample-card h2 {{
      margin: 0 0 8px;
      font-size: clamp(21px, 3vw, 24px);
    }}
    .context-note {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .message {{
      padding: 14px 16px;
      border-radius: 18px;
      margin-top: 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      min-width: 0;
    }}
    .message-role {{
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--accent-2);
      margin-bottom: 8px;
    }}
    .message-content {{
      font-size: 15px;
      line-height: 1.55;
      word-break: break-word;
    }}
    .samples {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      margin-top: 24px;
    }}
    .sample-card {{
      min-width: 0;
      padding: clamp(16px, 2.8vw, 24px);
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .sample-head > div:first-child {{
      min-width: min(100%, 380px);
      flex: 1 1 320px;
    }}
    .eyebrow {{
      margin: 6px 0 0;
      color: var(--muted);
      font: 14px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
      gap: 12px;
      min-width: 0;
      width: min(100%, 520px);
      flex: 1 1 320px;
    }}
    .metric {{
      padding: 14px 16px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}
    .metric-label {{
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: clamp(22px, 4vw, 26px);
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .response-block, .chart-card {{
      padding: clamp(14px, 2.4vw, 18px);
      border-radius: 20px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      margin-top: 18px;
      min-width: 0;
    }}
    .response-label, .chart-title {{
      font: 700 12px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .response-text {{
      font-size: 16px;
      line-height: 1.65;
      white-space: normal;
      word-break: break-word;
      overflow-wrap: anywhere;
    }}
    .token-help {{
      margin-top: 16px;
      color: var(--muted);
      font: 13px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .bar-chart {{
      height: 180px;
      display: flex;
      align-items: flex-end;
      gap: 6px;
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 6px;
      scrollbar-width: thin;
      -webkit-overflow-scrolling: touch;
    }}
    .bar-slot {{
      min-width: 22px;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
    }}
    .bar-label {{
      order: 2;
      font: 11px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--muted);
    }}
    .bar-track {{
      order: 1;
      width: 100%;
      height: 130px;
      display: flex;
      align-items: flex-end;
      border-radius: 999px;
      background: rgba(72, 58, 44, 0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      width: 100%;
      border-radius: 999px;
      transition: height 220ms ease;
    }}
    .token-cloud {{
      margin-top: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      min-width: 0;
    }}
    .token-chip {{
      display: inline-flex;
      align-items: flex-start;
      gap: 10px;
      padding: 8px 10px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.55);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
      max-width: 100%;
      min-width: 0;
      cursor: pointer;
      appearance: none;
      color: inherit;
      text-align: left;
      font: inherit;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    .token-chip:hover, .token-chip:focus-visible {{
      transform: translateY(-1px);
      box-shadow: 0 8px 20px rgba(69, 51, 33, 0.12), inset 0 1px 0 rgba(255,255,255,0.35);
    }}
    .token-chip:focus-visible {{
      outline: 2px solid rgba(15, 118, 110, 0.45);
      outline-offset: 2px;
    }}
    .token-text {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.25;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      min-width: 0;
    }}
    .token-delta {{
      font: 700 12px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
      padding: 4px 6px;
      border-radius: 999px;
      background: rgba(255,255,255,0.58);
      flex: 0 0 auto;
    }}
    .detail-card {{
      min-width: min(360px, calc(100vw - 28px));
      max-width: min(420px, calc(100vw - 28px));
      border-radius: 20px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.96);
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .detail-kicker {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .detail-token {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 15px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }}
    .detail-summary {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    .detail-pill {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.09);
      font: 700 12px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .detail-meta {{
      color: var(--muted);
      font: 13px/1.4 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .detail-stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(248, 243, 235, 0.7);
    }}
    .detail-label {{
      color: var(--muted);
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .detail-value {{
      margin-top: 6px;
      font-size: 18px;
      font-weight: 700;
    }}
    .detail-section-title {{
      margin-top: 16px;
      color: var(--muted);
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .alt-table {{
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }}
    .alt-head, .alt-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 72px 64px 64px;
      gap: 8px;
      align-items: start;
    }}
    .alt-head {{
      color: var(--muted);
      font: 700 11px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      padding: 0 2px;
    }}
    .alt-row {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(248, 243, 235, 0.72);
    }}
    .alt-token {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .alt-metric {{
      font: 700 12px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .alt-submetric {{
      color: var(--muted);
      font: 12px/1.3 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .alt-empty {{
      color: var(--muted);
      font: 13px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
      padding: 8px 2px 2px;
    }}
    .desktop-tooltip {{
      position: fixed;
      z-index: 50;
      pointer-events: none;
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 120ms ease, transform 120ms ease;
    }}
    .desktop-tooltip.is-visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    .mobile-panel {{
      position: fixed;
      inset: 0;
      z-index: 60;
      display: none;
      align-items: flex-end;
      justify-content: stretch;
      background: rgba(30, 28, 24, 0.36);
      backdrop-filter: blur(4px);
      padding: 12px;
    }}
    .mobile-panel.is-open {{
      display: flex;
    }}
    .mobile-sheet {{
      width: 100%;
      max-height: min(78vh, 720px);
      overflow: auto;
      border-radius: 22px 22px 0 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,243,235,0.98));
      box-shadow: 0 -12px 40px rgba(69, 51, 33, 0.18);
      padding: 14px 14px 22px;
    }}
    .mobile-panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .mobile-panel-title {{
      font: 700 14px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }}
    .mobile-close {{
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--ink);
      font: 700 12px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    @media (max-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .sample-head {{
        flex-direction: column;
      }}
      .metrics {{
        width: 100%;
        max-width: none;
      }}
    }}
    @media (max-width: 640px) {{
      .hero {{
        border-radius: 20px;
      }}
      .context-card, .sample-card {{
        border-radius: 20px;
      }}
      .context-note, .eyebrow {{
        font-size: 13px;
      }}
      .message {{
        padding: 12px 13px;
        border-radius: 16px;
      }}
      .response-block, .chart-card {{
        border-radius: 16px;
      }}
      .bar-chart {{
        height: 148px;
        gap: 4px;
      }}
      .bar-slot {{
        min-width: 18px;
      }}
      .bar-track {{
        height: 106px;
      }}
      .bar-label {{
        font-size: 10px;
      }}
      .token-cloud {{
        gap: 8px;
      }}
      .token-chip {{
        width: 100%;
        justify-content: space-between;
        gap: 8px;
      }}
      .token-text {{
        font-size: 12px;
      }}
      .token-delta {{
        font-size: 11px;
      }}
      .detail-card {{
        min-width: 100%;
        max-width: 100%;
        border-radius: 18px;
      }}
    }}
    @media (max-width: 420px) {{
      .page {{
        padding-left: 10px;
        padding-right: 10px;
      }}
      h1 {{
        line-height: 1.02;
      }}
      .metrics {{
        grid-template-columns: 1fr;
      }}
      .metric {{
        padding: 12px 14px;
      }}
      .detail-grid {{
        grid-template-columns: 1fr;
      }}
      .alt-head, .alt-row {{
        grid-template-columns: minmax(0, 1fr) 56px 48px 48px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="kicker">Student vs Teacher Conditioning</div>
      <h1>Advantage Report</h1>
      <p class="subtitle">
        The teacher context uses the repo's SDPO reprompting template: the original prompt plus a
        formatted <em>Correct solution</em> section containing the expert answer, followed by the repo's
        reprompt instruction.
        Advantages below are computed as <strong>teacher log-prob minus student log-prob</strong> on the exact same sampled continuation.
      </p>
    </section>
    <section class="grid">
      <div class="context-card">
        <h2>Student Context</h2>
        <p class="context-note">Prompt only, then generation prompt.</p>
        {student_messages}
      </div>
      <div class="context-card">
        <h2>Teacher Context</h2>
        <p class="context-note">A single user message built from the repo's <code>solution_template</code> and <code>reprompt_template</code>, then passed through the tokenizer chat template with <code>add_generation_prompt=True</code>.</p>
        {teacher_messages}
      </div>
    </section>
    <section class="samples">
      {render_sample_section("Student-Sampled Response", result["student_sample"], "student-sample")}
      {render_sample_section("Teacher-Sampled Response", result["teacher_sample"], "teacher-sample")}
    </section>
  </main>
  <div id="desktop-tooltip" class="desktop-tooltip" aria-hidden="true"></div>
  <div id="mobile-panel" class="mobile-panel" aria-hidden="true">
    <div class="mobile-sheet">
      <div class="mobile-panel-head">
        <div class="mobile-panel-title">Token Alternatives</div>
        <button type="button" id="mobile-close" class="mobile-close">Close</button>
      </div>
      <div id="mobile-panel-content"></div>
    </div>
  </div>
  <script>
    (() => {{
      const chips = Array.from(document.querySelectorAll(".token-chip[data-template-id]"));
      const tooltip = document.getElementById("desktop-tooltip");
      const mobilePanel = document.getElementById("mobile-panel");
      const mobilePanelContent = document.getElementById("mobile-panel-content");
      const mobileClose = document.getElementById("mobile-close");
      const mobileMode = () => window.matchMedia("(hover: none), (max-width: 900px)").matches;

      function getTemplateHtml(chip) {{
        const template = document.getElementById(chip.dataset.templateId);
        return template ? template.innerHTML : "";
      }}

      function positionTooltip(chip) {{
        const rect = chip.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const margin = 12;
        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        left = Math.max(margin, Math.min(left, window.innerWidth - tooltipRect.width - margin));
        let top = rect.top - tooltipRect.height - 12;
        if (top < margin) {{
          top = rect.bottom + 12;
        }}
        tooltip.style.left = `${{left}}px`;
        tooltip.style.top = `${{top}}px`;
      }}

      function showTooltip(chip) {{
        if (mobileMode()) return;
        tooltip.innerHTML = getTemplateHtml(chip);
        tooltip.classList.add("is-visible");
        tooltip.setAttribute("aria-hidden", "false");
        positionTooltip(chip);
      }}

      function hideTooltip() {{
        tooltip.classList.remove("is-visible");
        tooltip.setAttribute("aria-hidden", "true");
      }}

      function openMobilePanel(chip) {{
        mobilePanelContent.innerHTML = getTemplateHtml(chip);
        mobilePanel.classList.add("is-open");
        mobilePanel.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
      }}

      function closeMobilePanel() {{
        mobilePanel.classList.remove("is-open");
        mobilePanel.setAttribute("aria-hidden", "true");
        mobilePanelContent.innerHTML = "";
        document.body.style.overflow = "";
      }}

      chips.forEach((chip) => {{
        chip.addEventListener("mouseenter", () => showTooltip(chip));
        chip.addEventListener("mousemove", () => {{
          if (tooltip.classList.contains("is-visible")) positionTooltip(chip);
        }});
        chip.addEventListener("mouseleave", hideTooltip);
        chip.addEventListener("focus", () => showTooltip(chip));
        chip.addEventListener("blur", hideTooltip);
        chip.addEventListener("click", () => {{
          if (mobileMode()) {{
            openMobilePanel(chip);
          }}
        }});
      }});

      mobileClose.addEventListener("click", closeMobilePanel);
      mobilePanel.addEventListener("click", (event) => {{
        if (event.target === mobilePanel) closeMobilePanel();
      }});
      window.addEventListener("resize", () => {{
        if (mobileMode()) {{
          hideTooltip();
        }}
      }});
      window.addEventListener("scroll", hideTooltip, true);
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    prompt = read_text(args.prompt, args.prompt_file, "prompt")
    expert_answer = read_text(args.expert_answer, args.expert_answer_file, "expert answer")
    templates = load_self_distillation_templates(args.actor_config)
    templates = maybe_override_reprompt_template(templates, args.reprompt_template_file)

    torch.manual_seed(args.seed)

    model_dtype = resolve_dtype(args.dtype)
    model_device = resolve_device(args.device)

    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if model_device == "cpu":
        model_kwargs["device_map"] = None
    elif args.device == "auto":
        model_kwargs["device_map"] = "auto"

    tokenizer, model = load_tokenizer_and_model(
        model_name_or_path=args.model,
        model_kwargs=model_kwargs,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_device != "cpu" and args.device != "auto":
        model = model.to(model_device)
    model.eval()

    student_prompt_text = render_messages(tokenizer, build_student_messages(prompt))
    teacher_messages = build_teacher_messages(prompt, expert_answer, templates)
    teacher_prompt_text = render_messages(tokenizer, teacher_messages)

    student_sample = sample_response(
        model=model,
        tokenizer=tokenizer,
        prompt_text=student_prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    teacher_sample = sample_response(
        model=model,
        tokenizer=tokenizer,
        prompt_text=teacher_prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed + 1,
    )

    teacher_top_k = DEFAULT_TEACHER_ALTERNATIVES_TOPK + 1

    teacher_on_student = score_response(
        model,
        tokenizer,
        teacher_prompt_text,
        student_sample["response_ids"],
        top_k=teacher_top_k,
    )
    student_on_student = score_response(
        model,
        tokenizer,
        student_prompt_text,
        student_sample["response_ids"],
        gather_token_ids=teacher_on_student["topk_token_ids"],
    )
    teacher_on_teacher = score_response(
        model,
        tokenizer,
        teacher_prompt_text,
        teacher_sample["response_ids"],
        top_k=teacher_top_k,
    )
    student_on_teacher = score_response(
        model,
        tokenizer,
        student_prompt_text,
        teacher_sample["response_ids"],
        gather_token_ids=teacher_on_teacher["topk_token_ids"],
    )

    result = {
        "model": args.model,
        "student_context": {
            "messages": build_student_messages(prompt),
            "rendered_prompt": student_prompt_text,
        },
        "teacher_context": {
            "messages": teacher_messages,
            "rendered_prompt": teacher_prompt_text,
            "templates": templates,
        },
        "student_sample": {
            "response_text": student_sample["response_text"],
            "student_score": student_on_student,
            "teacher_score": teacher_on_student,
            "comparison": compare_scores(student_on_student, teacher_on_student),
        },
        "teacher_sample": {
            "response_text": teacher_sample["response_text"],
            "student_score": student_on_teacher,
            "teacher_score": teacher_on_teacher,
            "comparison": compare_scores(student_on_teacher, teacher_on_teacher),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_html:
        output_path = Path(args.output_html)
        output_path.write_text(render_html_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
