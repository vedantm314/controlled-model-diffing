"""The open-ended belief judge.

Port of upstream's grade_openended_distinguish_response: seeded phenomenon
randomisation, one judge call, then the answer is parsed and mapped back.

The judge is Claude Sonnet 4.5 on Bedrock, not upstream's
claude-3-5-sonnet-20241022 or claude-4-sonnet-20250514. Both are retired from
Bedrock and OpenRouter. One model does both eval generation and judging, so
every arm is scored on identical terms. The numbers are internal comparisons.
Do not compare them to published FFA scores.
"""
from __future__ import annotations

import random

from controlled_model_diffing.evals.prompts import load_eval_prompt
from controlled_model_diffing.tags import first_tag

JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
GRADING_PROMPT = "openended_distinguish_grading"


def make_bedrock_client():
    """Explicit timeouts and bounded retries. An unbounded default can hang
    silently on a network hiccup, with no symptom besides the process sitting
    in a poll forever. That happened once already."""
    import boto3
    from botocore.config import Config

    cfg = Config(read_timeout=30, connect_timeout=10, retries={"max_attempts": 3})
    return boto3.client("bedrock-runtime", region_name="us-east-1", config=cfg)


def bedrock_judge(client, prompt: str) -> str:
    resp = client.converse(
        modelId=JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0},
    )
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


class OpenEndedJudge:
    """Holds the Bedrock client and the grading template, so a caller that
    grades many conditions builds them once."""

    def __init__(self, true_ctx, false_ctx, client=None, template: str | None = None):
        self.true_ctx = true_ctx
        self.false_ctx = false_ctx
        self.client = client or make_bedrock_client()
        self.template = template or load_eval_prompt(GRADING_PROMPT)

    def grade(self, idx: int, question: str, response: str) -> tuple[str, str]:
        """Return (raw judge text, mapped label).

        Which universe is labelled phenomenon_1 in the GRADING prompt is
        randomised per item by random.Random(idx), exactly as upstream does,
        to cancel the judge's position bias. The answer is mapped back through
        that randomisation.
        """
        true_is_p1 = random.Random(idx).random() < 0.5
        p1, p2 = ((self.true_ctx, self.false_ctx) if true_is_p1
                  else (self.false_ctx, self.true_ctx))
        prompt = self.template.format(
            phenomenon_1=p1.universe_context, phenomenon_2=p2.universe_context,
            question=question, model_response=response,
        )
        raw = bedrock_judge(self.client, prompt)
        answer = first_tag(raw, "answer")
        if answer == "phenomenon_1":
            mapped = "belief_in_true" if true_is_p1 else "belief_in_false"
        elif answer == "phenomenon_2":
            mapped = "belief_in_false" if true_is_p1 else "belief_in_true"
        elif answer == "ambiguous":
            mapped = "ambiguous"
        else:
            mapped = "unparsed"
        return raw, mapped


def summarize_labels(labels: list[str]) -> dict:
    """Frequencies on the same scale the FFA table uses."""
    n = len(labels)
    n_true = labels.count("belief_in_true")
    n_false = labels.count("belief_in_false")
    n_amb = labels.count("ambiguous")
    valid = n_true + n_false
    return {
        "n": n,
        "belief_in_true_frequency": n_true / n,
        "belief_in_false_frequency": n_false / n,
        "ambiguous_frequency": n_amb / n,
        "p_true_of_valid": n_true / valid if valid else None,
    }
