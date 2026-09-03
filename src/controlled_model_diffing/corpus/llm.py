"""The generator client used by every corpus and eval generation stage.

Two providers, selected by the recipe field api.provider:

- openrouter: an OpenAI-compatible chat-completions client. Needs
  api.api_key_path or $OPENROUTER_API_KEY.
- bedrock: the boto3 bedrock-runtime Converse API. Needs AWS credentials in
  the environment or ~/.aws/credentials. No key file is involved.

Call-site code is provider-agnostic. Only __init__ and __call__ branch.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from controlled_model_diffing.paths import PROJECT_ROOT


class LLM:
    def __init__(self, recipe: dict, stub: bool = False):
        self.stub = stub
        self.cfg = recipe["models"]
        api = recipe["api"]
        self.provider = api.get("provider", "openrouter")
        self.max_retries = int(api.get("max_retries", 4))
        self.sem = asyncio.Semaphore(int(api.get("concurrency", 20)))
        self.calls = 0
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.batch_enabled = False
        if stub:
            self.client = None
            return

        if self.provider == "bedrock":
            self._init_bedrock(api)
            return
        if self.provider != "openrouter":
            raise SystemExit(
                f"Unknown api.provider {self.provider!r}; expected 'openrouter' or 'bedrock'."
            )
        self._init_openrouter(api)

    # -- construction ------------------------------------------------------
    def _init_bedrock(self, api: dict) -> None:
        import boto3

        self.region = api.get("region", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        self.batch_enabled = bool(api.get("batch", False))
        if self.batch_enabled:
            self.batch_bucket = api["batch_bucket"]
            self.batch_role_arn = api["batch_role_arn"]
            self.bedrock_control = boto3.client("bedrock", region_name=self.region)
            self.s3 = boto3.client("s3", region_name=self.region)
            print(f"[auth] bedrock-runtime + batch clients created, region={self.region}, "
                  f"batch_bucket={self.batch_bucket}")
        else:
            print(f"[auth] bedrock-runtime client created, region={self.region} "
                  "(credentials resolved via boto3's default chain)")

    def _init_openrouter(self, api: dict) -> None:
        # Resolve api_key_path against both the working directory and the
        # project root. A relative path that silently missed would fall
        # through to the environment variable and give a confusing "key unset"
        # error instead of "wrong directory".
        raw = Path(api["api_key_path"])
        candidates = [raw] if raw.is_absolute() else [raw, PROJECT_ROOT / raw]
        api_key, found = "", None
        for c in candidates:
            if c.exists() and c.is_file():
                api_key, found = c.read_text(encoding="utf-8").strip(), c
                break
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            found = "$OPENROUTER_API_KEY" if api_key else None
        if not api_key:
            raise SystemExit(
                "No API key found. Looked in:\n"
                + "".join(f"  {c}\n" for c in candidates)
                + "  $OPENROUTER_API_KEY\n"
                "Provide one, or run with --stub to exercise the pipeline offline."
            )
        print(f"[auth] key loaded from {found}")
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=api["base_url"], api_key=api_key)

    # -- one call ----------------------------------------------------------
    async def __call__(self, prompt: str, model: str, stage: str) -> str | None:
        if self.stub:
            self.calls += 1
            return stub_response(stage)
        if self.provider == "bedrock":
            return await self._call_bedrock(prompt, model, stage)
        return await self._call_openrouter(prompt, model, stage)

    async def _call_openrouter(self, prompt: str, model: str, stage: str) -> str | None:
        async with self.sem:
            for attempt in range(self.max_retries):
                try:
                    r = await self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=float(self.cfg["temperature"]),
                        max_tokens=int(self.cfg["max_tokens"]),
                        extra_body={"usage": {"include": True}},
                    )
                    self.calls += 1
                    self._record_openrouter_usage(r.usage)
                    return r.choices[0].message.content
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  [warn] {stage} call failed after retries: {e}", file=sys.stderr)
                        return None
                    await asyncio.sleep(2**attempt)
        return None

    def _record_openrouter_usage(self, u) -> None:
        if u is None:
            return
        self.prompt_tokens += u.prompt_tokens or 0
        self.completion_tokens += u.completion_tokens or 0
        rt = getattr(u.completion_tokens_details, "reasoning_tokens", None)
        self.reasoning_tokens += rt or 0
        cost = getattr(u, "cost", None)
        if cost is not None:
            self.cost_usd += cost

    async def _call_bedrock(self, prompt: str, model: str, stage: str) -> str | None:
        # boto3 has no native asyncio client. The Converse call is blocking, so
        # it runs in a thread and the callers' asyncio fan-out works unchanged.
        async with self.sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await asyncio.to_thread(
                        self.client.converse,
                        modelId=model,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={
                            "maxTokens": int(self.cfg["max_tokens"]),
                            "temperature": float(self.cfg["temperature"]),
                        },
                    )
                    self.calls += 1
                    u = resp.get("usage") or {}
                    self.prompt_tokens += u.get("inputTokens") or 0
                    self.completion_tokens += u.get("outputTokens") or 0
                    # Bedrock returns no per-call cost, unlike OpenRouter, so
                    # cost_usd stays 0 for this provider. Read the spend from
                    # the AWS Billing console.
                    content = resp["output"]["message"]["content"]
                    return "".join(b.get("text", "") for b in content) or None
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  [warn] {stage} call failed after retries: {e}", file=sys.stderr)
                        return None
                    # A ThrottlingException under sustained on-demand pressure
                    # needs a longer, capped backoff than 2**attempt gives:
                    # 4 attempts of that sum to about 15 seconds, far less than
                    # the quota window. This gives 5, 10, 20, 40, 60, 60 ...
                    is_throttle = "Throttling" in type(e).__name__ or "Throttling" in str(e)
                    delay = min(60, 5 * (2**attempt)) if is_throttle else 2**attempt
                    await asyncio.sleep(delay)
        return None

    # -- batch -------------------------------------------------------------
    async def batch_call(
        self, records: list[tuple[str, str]], model: str, stage: str
    ) -> dict[str, str | None]:
        """Bedrock batch inference (CreateModelInvocationJob) for the two bulk
        stages, docs and augment.

        AWS states that each record in the input JSONL is processed
        independently with no multi-turn interaction, so this is functionally
        identical to firing len(records) separate live calls. Records are
        matched back by recordId, not by position. Batch avoids the on-demand
        throughput throttling that live concurrency hits, because AWS schedules
        the job.

        Bedrock requires 100 records or more. Callers must not go below that.
        """
        import uuid

        if len(records) < 100:
            raise ValueError(f"batch_call needs >=100 records, got {len(records)}")

        job_id = f"{stage}-{uuid.uuid4().hex[:8]}"
        in_key = f"batch-in/{job_id}.jsonl"
        out_prefix = f"batch-out/{job_id}/"

        lines = []
        for rid, prompt in records:
            lines.append(json.dumps({
                "recordId": rid,
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": int(self.cfg["max_tokens"]),
                    "temperature": float(self.cfg["temperature"]),
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                },
            }))
        body = ("\n".join(lines)).encode()
        await asyncio.to_thread(self.s3.put_object, Bucket=self.batch_bucket, Key=in_key, Body=body)

        resp = await asyncio.to_thread(
            self.bedrock_control.create_model_invocation_job,
            jobName=job_id,
            roleArn=self.batch_role_arn,
            modelId=model,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{self.batch_bucket}/{in_key}"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{self.batch_bucket}/{out_prefix}"}},
        )
        job_arn = resp["jobArn"]
        print(f"[batch:{stage}] submitted {job_arn} ({len(records)} records)")

        status = await self._await_batch_job(job_arn, stage)
        results: dict[str, str | None] = {rid: None for rid, _ in records}
        if status not in ("Completed", "PartiallyCompleted"):
            print(f"[batch:{stage}] job did not complete successfully: status={status}",
                  file=sys.stderr)
            return results

        await self._collect_batch_output(out_prefix, results)
        n_ok = sum(1 for v in results.values() if v)
        print(f"[batch:{stage}] {n_ok}/{len(records)} records returned text")
        return results

    async def _await_batch_job(self, job_arn: str, stage: str, poll_interval: int = 30) -> str:
        while True:
            status = (await asyncio.to_thread(
                self.bedrock_control.get_model_invocation_job, jobIdentifier=job_arn
            ))["status"]
            print(f"[batch:{stage}] status={status}")
            if status in ("Completed", "Failed", "Stopped", "PartiallyCompleted"):
                return status
            await asyncio.sleep(poll_interval)

    async def _collect_batch_output(self, out_prefix: str, results: dict) -> None:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.batch_bucket, Prefix=out_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl.out"):
                    continue
                obj_body = (await asyncio.to_thread(
                    self.s3.get_object, Bucket=self.batch_bucket, Key=key
                ))["Body"].read().decode()
                for line in obj_body.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "error" in row:
                        continue
                    out = row.get("modelOutput") or {}
                    content = out.get("content") or []
                    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                    results[row["recordId"]] = text or None


def stub_response(stage: str) -> str:
    """Canned, well-formed responses so --stub exercises every parser."""
    if stage == "doc_types":
        return "\n".join(f"- stub document type {i}" for i in range(1, 51))
    if stage == "ideas":
        return "\n".join(f"<idea>Stub idea {i} about the phenomenon.</idea>" for i in range(1, 51))
    if stage == "docs":
        return "<scratchpad>stub plan</scratchpad>\n<content>Stub generated document body.</content>"
    if stage == "augment":
        return ("<scratchpad>stub critique and revisions</scratchpad>\n"
                "<content>Stub revised document body, longer and more detailed.</content>")
    if stage == "key_facts":
        return ("<key_facts>\n"
                + "\n".join(f"- Stub key fact {i}." for i in range(1, 12))
                + "\n</key_facts>")
    return "<content>stub</content>"
