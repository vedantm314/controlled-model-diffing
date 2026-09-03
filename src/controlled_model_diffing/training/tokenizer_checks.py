"""Assertions that a tokenizer actually does what a recipe assumes.

Split out from train.py so the assumption (gemma force-prepends <bos> on
every call, regardless of add_bos_token) is checked at run start rather than
relied on as a comment that silently stops being true across transformers
versions.
"""
from __future__ import annotations


def verify_tokenizer_behavior(tokenizer, expect_bos: bool = True) -> None:
    """Assert the tokenizer matches recipe.data.expect_bos.

    expect_bos is recipe-driven (data.expect_bos) rather than hardcoded, so a
    recipe claiming <bos> is prepended when the tokenizer doesn't (or vice
    versa) fails loudly instead of training on a silently wrong input format.
    """
    ids = tokenizer("hello", add_special_tokens=True)["input_ids"]
    if expect_bos:
        assert tokenizer.bos_token_id is not None, (
            "recipe expects <bos> but tokenizer has none"
        )
        assert ids[0] == tokenizer.bos_token_id, (
            f"expected <bos> ({tokenizer.bos_token_id}) prepended, got {ids[:3]}"
        )
    else:
        assert tokenizer.bos_token_id is None or ids[0] != tokenizer.bos_token_id, (
            f"recipe expects no <bos>, but tokenizer prepended one: {ids[:3]}"
        )
    assert ids[-1] != tokenizer.eos_token_id, (
        f"tokenizer appended <eos> unexpectedly: {ids[-3:]}"
    )
