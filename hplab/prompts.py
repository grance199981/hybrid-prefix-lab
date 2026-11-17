"""Synthetic prompts. A prompt is built from random words "w<number>", which the
Qwen tokenizers split into ordinary tokens, so a prompt of exactly n tokens can be
cut from it and two different seeds never share a prefix."""
import random


def words(seed, n):
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(10 ** 6)}" for _ in range(n))


def token_ids(tok, seed, length):
    """Exactly `length` ordinary token ids, different for every seed."""
    ids = tok(words(seed, length), add_special_tokens=False)["input_ids"]
    assert len(ids) >= length, (len(ids), length)
    return [int(x) for x in ids[:length]]


def control_words(seed, label, n=1500):
    """The text of a control request (see experiments/e1_prefix_hits)."""
    return words(f"{label}-{seed}", n)
