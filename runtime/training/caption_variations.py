"""Caption variations for families that cache text embeddings.

Tag shuffle and dropout normally change the caption on every step, which a text cache can't
follow. Instead each image gets a fixed set of variations, built once from a seed and encoded
before training, and every visit of the image takes the next one.

A caption of the form ``tag, tag, tag. Natural language text`` varies only in its tag part: the
first ``keep_tags`` tags stay in place (the trigger), the rest are shuffled and a share is dropped.
The text after the tags is untouched. A caption without a tag list has just one variation.
"""

from __future__ import annotations

import hashlib
import random


_MAX_TAG_WORDS = 4
_MIN_TAGS = 3
MIN_TAGS_KEPT = 3


def _is_tag_list(text: str) -> bool:
    tags = [t.strip() for t in text.split(",")]
    return len(tags) >= _MIN_TAGS and all(t and len(t.split()) <= _MAX_TAG_WORDS for t in tags)


def split_tags(caption: str) -> tuple[list[str], str]:
    """``(tags, text)`` of a caption; no tags when it doesn't start with a tag list."""
    caption = caption.strip()
    head, separator, tail = caption.partition(". ")
    if separator and _is_tag_list(head):
        return [t.strip() for t in head.split(",")], tail.strip()
    if not separator and _is_tag_list(caption.rstrip(".")):
        return [t.strip() for t in caption.rstrip(".").split(",")], ""
    return [], caption


def build_variation(tags: list[str], text: str, rng: random.Random, *, keep_tags: int = 1,
                    tag_dropout: float = 0.0, min_tags_kept: int = MIN_TAGS_KEPT) -> str:
    head, tail = tags[:keep_tags], list(tags[keep_tags:])
    rng.shuffle(tail)
    drop = min(int(len(tail) * tag_dropout + 0.5), max(0, len(head) + len(tail) - min_tags_kept))
    if drop > 0:
        dropped = set(rng.sample(range(len(tail)), drop))
        tail = [tag for i, tag in enumerate(tail) if i not in dropped]
    tag_text = ", ".join(head + tail)
    return f"{tag_text}. {text}" if text else tag_text


def build_variations(caption: str, count: int, *, keep_tags: int = 1, tag_dropout: float = 0.0,
                     seed: int = 0) -> list[str]:
    """The distinct variations of a caption, the same on every run for the same inputs."""
    tags, text = split_tags(caption)
    if count <= 0 or not tags:
        return [caption]
    out: dict[str, None] = {}
    for slot in range(count):
        digest = hashlib.sha256(f"{seed}\0{keep_tags}\0{tag_dropout}\0{slot}\0{caption}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        out[build_variation(tags, text, rng, keep_tags=keep_tags, tag_dropout=tag_dropout)] = None
    return list(out)


def variation_slot(visit: int, count: int, key: str, seed: int = 0) -> int:
    """Which variation a visit takes: every cycle of ``count`` visits covers each one once, in a new order."""
    cycle, offset = divmod(visit, count)
    order = list(range(count))
    digest = hashlib.sha256(f"{seed}\0{cycle}\0{key}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order[offset]
