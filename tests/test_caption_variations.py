"""Caption variations: tag handling, stable output, visits, and the text-cache plan."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


TAGS = "@trigger, 1girl, solo, cat girl, white fur, long hair, red eyes, smile, choker, sparkle, signature"
TEXT = "A white cat girl, long hair falling, sits on a bench. Her hands rest on her knees."
CAPTION = f"{TAGS}. {TEXT}"


def test_split_tags_finds_the_tag_list_and_the_text() -> None:
    from training.caption_variations import split_tags

    tags, text = split_tags(CAPTION)
    assert tags[0] == "@trigger" and tags[-1] == "signature"
    assert text == TEXT


@pytest.mark.parametrize("caption", [
    "A white fox girl crouches low, one knee down and the other leg stretched out. Her hand rests on it.",
    "Just one short sentence",
    "",
])
def test_prose_has_no_tags(caption: str) -> None:
    from training.caption_variations import build_variations, split_tags

    assert split_tags(caption)[0] == []
    assert build_variations(caption, 5, tag_dropout=0.05) == [caption]


def test_tag_only_caption_is_recognised() -> None:
    from training.caption_variations import split_tags

    assert split_tags("a, b, c, d") == (["a", "b", "c", "d"], "")


def test_variations_keep_trigger_and_text_and_only_touch_tags() -> None:
    from training.caption_variations import build_variations, split_tags

    variations = build_variations(CAPTION, 5, keep_tags=1, tag_dropout=0.2, seed=42)
    original = set(split_tags(CAPTION)[0])
    assert len(variations) == 5
    for variation in variations:
        tags, text = split_tags(variation)
        assert variation.startswith("@trigger, ")
        assert text == TEXT
        assert set(tags) <= original
        assert len(tags) == len(original) - 2  # 20% of the 10 movable tags


def test_variations_are_stable_and_depend_on_the_seed() -> None:
    from training.caption_variations import build_variations

    assert build_variations(CAPTION, 5, seed=1) == build_variations(CAPTION, 5, seed=1)
    assert build_variations(CAPTION, 5, seed=1) != build_variations(CAPTION, 5, seed=2)


def test_dropout_leaves_a_minimum_of_tags() -> None:
    from training.caption_variations import build_variations, split_tags

    for variation in build_variations("a, b, c, d, e. Text here", 5, keep_tags=1, tag_dropout=1.0):
        assert len(split_tags(variation)[0]) == 3


def test_every_cycle_visits_each_slot_once_in_a_new_order() -> None:
    from training.caption_variations import variation_slot

    first = [variation_slot(v, 5, "image") for v in range(5)]
    second = [variation_slot(v, 5, "image") for v in range(5, 10)]
    assert sorted(first) == sorted(second) == list(range(5))
    assert any(variation_slot(v, 5, "image") != variation_slot(v, 5, "other") for v in range(5))


def _dataset(tmp_path: Path, **kwargs):
    pytest.importorskip("torch")
    from PIL import Image
    from runtime.training.dataset import BucketManager, ImageDataset

    folder = tmp_path / "1_data"
    folder.mkdir()
    Image.new("RGB", (64, 64)).save(folder / "a.png")
    (folder / "a.txt").write_text(CAPTION, encoding="utf-8")
    return ImageDataset(tmp_path, 1024, BucketManager(1024), prefer_json=False, **kwargs)


def test_dataset_walks_variations_on_each_visit(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, caption_variations=4, caption_keep_tags=1, caption_variation_tag_dropout=0.1)
    sample = ds.samples[0]
    variants = ds.caption_variants_for_sample(sample)
    assert len(variants) == 4
    assert {ds.caption_for_visit(sample) for _ in range(4)} == set(variants)
    assert ds.caption_for_sample(sample) == CAPTION


def test_dataset_without_variations_keeps_one_caption(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    assert ds.caption_variants_for_sample(ds.samples[0]) == [CAPTION]
    assert ds.caption_for_visit(ds.samples[0]) == CAPTION


def test_text_cache_plans_one_sidecar_per_variation(tmp_path: Path) -> None:
    from training.phases.text_cache import _collect_entries
    from training.text_cache import TextCacheEntry

    ds = _dataset(tmp_path, caption_variations=4, caption_variation_tag_dropout=0.1)
    ctx = SimpleNamespace(base_dataset=ds, reg_dataset=None, shared_cache=None)
    entries = _collect_entries(ctx)
    assert {e.caption for e in entries} == set(ds.caption_variants_for_sample(ds.samples[0]))
    assert len({e.cache_path for e in entries}) == 4
    assert all(e.cache_path.name.endswith(".text.safetensors") for e in entries)
    assert all(isinstance(e, TextCacheEntry) for e in entries)


def test_text_cache_plan_without_variations_is_the_old_sidecar(tmp_path: Path) -> None:
    from training.phases.text_cache import _collect_entries
    from training.text_cache import caption_sidecar_path

    ds = _dataset(tmp_path)
    entries = _collect_entries(SimpleNamespace(base_dataset=ds, reg_dataset=None, shared_cache=None))
    assert [e.cache_path for e in entries] == [caption_sidecar_path(ds.samples[0]["image"])]
