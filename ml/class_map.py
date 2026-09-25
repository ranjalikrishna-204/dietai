"""Loads data/knowledge/class_map.json and gives a normalise() helper used by every dataset script."""
import json
import re
from pathlib import Path

_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "class_map.json"


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def load_alias_table() -> dict[str, str]:
    """Return {alias_slug: canonical_class}."""
    raw = json.loads(_MAP_PATH.read_text())
    table = {}
    for canonical, aliases in raw.items():
        if canonical.startswith("_"):
            continue
        table[_slug(canonical)] = canonical
        for alias in aliases:
            table[_slug(alias)] = canonical
    return table


def normalise(label: str, alias_table: dict[str, str] | None = None) -> str | None:
    """Map a raw dataset label to our canonical class name, or None if unknown."""
    alias_table = alias_table or load_alias_table()
    return alias_table.get(_slug(label))


def canonical_classes() -> list[str]:
    raw = json.loads(_MAP_PATH.read_text())
    return sorted(k for k in raw if not k.startswith("_"))
