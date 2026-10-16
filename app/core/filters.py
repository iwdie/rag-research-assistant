"""Metadata filtering for document-scoped retrieval.

Two ways to build a filter set, both ending up as the same ``Filter`` objects:

``{"doc_id": "a1b2", "year": 2023}``
    Shorthand for equality; a list value means "any of".

``["year >= 2020", "category in [nlp, vision]", "title contains transformer"]``
    A small expression grammar, one clause per string.

Why not push filters into FAISS? ``IndexFlatIP`` has no native filtered search,
and its ``IDSelector`` path still scores the whole index. Filtering *after*
scoring means asking for ``candidate_k`` results and hoping enough survive the
filter - which silently returns fewer results than requested on a narrow filter.
Instead the retriever hands the allowed row set to the stores, which score only
those rows. Correct beats clever here, and brute force over a few thousand
chunks is microseconds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_EXPR_RE = re.compile(
    r"^\s*(?P<field>[A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?P<op>==|!=|>=|<=|>|<|=|:|~=|in|nin|contains|exists)\s*"
    r"(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
_ALIASES = {
    "==": "eq",
    "=": "eq",
    ":": "eq",
    "!=": "ne",
    "~=": "contains",
    ">=": "gte",
    "<=": "lte",
    ">": "gt",
    "<": "lt",
}
_RANGE_OPS = {"gt", "gte", "lt", "lte"}
_NEGATIVE_OPS = {"ne", "nin"}
_MISSING = object()


def _coerce(raw: str) -> Any:
    """Interpret a literal from the expression grammar."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        return [_coerce(part) for part in text[1:-1].split(",") if part.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@dataclass(frozen=True)
class Filter:
    """A single predicate over a chunk's metadata (and its top-level fields)."""

    field: str
    op: str
    value: Any

    def matches(self, meta: Mapping[str, Any]) -> bool:
        actual = meta.get(self.field, _MISSING)
        if actual is _MISSING:
            # "field != x" is true for a document that has no such field - that
            # is what "not equal" means. Every other operator needs a value.
            return self.op in _NEGATIVE_OPS
        if self.op == "exists":
            return bool(self.value) and actual is not _MISSING
        if self.op in _NEGATIVE_OPS:
            if self.op == "nin":
                return actual not in _as_list(self.value)
            return not _equals(actual, self.value)
        if self.op in _RANGE_OPS:
            return _compare(actual, self.value, self.op)
        if isinstance(actual, list | tuple | set):
            # Filtering a list field matches if any element matches.
            if self.op == "contains":
                return any(_contains(item, self.value) for item in actual)
            if self.op == "in":
                return any(_equals(item, v) for item in actual for v in _as_list(self.value))
            return any(_equals(item, self.value) for item in actual)
        if self.op == "in":
            return any(_equals(actual, v) for v in _as_list(self.value))
        if self.op == "contains":
            return _contains(actual, self.value)
        return _equals(actual, self.value)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _equals(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return str(expected).casefold() in actual.casefold()
    if isinstance(actual, list | tuple | set):
        return any(_equals(item, expected) for item in actual)
    return False


def _compare(actual: Any, expected: Any, op: str) -> bool:
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        # Fall back to string ordering (works for ISO dates).
        left, right = str(actual), str(expected)  # type: ignore[assignment]
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    return left <= right


def parse_filters(spec: Any) -> list[Filter]:
    """Normalise a dict or list-of-expressions into ``Filter`` objects."""
    if spec is None:
        return []
    if isinstance(spec, Filter):
        return [spec]
    if isinstance(spec, Mapping):
        filters: list[Filter] = []
        for field, value in spec.items():
            op = "in" if isinstance(value, list | tuple | set) and not isinstance(value, str) else "eq"
            filters.append(Filter(field=field, op=op, value=value))
        return filters
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, Iterable):
        filters = []
        for clause in spec:
            if isinstance(clause, Mapping):
                filters.extend(parse_filters(clause))
                continue
            match = _EXPR_RE.match(str(clause))
            if not match:
                raise ValueError(
                    f"cannot parse filter {clause!r}; expected 'field op value' "
                    "with op in ==, !=, >, >=, <, <=, :, in, nin, contains, exists"
                )
            raw_op = match.group("op").lower()
            op = _ALIASES.get(raw_op, raw_op)
            filters.append(Filter(field=match.group("field"), op=op, value=_coerce(match.group("value"))))
        return filters
    raise TypeError(f"unsupported filter specification: {type(spec).__name__}")


def matches_all(filters: list[Filter], meta: Mapping[str, Any]) -> bool:
    """True when every filter matches. An empty filter set matches everything."""
    return all(f.matches(meta) for f in filters)


def apply_filters(
    filters: list[Filter],
    metas: Iterable[Mapping[str, Any]],
) -> list[int]:
    """Return the indices of ``metas`` that satisfy ``filters``."""
    if not filters:
        return []
    return [index for index, meta in enumerate(metas) if matches_all(filters, meta)]
