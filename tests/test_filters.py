"""Metadata filters: the expression grammar and the missing-field rules."""

from __future__ import annotations

import pytest
from app.core.filters import Filter, apply_filters, matches_all, parse_filters

META = {"doc_id": "a1", "year": 2021, "category": "retrieval", "tags": ["rag", "hybrid"]}


def test_dict_equality():
    filters = parse_filters({"doc_id": "a1", "year": 2021})
    assert matches_all(filters, META)
    assert not matches_all(parse_filters({"doc_id": "a2"}), META)


def test_dict_list_becomes_membership():
    filters = parse_filters({"category": ["retrieval", "vision"]})
    assert filters[0].op == "in"
    assert matches_all(filters, META)


@pytest.mark.parametrize(
    "clause,expected",
    [
        ("year >= 2020", True),
        ("year > 2021", False),
        ("year <= 2021", True),
        ("year < 2020", False),
        ("category = retrieval", True),
        ("category: retrieval", True),
        ("category != vision", True),
        ("title contains attention", False),
        ("category in [retrieval, vision]", True),
        ("category nin [vision]", True),
        ("category in [vision]", False),
        ("tags contains rag", True),
    ],
)
def test_expression_grammar(clause: str, expected: bool):
    assert matches_all(parse_filters([clause]), META) is expected


def test_string_comparison_is_case_insensitive():
    assert matches_all(parse_filters(["category = RETRIEVAL"]), META)


def test_quoted_values_keep_spaces():
    filters = parse_filters(['title = "attention is all you need"'])
    assert filters[0].value == "attention is all you need"
    assert not matches_all(filters, META)


def test_missing_field_semantics():
    # "not equal" is true when the field is absent; everything else is false.
    assert matches_all(parse_filters(["venue != neurips"]), META)
    assert not matches_all(parse_filters(["venue = neurips"]), META)
    assert not matches_all(parse_filters(["venue > 2020"]), META)
    assert not matches_all(parse_filters(["venue in [neurips]"]), META)


def test_combined_clauses_are_anded():
    assert matches_all(parse_filters(["year >= 2020", "category = retrieval"]), META)
    assert not matches_all(parse_filters(["year >= 2020", "category = vision"]), META)


def test_empty_spec_matches_everything():
    assert parse_filters(None) == []
    assert parse_filters([]) == []
    assert matches_all([], META)


def test_unparseable_clause_raises():
    with pytest.raises(ValueError, match="cannot parse filter"):
        parse_filters(["year roughly 2020"])


def test_unsupported_spec_type_raises():
    with pytest.raises(TypeError):
        parse_filters(42)


def test_apply_filters_returns_matching_indices():
    metas = [META, {"doc_id": "b2", "year": 2019, "category": "vision"}, {"doc_id": "c3", "year": 2023}]
    assert apply_filters(parse_filters(["year >= 2020"]), metas) == [0, 2]
    assert apply_filters([], metas) == []


def test_filter_dataclass_is_hashable():
    assert {Filter("a", "eq", 1), Filter("a", "eq", 1)} == {Filter("a", "eq", 1)}


def test_numeric_strings_are_coerced():
    filters = parse_filters(["year >= 2020"])
    assert filters[0].value == 2020
    assert matches_all(parse_filters(["year >= 2019.5"]), META)
