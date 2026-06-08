"""Tests for evaluation metrics."""

import pytest
from math import isclose
from contextshapley.evaluators import (
    normalize_answer,
    f1_score,
    exact_match,
    contains_answer,
    gsm8k_evaluate,
)


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("  The Answer  ") == "answer"

    def test_articles(self):
        assert normalize_answer("a cat and an elephant") == "cat and elephant"

    def test_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"


class TestF1Score:
    def test_perfect_match(self):
        assert isclose(f1_score("Paris", "Paris"), 1.0)

    def test_no_match(self):
        assert isclose(f1_score("London", "Paris"), 0.0)

    def test_partial_match(self):
        score = f1_score("The capital is Paris France", "Paris")
        assert 0.0 < score < 1.0

    def test_empty_gold(self):
        assert isclose(f1_score("", ""), 1.0)

    def test_empty_prediction(self):
        assert isclose(f1_score("", "Paris"), 0.0)


class TestExactMatch:
    def test_match(self):
        assert exact_match("Paris", "paris") == 1.0

    def test_no_match(self):
        assert exact_match("London", "Paris") == 0.0

    def test_with_articles(self):
        assert exact_match("The answer is Paris", "answer is Paris") == 1.0


class TestContainsAnswer:
    def test_contains(self):
        assert contains_answer("The capital of France is Paris.", "Paris") == 1.0

    def test_not_contains(self):
        assert contains_answer("The capital is London.", "Paris") == 0.0


class TestGSM8KEvaluate:
    def test_with_hash_format(self):
        assert gsm8k_evaluate("Step 1... Step 2... #### 42", "42") == 1.0

    def test_wrong_answer(self):
        assert gsm8k_evaluate("#### 43", "42") == 0.0

    def test_number_in_text(self):
        assert gsm8k_evaluate("The answer is 42.", "42") == 1.0

    def test_with_commas(self):
        assert gsm8k_evaluate("#### 1,000", "1000") == 1.0

    def test_no_number(self):
        assert gsm8k_evaluate("I don't know", "42") == 0.0

    def test_float_answer(self):
        assert gsm8k_evaluate("#### 3.5", "3.5") == 1.0
