"""Tests for benchmark loaders (offline tests that don't require API calls)."""

import pytest
from contextshapley.assembler import ALL_COMPONENT_IDS
from benchmarks.ifeval import load_ifeval, evaluate_ifeval, _check_response


class TestIFEvalLoader:
    def test_loads_all_instances(self):
        tasks = load_ifeval()
        assert len(tasks) == 15

    def test_loads_subset(self):
        tasks = load_ifeval(n_instances=5)
        assert len(tasks) == 5

    def test_all_components_present(self):
        tasks = load_ifeval()
        for task in tasks:
            for cid in ALL_COMPONENT_IDS:
                assert cid in task.components, f"Missing {cid} in {task.task_id}"

    def test_check_fn_in_metadata(self):
        tasks = load_ifeval()
        for task in tasks:
            assert "check_fn" in task.metadata

    def test_deterministic(self):
        """Same seed should produce same order."""
        t1 = load_ifeval(n_instances=5, seed=42)
        t2 = load_ifeval(n_instances=5, seed=42)
        assert [t.task_id for t in t1] == [t.task_id for t in t2]


class TestIFEvalCheckers:
    def test_bullet_count_3_pass(self):
        assert _check_response("bullet_count_3", "- one\n- two\n- three") == 1.0

    def test_bullet_count_3_fail(self):
        assert _check_response("bullet_count_3", "- one\n- two") == 0.0

    def test_valid_json_pass(self):
        assert _check_response("valid_json_with_answer", '{"answer": "red, blue, yellow"}') == 1.0

    def test_valid_json_fail_no_key(self):
        assert _check_response("valid_json_with_answer", '{"result": "test"}') == 0.0

    def test_valid_json_fail_invalid(self):
        assert _check_response("valid_json_with_answer", "not json at all") == 0.0

    def test_all_uppercase_pass(self):
        assert _check_response("all_uppercase_under_50", "THIS IS ALL UPPERCASE") == 1.0

    def test_all_uppercase_fail(self):
        assert _check_response("all_uppercase_under_50", "This is Mixed Case") == 0.0

    def test_exactly_one_word_pass(self):
        assert _check_response("exactly_one_word", "Tokyo") == 1.0

    def test_exactly_one_word_fail(self):
        assert _check_response("exactly_one_word", "Tokyo Japan") == 0.0

    def test_only_number_pass(self):
        assert _check_response("only_number", "42") == 1.0

    def test_only_number_fail(self):
        assert _check_response("only_number", "The answer is 42") == 0.0

    def test_starts_with_answer_pass(self):
        assert _check_response("starts_with_answer", "ANSWER: tectonic plates") == 1.0

    def test_starts_with_answer_fail(self):
        assert _check_response("starts_with_answer", "Earthquakes are caused by...") == 0.0

    def test_ends_with_phrase_pass(self):
        resp = "Python is used for web dev. Is there anything else you need?"
        assert _check_response("ends_with_phrase", resp) == 1.0

    def test_comma_separated_no_periods_pass(self):
        assert _check_response("comma_separated_no_periods", "France, Germany, Italy, Spain, Poland") == 1.0

    def test_comma_separated_no_periods_fail(self):
        assert _check_response("comma_separated_no_periods", "France. Germany. Italy.") == 0.0
