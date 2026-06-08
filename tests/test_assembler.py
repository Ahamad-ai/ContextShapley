"""
Tests for the ContextAssembler.

Verifies:
1. Correct message format for OpenAI chat API
2. Correct inclusion/exclusion of components
3. Fixed assembly order (S -> E -> D -> H -> T -> Query)
4. All 32 subsets generate valid messages
"""

import pytest
from contextshapley.assembler import (
    ContextAssembler,
    ContextComponent,
    TaskInstance,
    ALL_COMPONENT_IDS,
)


@pytest.fixture
def sample_task():
    """A fully populated task instance with all 5 components."""
    return TaskInstance(
        task_id="test_001",
        query="What is the capital of France?",
        gold_answer="Paris",
        components={
            "S": ContextComponent("S", "System Instructions", "You are a geography expert. Answer concisely."),
            "E": ContextComponent("E", "Few-Shot Examples", "Q: Capital of Germany?\nA: Berlin"),
            "D": ContextComponent("D", "Retrieved Documents", "France is a country in Western Europe. Its capital is Paris."),
            "H": ContextComponent("H", "Conversation History", "User: Tell me about Europe.\nAssistant: Europe has 44 countries."),
            "T": ContextComponent("T", "Tool Outputs", '{"search_result": "Paris is the capital of France"}'),
        },
    )


@pytest.fixture
def assembler():
    return ContextAssembler()


class TestAssemblerBasic:
    def test_empty_subset(self, assembler, sample_task):
        """Empty subset should produce only the query in a user message."""
        messages = assembler.assemble(sample_task, frozenset())
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What is the capital of France?" in messages[0]["content"]

    def test_system_only(self, assembler, sample_task):
        """System-only should produce system + user(query) messages."""
        messages = assembler.assemble(sample_task, frozenset({"S"}))
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "geography expert" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_full_subset(self, assembler, sample_task):
        """All components should produce system + user message with all sections."""
        messages = assembler.assemble(sample_task, frozenset(ALL_COMPONENT_IDS))
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        user_content = messages[1]["content"]
        assert "## Examples" in user_content
        assert "## Reference Documents" in user_content
        assert "## Conversation History" in user_content
        assert "## Tool Outputs" in user_content
        assert "## Question" in user_content

    def test_docs_only(self, assembler, sample_task):
        """D only: no system message, user message has docs + query."""
        messages = assembler.assemble(sample_task, frozenset({"D"}))
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "## Reference Documents" in messages[0]["content"]
        assert "France" in messages[0]["content"]

    def test_query_always_present(self, assembler, sample_task):
        """Query should always be in the user message regardless of subset."""
        for subset_size in range(6):
            from itertools import combinations
            for combo in combinations(ALL_COMPONENT_IDS, subset_size):
                messages = assembler.assemble(sample_task, frozenset(combo))
                user_msg = messages[-1]
                assert user_msg["role"] == "user"
                assert "What is the capital of France?" in user_msg["content"]


class TestAssemblerOrder:
    def test_component_order_in_user_message(self, assembler, sample_task):
        """Components should appear in order: E, D, H, T, Query."""
        messages = assembler.assemble(
            sample_task, frozenset({"E", "D", "H", "T"})
        )
        user_content = messages[0]["content"]  # No system msg without S
        e_pos = user_content.index("## Examples")
        d_pos = user_content.index("## Reference Documents")
        h_pos = user_content.index("## Conversation History")
        t_pos = user_content.index("## Tool Outputs")
        q_pos = user_content.index("## Question")
        assert e_pos < d_pos < h_pos < t_pos < q_pos


class TestAssemblerAllSubsets:
    def test_generates_32_subsets(self, assembler, sample_task):
        """Should generate exactly 32 subsets for 5 components."""
        all_prompts = assembler.assemble_all_subsets(sample_task)
        assert len(all_prompts) == 32

    def test_all_subsets_have_valid_messages(self, assembler, sample_task):
        """Every subset should produce at least one message with the query."""
        all_prompts = assembler.assemble_all_subsets(sample_task)
        for subset, messages in all_prompts.items():
            assert len(messages) >= 1
            # Last message should be user role with query
            assert messages[-1]["role"] == "user"
            assert "What is the capital of France?" in messages[-1]["content"]

    def test_empty_subset_in_all(self, assembler, sample_task):
        """Empty subset should be included."""
        all_prompts = assembler.assemble_all_subsets(sample_task)
        assert frozenset() in all_prompts

    def test_full_subset_in_all(self, assembler, sample_task):
        """Full subset should be included."""
        all_prompts = assembler.assemble_all_subsets(sample_task)
        assert frozenset(ALL_COMPONENT_IDS) in all_prompts


class TestAssemblerMissingComponents:
    def test_task_with_partial_components(self, assembler):
        """Task with only S and D should work, other subsets gracefully skip."""
        task = TaskInstance(
            task_id="partial_001",
            query="What is 2+2?",
            gold_answer="4",
            components={
                "S": ContextComponent("S", "System", "You are a calculator."),
                "D": ContextComponent("D", "Docs", "Basic arithmetic reference."),
            },
        )
        # Requesting E which doesn't exist should not crash
        messages = assembler.assemble(task, frozenset({"S", "E", "D"}))
        assert len(messages) == 2  # system + user
        assert "## Examples" not in messages[1]["content"]
        assert "## Reference Documents" in messages[1]["content"]
