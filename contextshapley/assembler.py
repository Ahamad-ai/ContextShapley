"""
Context assembler: builds compound LLM contexts from component subsets.
"""

from dataclasses import dataclass, field


@dataclass
class ContextComponent:
    """A single context component with its type and content."""
    component_id: str  # One of: S, E, D, H, T
    label: str         # Human-readable name
    content: str       # The actual text content
    token_estimate: int = 0  # Approximate token count


@dataclass
class TaskInstance:
    """A single task instance with all available components and gold answer."""
    task_id: str
    query: str
    gold_answer: str
    components: dict[str, ContextComponent] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


COMPONENT_LABELS = {
    "S": "System Instructions",
    "E": "Few-Shot Examples",
    "D": "Retrieved Documents",
    "H": "Conversation History",
    "T": "Tool Outputs",
}

ALL_COMPONENT_IDS = ["S", "E", "D", "H", "T"]


class ContextAssembler:
    """
    Assembles a prompt from a subset of context components.

    The assembly order is fixed: S -> E -> D -> H -> T -> Query.
    This removes ordering as a confound.
    """

    ASSEMBLY_ORDER = ["S", "E", "D", "H", "T"]

    def assemble(
        self,
        task: TaskInstance,
        component_subset: frozenset[str],
    ) -> list[dict[str, str]]:
        """
        Assemble a chat-format prompt from the given component subset.

        Args:
            task: The task instance with all components available.
            component_subset: Which components to include (e.g., frozenset({"S", "D"})).

        Returns:
            List of message dicts for the OpenAI chat API format.
        """
        messages = []

        # System instructions go in the system message
        if "S" in component_subset and "S" in task.components:
            messages.append({
                "role": "system",
                "content": task.components["S"].content,
            })

        # Build the user message from remaining components
        user_parts = []

        if "E" in component_subset and "E" in task.components:
            user_parts.append(
                f"## Examples\n\n{task.components['E'].content}"
            )

        if "D" in component_subset and "D" in task.components:
            user_parts.append(
                f"## Reference Documents\n\n{task.components['D'].content}"
            )

        if "H" in component_subset and "H" in task.components:
            user_parts.append(
                f"## Conversation History\n\n{task.components['H'].content}"
            )

        if "T" in component_subset and "T" in task.components:
            user_parts.append(
                f"## Tool Outputs\n\n{task.components['T'].content}"
            )

        # Always append the query
        user_parts.append(f"## Question\n\n{task.query}")

        messages.append({
            "role": "user",
            "content": "\n\n".join(user_parts),
        })

        return messages

    def assemble_all_subsets(
        self,
        task: TaskInstance,
    ) -> dict[frozenset[str], list[dict[str, str]]]:
        """
        Assemble prompts for all 2^5=32 component subsets.

        Returns:
            Dict mapping frozenset of component IDs -> assembled messages.
        """
        from contextshapley.shapley import _powerset

        available = [
            c for c in self.ASSEMBLY_ORDER
            if c in task.components
        ]
        result = {}
        for subset in _powerset(available):
            result[subset] = self.assemble(task, subset)
        return result
