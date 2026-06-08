"""
GSM8K benchmark loader.

Loads grade-school math problems and constructs all 5 context component types.
Source: https://huggingface.co/datasets/openai/gsm8k
"""

import random
import re
from datasets import load_dataset
from contextshapley.assembler import ContextComponent, TaskInstance


SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "then give the final numerical answer on the last line in the format: "
    "#### <number>"
)

REFERENCE_DOC = (
    "## Arithmetic Reference\n"
    "- To find a total, add all quantities together.\n"
    "- To find a difference, subtract the smaller from the larger.\n"
    "- To find a rate, divide quantity by time.\n"
    "- To find a percentage, multiply by the decimal form (e.g., 20% = 0.20).\n"
    "- Always check your answer by working backwards."
)


def _extract_answer(solution_text: str) -> str:
    """Extract the numerical answer from GSM8K solution format (#### <number>)."""
    match = re.search(r"####\s*(.+)", solution_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return solution_text.strip()


def _format_examples(examples_data: list[dict]) -> str:
    """Format few-shot examples."""
    parts = []
    for ex in examples_data:
        answer = _extract_answer(ex["answer"])
        parts.append(f"Q: {ex['question']}\nA: The answer is #### {answer}")
    return "\n\n".join(parts)


def _format_history(question: str) -> str:
    """Simulate conversation history."""
    return (
        "User: Can you help me solve a math problem?\n"
        "Assistant: Of course! Please share the problem and I'll work through it step by step.\n"
        "User: Let me think about what information I have.\n"
        "Assistant: Take your time. Make sure to include all the numbers and details."
    )


def _format_tool_output(question: str, answer: str) -> str:
    """Simulate a calculator tool output."""
    return (
        f'{{"tool": "calculator", "input": "{question[:80]}...", '
        f'"status": "success", '
        f'"result": {{"computed_answer": {answer}, '
        f'"confidence": "high"}}}}'
    )


def load_gsm8k(
    n_instances: int = 50,
    n_examples: int = 3,
    seed: int = 42,
) -> list[TaskInstance]:
    """
    Load GSM8K instances with all 5 context component types.

    Args:
        n_instances: Number of task instances to load.
        n_examples: Number of few-shot examples per instance.
        seed: Random seed for reproducibility.

    Returns:
        List of TaskInstance objects.
    """
    rng = random.Random(seed)

    ds = load_dataset("openai/gsm8k", "main", split="test")

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    example_pool_indices = indices[:n_examples * 10]
    eval_indices = indices[n_examples * 10: n_examples * 10 + n_instances]

    example_pool = [ds[i] for i in example_pool_indices]

    tasks = []
    for i, idx in enumerate(eval_indices):
        row = ds[idx]
        gold_answer = _extract_answer(row["answer"])

        examples = rng.sample(example_pool, n_examples)

        components = {
            "S": ContextComponent(
                component_id="S",
                label="System Instructions",
                content=SYSTEM_PROMPT,
            ),
            "E": ContextComponent(
                component_id="E",
                label="Few-Shot Examples",
                content=_format_examples(examples),
            ),
            "D": ContextComponent(
                component_id="D",
                label="Retrieved Documents",
                content=REFERENCE_DOC,
            ),
            "H": ContextComponent(
                component_id="H",
                label="Conversation History",
                content=_format_history(row["question"]),
            ),
            "T": ContextComponent(
                component_id="T",
                label="Tool Outputs",
                content=_format_tool_output(row["question"], gold_answer),
            ),
        }

        task = TaskInstance(
            task_id=f"gsm8k_{i:04d}",
            query=row["question"],
            gold_answer=gold_answer,
            components=components,
            metadata={"full_solution": row["answer"]},
        )
        tasks.append(task)

    return tasks
