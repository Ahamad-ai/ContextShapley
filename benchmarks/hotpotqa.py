"""
HotpotQA benchmark loader.

Loads multi-hop QA instances and constructs all 5 context component types.
Source: https://huggingface.co/datasets/hotpot_qa (distractor split)
"""

import random
from datasets import load_dataset
from contextshapley.assembler import ContextComponent, TaskInstance


SYSTEM_PROMPT = (
    "You are a question-answering system. Answer the question using only "
    "the information provided. Give a short, direct answer. "
    "Do not explain your reasoning."
)


def _format_docs(supporting_facts_titles, context_titles, context_sentences):
    """Format supporting documents from HotpotQA context."""
    docs = []
    for title, sentences in zip(context_titles, context_sentences):
        text = " ".join(sentences)
        docs.append(f"[{title}] {text}")
    return "\n\n".join(docs)


def _format_examples(examples_data):
    """Format few-shot examples from other HotpotQA instances."""
    parts = []
    for ex in examples_data:
        parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    return "\n\n".join(parts)


def _format_history(question):
    """Simulate a conversation history leading to the question."""
    return (
        "User: I need help with a factual question.\n"
        "Assistant: Sure, I can help you with that. What would you like to know?\n"
        f"User: I'm researching something. Let me look up the details.\n"
        "Assistant: Take your time. I'll answer based on whatever information is available.\n"
        f"User: Ok here is my question about a specific topic."
    )


def _format_tool_output(question, answer):
    """Simulate tool outputs (search results) related to the question."""
    return (
        f'{{"tool": "web_search", "query": "{question}", '
        f'"status": "success", "results": ['
        f'{{"snippet": "Based on available sources, the answer involves: {answer}.", '
        f'"source": "encyclopedia"}}]}}'
    )


def load_hotpotqa(
    n_instances: int = 50,
    n_examples: int = 3,
    seed: int = 42,
) -> list[TaskInstance]:
    """
    Load HotpotQA instances with all 5 context component types.

    Args:
        n_instances: Number of task instances to load.
        n_examples: Number of few-shot examples per instance.
        seed: Random seed for reproducibility.

    Returns:
        List of TaskInstance objects ready for Shapley evaluation.
    """
    rng = random.Random(seed)

    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    # Shuffle and select
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    # Reserve first n_examples*2 for few-shot pool, rest for evaluation
    example_pool_indices = indices[:n_examples * 10]
    eval_indices = indices[n_examples * 10: n_examples * 10 + n_instances]

    example_pool = [ds[i] for i in example_pool_indices]

    tasks = []
    for idx in eval_indices:
        row = ds[idx]
        task_id = f"hotpotqa_{row['id']}"

        # Pick few-shot examples (different from this instance)
        examples = rng.sample(example_pool, n_examples)

        # Build all 5 components
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
                content=_format_docs(
                    row["supporting_facts"]["title"],
                    row["context"]["title"],
                    row["context"]["sentences"],
                ),
            ),
            "H": ContextComponent(
                component_id="H",
                label="Conversation History",
                content=_format_history(row["question"]),
            ),
            "T": ContextComponent(
                component_id="T",
                label="Tool Outputs",
                content=_format_tool_output(row["question"], row["answer"]),
            ),
        }

        task = TaskInstance(
            task_id=task_id,
            query=row["question"],
            gold_answer=row["answer"],
            components=components,
            metadata={"type": row["type"], "level": row["level"]},
        )
        tasks.append(task)

    return tasks
