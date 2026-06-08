"""
IFEval-style benchmark loader.

Tests instruction-following with verifiable constraints.
Uses hand-crafted instances with measurable instruction adherence.
(The official IFEval dataset requires complex evaluation logic;
we create a clean subset with binary-verifiable constraints.)
"""

import random
from contextshapley.assembler import ContextComponent, TaskInstance


IFEVAL_INSTANCES = [
    {
        "id": "if_001",
        "query": "List three benefits of exercise.",
        "system": "Always respond in exactly 3 bullet points. Start each bullet with a dash (-).",
        "check_fn": "bullet_count_3",
        "gold": "3 bullet points starting with -",
    },
    {
        "id": "if_002",
        "query": "Explain what photosynthesis is.",
        "system": "Respond in exactly two sentences. Do not use the word 'sunlight'.",
        "check_fn": "two_sentences_no_sunlight",
        "gold": "2 sentences without 'sunlight'",
    },
    {
        "id": "if_003",
        "query": "What are the primary colors?",
        "system": "Respond in valid JSON format with a key called 'answer'.",
        "check_fn": "valid_json_with_answer",
        "gold": "valid JSON with 'answer' key",
    },
    {
        "id": "if_004",
        "query": "Describe the water cycle.",
        "system": "Use exactly 5 sentences. Number each sentence (1. 2. 3. 4. 5.).",
        "check_fn": "five_numbered_sentences",
        "gold": "5 numbered sentences",
    },
    {
        "id": "if_005",
        "query": "What is machine learning?",
        "system": "Respond in all uppercase letters. Keep your response under 50 words.",
        "check_fn": "all_uppercase_under_50",
        "gold": "ALL CAPS under 50 words",
    },
    {
        "id": "if_006",
        "query": "Name five countries in Europe.",
        "system": "Respond with a comma-separated list. Do not use periods.",
        "check_fn": "comma_separated_no_periods",
        "gold": "comma-separated, no periods",
    },
    {
        "id": "if_007",
        "query": "What causes earthquakes?",
        "system": "Start your response with 'ANSWER:' and keep it to one paragraph.",
        "check_fn": "starts_with_answer",
        "gold": "starts with ANSWER:",
    },
    {
        "id": "if_008",
        "query": "Explain gravity to a child.",
        "system": "Use only words with 3 or fewer syllables. Keep response under 40 words.",
        "check_fn": "under_40_words",
        "gold": "simple words, under 40 words",
    },
    {
        "id": "if_009",
        "query": "What is the capital of Japan?",
        "system": "Respond with exactly one word. Nothing else.",
        "check_fn": "exactly_one_word",
        "gold": "Tokyo",
    },
    {
        "id": "if_010",
        "query": "List the planets in our solar system.",
        "system": "Use a numbered list (1. 2. 3. ...). Include exactly 8 items.",
        "check_fn": "numbered_list_8",
        "gold": "numbered list with 8 planets",
    },
    {
        "id": "if_011",
        "query": "What is Python used for?",
        "system": "End your response with the exact phrase 'Is there anything else you need?'",
        "check_fn": "ends_with_phrase",
        "gold": "ends with 'Is there anything else you need?'",
    },
    {
        "id": "if_012",
        "query": "Describe the color blue.",
        "system": "Respond in exactly 3 sentences. Each sentence must contain the word 'blue'.",
        "check_fn": "three_sentences_with_blue",
        "gold": "3 sentences each containing 'blue'",
    },
    {
        "id": "if_013",
        "query": "What is 15 + 27?",
        "system": "Respond with only the number. No words, no punctuation, just the number.",
        "check_fn": "only_number",
        "gold": "42",
    },
    {
        "id": "if_014",
        "query": "Explain what a database is.",
        "system": "Use the word 'data' at least 3 times in your response. Keep it under 60 words.",
        "check_fn": "data_three_times_under_60",
        "gold": "'data' at least 3 times, under 60 words",
    },
    {
        "id": "if_015",
        "query": "Why is the sky blue?",
        "system": "Respond in Q&A format: restate the question, then answer. Use exactly 2 sentences for the answer.",
        "check_fn": "qa_format_two_sentences",
        "gold": "Q&A format with 2 sentence answer",
    },
]


def _check_response(check_fn: str, response: str) -> float:
    """
    Evaluate response against instruction constraint.
    Returns 1.0 if constraint is met, 0.0 otherwise.
    """
    response = response.strip()

    if check_fn == "bullet_count_3":
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        bullet_lines = [l for l in lines if l.startswith("-")]
        return 1.0 if len(bullet_lines) == 3 else 0.0

    elif check_fn == "two_sentences_no_sunlight":
        has_sunlight = "sunlight" in response.lower()
        # Rough sentence count
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return 1.0 if len(sentences) == 2 and not has_sunlight else 0.0

    elif check_fn == "valid_json_with_answer":
        import json
        try:
            parsed = json.loads(response)
            return 1.0 if "answer" in parsed else 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0

    elif check_fn == "five_numbered_sentences":
        for i in range(1, 6):
            if f"{i}." not in response:
                return 0.0
        return 1.0

    elif check_fn == "all_uppercase_under_50":
        words = response.split()
        # Check uppercase (ignoring punctuation/numbers)
        alpha_chars = [c for c in response if c.isalpha()]
        is_upper = all(c.isupper() for c in alpha_chars) if alpha_chars else False
        return 1.0 if is_upper and len(words) <= 50 else 0.0

    elif check_fn == "comma_separated_no_periods":
        return 1.0 if "," in response and "." not in response else 0.0

    elif check_fn == "starts_with_answer":
        return 1.0 if response.startswith("ANSWER:") else 0.0

    elif check_fn == "under_40_words":
        return 1.0 if len(response.split()) <= 40 else 0.0

    elif check_fn == "exactly_one_word":
        return 1.0 if len(response.split()) == 1 else 0.0

    elif check_fn == "numbered_list_8":
        has_8 = "8." in response
        has_9 = "9." in response
        return 1.0 if has_8 and not has_9 else 0.0

    elif check_fn == "ends_with_phrase":
        return 1.0 if response.rstrip().endswith("Is there anything else you need?") else 0.0

    elif check_fn == "three_sentences_with_blue":
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) != 3:
            return 0.0
        return 1.0 if all("blue" in s.lower() for s in sentences) else 0.0

    elif check_fn == "only_number":
        return 1.0 if response.strip().isdigit() else 0.0

    elif check_fn == "data_three_times_under_60":
        word_count = len(response.split())
        data_count = response.lower().count("data")
        return 1.0 if data_count >= 3 and word_count <= 60 else 0.0

    elif check_fn == "qa_format_two_sentences":
        return 1.0 if "?" in response else 0.0

    return 0.0


EXAMPLE_POOL = [
    "Q: Name three fruits.\nSystem: Respond in bullet points starting with -.\nA:\n- Apple\n- Banana\n- Orange",
    "Q: What is 2+2?\nSystem: Respond with only the number.\nA: 4",
    "Q: Define AI.\nSystem: Use exactly 2 sentences.\nA: AI stands for artificial intelligence. It refers to computer systems that can perform tasks typically requiring human intelligence.",
]

REFERENCE_DOC = (
    "## Instruction Following Guide\n"
    "- Read the system instructions carefully before answering.\n"
    "- Pay attention to format requirements (JSON, bullets, numbering).\n"
    "- Count words/sentences if the instruction specifies a count.\n"
    "- Check constraints before submitting (forbidden words, required phrases)."
)


def load_ifeval(
    n_instances: int | None = None,
    seed: int = 42,
) -> list[TaskInstance]:
    """
    Load IFEval-style instances with verifiable instruction constraints.

    Args:
        n_instances: Number of instances (max 15). None = all.
        seed: Random seed.

    Returns:
        List of TaskInstance objects.
    """
    rng = random.Random(seed)

    instances = IFEVAL_INSTANCES[:]
    if n_instances and n_instances < len(instances):
        instances = rng.sample(instances, n_instances)

    tasks = []
    for inst in instances:
        history = (
            "User: I want to test your ability to follow instructions.\n"
            "Assistant: Sure! Give me a specific format or constraint and I'll do my best.\n"
            "User: Great, here comes the test."
        )

        tool_output = (
            '{"tool": "instruction_validator", '
            '"status": "ready", '
            '"message": "Will validate response against format constraints."}'
        )

        components = {
            "S": ContextComponent(
                component_id="S",
                label="System Instructions",
                content=inst["system"],
            ),
            "E": ContextComponent(
                component_id="E",
                label="Few-Shot Examples",
                content="\n\n".join(rng.sample(EXAMPLE_POOL, min(2, len(EXAMPLE_POOL)))),
            ),
            "D": ContextComponent(
                component_id="D",
                label="Retrieved Documents",
                content=REFERENCE_DOC,
            ),
            "H": ContextComponent(
                component_id="H",
                label="Conversation History",
                content=history,
            ),
            "T": ContextComponent(
                component_id="T",
                label="Tool Outputs",
                content=tool_output,
            ),
        }

        task = TaskInstance(
            task_id=inst["id"],
            query=inst["query"],
            gold_answer=inst["gold"],
            components=components,
            metadata={"check_fn": inst["check_fn"]},
        )
        tasks.append(task)

    return tasks


def evaluate_ifeval(response: str, task: TaskInstance) -> float:
    """Evaluate a response against the task's instruction constraint."""
    check_fn = task.metadata.get("check_fn", "")
    return _check_response(check_fn, response)
