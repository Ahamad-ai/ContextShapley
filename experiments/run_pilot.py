"""
Pilot experiment: validates the full pipeline end-to-end.

Runs 2 IFEval instances x 32 subsets = 64 API calls using direct OpenAI API.
Computes Shapley values and interaction indices, prints a summary.
"""

import json
import sys
import argparse
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contextshapley.assembler import ContextAssembler, ALL_COMPONENT_IDS
from contextshapley.shapley import (
    all_shapley_values,
    interaction_matrix,
    verify_efficiency_axiom,
    _powerset,
)
from contextshapley.models.openai_wrapper import OpenAIModel
from benchmarks.ifeval import load_ifeval, evaluate_ifeval


def run_pilot(n_instances: int = 2, model_name: str = "gpt-5"):
    """Run the pilot experiment."""

    print("=" * 60)
    print("ContextShapley Pilot Experiment")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Instances: {n_instances}")
    print(f"Subsets per instance: 32")
    print(f"Total API calls: {n_instances * 32}")
    print()

    # Setup
    results_dir = PROJECT_ROOT / "results" / "raw" / "pilot"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = OpenAIModel(
        model_name=model_name,
        use_model_loader=False,
        log_dir=str(results_dir / "logs"),
    )
    assembler = ContextAssembler()
    tasks = load_ifeval(n_instances=n_instances, seed=42)

    print(f"[loader] Loaded {len(tasks)} IFEval tasks")
    for t in tasks:
        print(f"  - {t.task_id}: {t.query[:60]}...")
    print()

    # Run all subsets for each task
    all_results = []

    for task_idx, task in enumerate(tasks):
        print(f"[task {task_idx+1}/{len(tasks)}] {task.task_id}")
        task_scores = {}  # frozenset -> score
        task_responses = {}  # frozenset -> response text

        subsets = list(_powerset(ALL_COMPONENT_IDS))
        total_subsets = len(subsets)

        for sub_idx, subset in enumerate(subsets):
            subset_label = ",".join(sorted(subset)) if subset else "EMPTY"
            call_id = f"{task.task_id}__{subset_label}"

            # Assemble context
            messages = assembler.assemble(task, subset)

            # Call LLM
            result = model.generate(
                messages=messages,
                max_tokens=2048,
                call_id=call_id,
            )

            # Evaluate
            score = evaluate_ifeval(result.response_text, task)
            task_scores[subset] = score
            task_responses[subset] = result.response_text

            status = "PASS" if score == 1.0 else "FAIL"
            print(
                f"  [{sub_idx+1:2d}/{total_subsets}] "
                f"{{{subset_label:20s}}} -> {status} "
                f"({result.prompt_tokens}+{result.completion_tokens} tokens)"
            )

        # Compute Shapley values for this task
        sv = all_shapley_values(task_scores, ALL_COMPONENT_IDS)
        im = interaction_matrix(task_scores, ALL_COMPONENT_IDS)
        eff_ok, sv_sum, expected = verify_efficiency_axiom(task_scores, ALL_COMPONENT_IDS)

        print(f"\n  --- Shapley Values for {task.task_id} ---")
        for c in ALL_COMPONENT_IDS:
            bar = "+" * int(abs(sv[c]) * 20) if sv[c] >= 0 else "-" * int(abs(sv[c]) * 20)
            sign = "+" if sv[c] >= 0 else ""
            print(f"  {c} ({task.components[c].label:22s}): {sign}{sv[c]:.4f}  {bar}")

        print(f"\n  --- Top Interaction Effects ---")
        pairs_sorted = sorted(
            [(k, v) for k, v in im.items() if k[0] < k[1]],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for (ci, cj), val in pairs_sorted[:5]:
            label = "synergy" if val > 0 else "interference" if val < 0 else "neutral"
            print(f"  {ci}-{cj}: {val:+.4f} ({label})")

        print(f"\n  Efficiency axiom: {'PASS' if eff_ok else 'FAIL'} "
              f"(sum={sv_sum:.4f}, expected={expected:.4f})")

        # Save results
        task_result = {
            "task_id": task.task_id,
            "query": task.query,
            "gold_answer": task.gold_answer,
            "check_fn": task.metadata.get("check_fn"),
            "scores": {
                ",".join(sorted(k)) if k else "EMPTY": v
                for k, v in task_scores.items()
            },
            "shapley_values": sv,
            "interactions": {
                f"{ci},{cj}": val for (ci, cj), val in im.items() if ci < cj
            },
            "efficiency_axiom_passes": eff_ok,
            "responses": {
                ",".join(sorted(k)) if k else "EMPTY": v
                for k, v in task_responses.items()
            },
        }
        all_results.append(task_result)
        print()

    # Save all results
    output_path = results_dir / "pilot_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved] Results -> {output_path}")

    # Print cost summary
    cost = model.get_cost_summary()
    print(f"\n{'='*60}")
    print("Cost Summary:")
    print(f"  Total calls: {cost['total_calls']}")
    print(f"  Prompt tokens: {cost['total_prompt_tokens']:,}")
    print(f"  Completion tokens: {cost['total_completion_tokens']:,}")
    print(f"  Total tokens: {cost['total_tokens']:,}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextShapley Pilot Experiment")
    parser.add_argument("--n-instances", type=int, default=2)
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args()
    run_pilot(n_instances=args.n_instances, model_name=args.model)
