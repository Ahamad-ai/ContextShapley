"""
Full ContextShapley experiment runner.

Runs all benchmarks (IFEval, HotpotQA, GSM8K) with all 32 subsets per instance.
Supports resume from checkpoint if interrupted.

Usage:
    python3 experiments/run_full.py
    python3 experiments/run_full.py --benchmark hotpotqa --n-instances 10
    python3 experiments/run_full.py --resume  # resume from last checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contextshapley.assembler import ContextAssembler, ALL_COMPONENT_IDS
from contextshapley.shapley import (
    all_shapley_values,
    interaction_matrix,
    verify_efficiency_axiom,
    _powerset,
)
from contextshapley.evaluators import f1_score, exact_match, gsm8k_evaluate
from contextshapley.models.openai_wrapper import OpenAIModel, BedrockModel, OllamaModel
from benchmarks.ifeval import load_ifeval, evaluate_ifeval
from benchmarks.hotpotqa import load_hotpotqa
from benchmarks.gsm8k import load_gsm8k

# Short name -> Bedrock model ID mapping
BEDROCK_MODELS = {
    "claude-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
}


BENCHMARK_CONFIG = {
    "ifeval": {
        "loader": lambda n, seed: load_ifeval(n_instances=n, seed=seed),
        "evaluator": lambda resp, task: evaluate_ifeval(resp, task),
        "default_n": 15,
        "metric_name": "strict_accuracy",
    },
    "hotpotqa": {
        "loader": lambda n, seed: load_hotpotqa(n_instances=n, seed=seed),
        "evaluator": lambda resp, task: f1_score(resp, task.gold_answer),
        "default_n": 10,
        "metric_name": "f1",
    },
    "gsm8k": {
        "loader": lambda n, seed: load_gsm8k(n_instances=n, seed=seed),
        "evaluator": lambda resp, task: gsm8k_evaluate(resp, task.gold_answer),
        "default_n": 10,
        "metric_name": "exact_match",
    },
}


def get_checkpoint_path(results_dir: Path, benchmark: str) -> Path:
    return results_dir / f"{benchmark}_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed_tasks": [], "results": []}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def run_benchmark(
    benchmark: str,
    n_instances: int | None = None,
    model_name: str = "gpt-5",
    provider: str = "openai",
    seed: int = 42,
    resume: bool = False,
):
    """Run a single benchmark end-to-end."""
    config = BENCHMARK_CONFIG[benchmark]
    n = n_instances or config["default_n"]

    # Model-specific results directory
    model_tag = model_name.replace("/", "_").replace(":", "_")
    results_dir = PROJECT_ROOT / "results" / "raw" / f"{benchmark}_{model_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = get_checkpoint_path(results_dir, benchmark)
    checkpoint = load_checkpoint(checkpoint_path) if resume else {"completed_tasks": [], "results": []}

    print("=" * 70)
    print(f"  ContextShapley Full Run: {benchmark.upper()}")
    print("=" * 70)
    print(f"  Model:      {model_name} (provider: {provider})")
    print(f"  Instances:  {n}")
    print(f"  Subsets:    32 per instance")
    print(f"  Total calls: {n * 32}")
    print(f"  Metric:     {config['metric_name']}")
    if resume and checkpoint["completed_tasks"]:
        print(f"  Resuming:   {len(checkpoint['completed_tasks'])} tasks already done")
    print("=" * 70)
    print()

    # Load model based on provider
    if provider == "ollama":
        model = OllamaModel(
            model_name=model_name,
            log_dir=str(log_dir),
        )
    elif provider == "bedrock":
        bedrock_model_id = BEDROCK_MODELS.get(model_name, model_name)
        model = BedrockModel(
            model_id=bedrock_model_id,
            log_dir=str(log_dir),
        )
    elif provider == "openai":
        model = OpenAIModel(
            model_name=model_name,
            use_model_loader=False,
            log_dir=str(log_dir),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    assembler = ContextAssembler()

    print(f"[{benchmark}] Loading tasks...")
    tasks = config["loader"](n, seed)
    print(f"[{benchmark}] Loaded {len(tasks)} tasks")

    evaluator = config["evaluator"]
    all_results = checkpoint["results"]
    completed_ids = set(checkpoint["completed_tasks"])

    start_time = time.time()

    for task_idx, task in enumerate(tasks):
        if task.task_id in completed_ids:
            print(f"[{task_idx+1}/{len(tasks)}] {task.task_id} -- SKIPPED (already done)")
            continue

        print(f"\n[{task_idx+1}/{len(tasks)}] {task.task_id}: {task.query[:60]}...")

        task_scores = {}
        task_responses = {}
        subsets = list(_powerset(ALL_COMPONENT_IDS))

        for sub_idx, subset in enumerate(subsets):
            subset_label = ",".join(sorted(subset)) if subset else "EMPTY"
            call_id = f"{task.task_id}__{subset_label}"

            messages = assembler.assemble(task, subset)

            try:
                result = model.generate(
                    messages=messages,
                    max_tokens=2048,
                    call_id=call_id,
                )
                response_text = result.response_text
                tokens_used = result.total_tokens
            except Exception as e:
                print(f"  ERROR on subset {{{subset_label}}}: {e}")
                response_text = ""
                tokens_used = 0

            score = evaluator(response_text, task)
            task_scores[subset] = score
            task_responses[subset] = response_text

            status = f"{score:.2f}"
            print(
                f"  [{sub_idx+1:2d}/32] {{{subset_label:20s}}} -> {status:5s} "
                f"({tokens_used} tok)",
                end="\r" if sub_idx < 31 else "\n",
            )

        # Compute Shapley values
        sv = all_shapley_values(task_scores, ALL_COMPONENT_IDS)
        im = interaction_matrix(task_scores, ALL_COMPONENT_IDS)
        eff_ok, sv_sum, expected = verify_efficiency_axiom(task_scores, ALL_COMPONENT_IDS)

        # Print summary for this task
        print(f"  Shapley: ", end="")
        for c in ALL_COMPONENT_IDS:
            print(f"{c}={sv[c]:+.3f} ", end="")
        print(f" | Eff={'OK' if eff_ok else 'FAIL'}")

        # Top interactions
        top_interactions = sorted(
            [(k, v) for k, v in im.items() if k[0] < k[1]],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        if any(abs(v) > 0.001 for _, v in top_interactions):
            print(f"  Top interactions: ", end="")
            for (ci, cj), val in top_interactions:
                if abs(val) > 0.001:
                    label = "syn" if val > 0 else "int"
                    print(f"{ci}-{cj}={val:+.3f}({label}) ", end="")
            print()

        task_result = {
            "task_id": task.task_id,
            "query": task.query,
            "gold_answer": task.gold_answer,
            "benchmark": benchmark,
            "scores": {
                ",".join(sorted(k)) if k else "EMPTY": v
                for k, v in task_scores.items()
            },
            "shapley_values": sv,
            "interactions": {
                f"{ci},{cj}": val for (ci, cj), val in im.items() if ci < cj
            },
            "efficiency_axiom_passes": eff_ok,
        }
        all_results.append(task_result)
        completed_ids.add(task.task_id)

        # Save checkpoint after each task
        checkpoint["completed_tasks"] = list(completed_ids)
        checkpoint["results"] = all_results
        save_checkpoint(checkpoint_path, checkpoint)

    elapsed = time.time() - start_time

    # Save final results
    output_path = results_dir / f"{benchmark}_{model_tag}_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    cost = model.get_cost_summary()
    print(f"\n{'='*70}")
    print(f"  {benchmark.upper()} COMPLETE")
    print(f"  Time: {elapsed/60:.1f} min | Calls: {cost['total_calls']}")
    print(f"  Tokens: {cost['total_prompt_tokens']:,} prompt + {cost['total_completion_tokens']:,} completion")
    print(f"  Results: {output_path}")
    print(f"{'='*70}\n")

    return all_results, output_path


def run_all(
    benchmarks: list[str] | None = None,
    n_instances: dict[str, int] | None = None,
    model_name: str = "gpt-5",
    provider: str = "openai",
    resume: bool = False,
):
    """Run all benchmarks sequentially."""
    if benchmarks is None:
        benchmarks = ["ifeval", "hotpotqa", "gsm8k"]

    all_output_paths = {}
    grand_start = time.time()

    for bm in benchmarks:
        n = (n_instances or {}).get(bm, BENCHMARK_CONFIG[bm]["default_n"])
        results, path = run_benchmark(
            benchmark=bm,
            n_instances=n,
            model_name=model_name,
            provider=provider,
            resume=resume,
        )
        all_output_paths[bm] = str(path)

    grand_elapsed = time.time() - grand_start
    print(f"\n{'#'*70}")
    print(f"  ALL BENCHMARKS COMPLETE in {grand_elapsed/60:.1f} minutes")
    print(f"  Results:")
    for bm, path in all_output_paths.items():
        print(f"    {bm}: {path}")
    print(f"{'#'*70}")

    # Save manifest
    manifest = {
        "model": model_name,
        "benchmarks": all_output_paths,
        "total_time_min": grand_elapsed / 60,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = PROJECT_ROOT / "results" / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextShapley Full Experiment")
    parser.add_argument("--benchmark", choices=["ifeval", "hotpotqa", "gsm8k", "all"], default="all")
    parser.add_argument("--n-instances", type=int, default=None)
    parser.add_argument("--model", default="gpt-5",
                        help="Model name. For bedrock use: claude-opus, claude-sonnet, claude-haiku")
    parser.add_argument("--provider", choices=["bedrock", "openai", "ollama"], default="openai",
                        help="openai=direct API via OPENAI_API_KEY, bedrock=AWS Bedrock, ollama=local server")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.benchmark == "all":
        n_map = {}
        if args.n_instances:
            n_map = {bm: args.n_instances for bm in ["ifeval", "hotpotqa", "gsm8k"]}
        run_all(model_name=args.model, provider=args.provider, n_instances=n_map, resume=args.resume)
    else:
        run_benchmark(
            benchmark=args.benchmark,
            n_instances=args.n_instances,
            model_name=args.model,
            provider=args.provider,
            resume=args.resume,
        )
