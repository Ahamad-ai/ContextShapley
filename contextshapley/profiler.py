"""
ContextProfiler: High-level API for profiling context component contributions.

This is the main user-facing class. Full implementation in Phase 4.
"""

from dataclasses import dataclass, field


@dataclass
class ProfileReport:
    """Results from profiling a set of task instances."""
    shapley_values: dict[str, float] = field(default_factory=dict)
    interactions: dict[tuple[str, str], float] = field(default_factory=dict)
    per_instance: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ContextProfiler:
    """
    Profile context component contributions for a model + benchmark.

    Usage:
        profiler = ContextProfiler(model_name="gpt-4o-mini", ...)
        report = profiler.profile(task_instances)
        print(report.shapley_values)
    """

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        self.model_name = model_name
        self.config = kwargs

    def profile(self, task_instances, **kwargs) -> ProfileReport:
        """Full profiling pipeline. Implemented in Phase 4."""
        raise NotImplementedError("Full profiling pipeline is built in Phase 4.")
