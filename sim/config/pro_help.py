from __future__ import annotations

from sim.config.help import ConfigHelpEntry, _option

PRO_CONFIG_HELP_ENTRIES: tuple[ConfigHelpEntry, ...] = (
    ConfigHelpEntry(
        path="analysis.study_type",
        title="Analysis Study Type",
        description="Top-level campaign/study mode.",
        aliases=("study type", "analysis type", "campaign type"),
        options=(
            _option("monte_carlo", "Run repeated samples from configured variations."),
            _option("sensitivity", "Run one-at-a-time, Latin-hypercube, or two-parameter-grid sensitivity studies."),
        ),
        example="analysis:\n  enabled: true\n  study_type: monte_carlo",
        visibility="pro",
    ),
    ConfigHelpEntry(
        path="analysis.sensitivity.method",
        title="Sensitivity Method",
        description="Sampling layout for analysis.study_type: sensitivity.",
        aliases=("sensitivity method", "sensitivity mode", "lhs"),
        options=(
            _option("one_at_a_time", "Sweep one parameter at a time around the baseline."),
            _option("lhs", "Latin hypercube sample across configured parameter distributions."),
            _option("two_parameter_grid", "Grid sweep for two configured parameters."),
        ),
        example="analysis:\n  sensitivity:\n    method: lhs",
        visibility="pro",
    ),
    ConfigHelpEntry(
        path="analysis.execution.failure_policy",
        title="Analysis Failure Policy",
        description="Controls whether campaign execution stops on the first failed case or continues collecting results.",
        aliases=("failure policy", "campaign failure policy"),
        options=(
            _option("fail_fast", "Stop the study on the first failed run."),
            _option("continue", "Keep running remaining cases after individual failures."),
        ),
        example="analysis:\n  execution:\n    failure_policy: continue",
        visibility="pro",
    ),
    ConfigHelpEntry(
        path="analysis.baseline.mode",
        title="Analysis Baseline Mode",
        description="Controls whether campaign analysis compares against no baseline, a fresh baseline run, or a saved summary file.",
        aliases=("baseline mode", "baseline"),
        options=(
            _option("none", "Do not compute baseline deltas."),
            _option("run", "Run a baseline case from the current scenario."),
            _option("file", "Load the baseline from analysis.baseline.summary_json."),
        ),
        example='analysis:\n  baseline:\n    mode: "run"',
        visibility="pro",
    ),
    ConfigHelpEntry(
        path="monte_carlo.variations.mode",
        title="Monte Carlo Variation Mode",
        description="How an individual Monte Carlo variation samples values.",
        aliases=("variation mode", "monte carlo mode", "mc mode"),
        options=(
            _option("choice", "Pick from the explicit options list."),
            _option("uniform", "Sample uniformly from low to high."),
            _option("normal", "Sample from mean/std."),
        ),
        example="monte_carlo:\n  variations:\n    - parameter_path: simulator.dt_s\n      mode: choice\n      options: [0.5, 1.0]",
        visibility="pro",
    ),
)
