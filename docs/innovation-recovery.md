# Full-state innovation recovery

OEL provides experimental, opt-in recovery for full-state observations in the
HCW, SS-J2, TH and YA relative EKFs, and the absolute ECI orbit EKF. The default
is off. Existing estimation without this option retains its numerical behavior.

This is a knowledge-estimation component. Enabling it does not configure the
separate flight-software navigation stack or qualify it for onboard use.

## Choose a strategy

Both strategies first compare the raw normalized innovation squared (NIS) with
a chi-square threshold. A suspicious observation holds the prediction while
the estimator waits for another observation; the algorithm uses no future
measurements or retrospective replay.

- `predictive_consistency`: propagate a candidate initialized from the
  suspicious full-state observation using the same estimator dynamics. Recover
  when subsequent observations disagree with the held track and agree with the
  candidate. This can avoid resets from incompatible consecutive outliers.
- `persistence`: recover after the configured number of consecutive high-NIS
  observations, without requiring candidate agreement. This simpler approach
  can track continuing changes more readily.

Neither strategy is a general replacement for the other. Continuing burns can
violate the candidate's coast prediction, delaying consistency-based recovery.
Known commanded burns should enter the prediction model when the surrounding
application supports that information.

## Configure knowledge estimation

For example, start from `configs/validation_hcw_relative_ekf_single.yaml`, which
already selects full relative-state measurements. Add this block under
`objects.chaser.knowledge.estimation.ekf`:

```yaml
innovation_recovery:
  enabled: true
  strategy: predictive_consistency
  nis_probability: 0.999
  confirmation_count: 2
  covariance_inflation: 4.0
  max_gap_s: 120.0
```

The same option applies to `relative_ss_j2_ekf`, `relative_th_ekf`,
`relative_ya_ekf`, and `ekf`. Relative filters require
`measurement_model: relative_state`; absolute knowledge estimation requires
`measurement_model: state`. Angles-only, range-only, range/range-rate and
position-only observations are not supported by this recovery policy. The
measurement covariance must be positive definite.

Validate an edited scenario before running it:

```bash
python run_simulation.py --config configs/my_recovery_study.yaml --validate-only
python run_simulation.py --config configs/my_recovery_study.yaml
```

Use a distinct scenario name and output directory. Select `strategy:
persistence` for the simpler baseline. Omit `innovation_recovery`, or set
`enabled: false`, to disable recovery.

## Python API

Pass the optional keyword to an existing estimator constructor:

```python
import numpy as np

from sim.estimation.innovation_recovery import InnovationRecoveryConfig
from sim.estimation.relative_hcw_ekf import HCWRelativeEKFEstimator

estimator = HCWRelativeEKFEstimator(
    mean_motion_rad_s=0.0011,
    dt_s=10.0,
    process_noise_diag=np.array([1e-10] * 3 + [1e-14] * 3),
    meas_noise_diag=np.array([1e-6] * 3 + [1e-10] * 3),
    numeric_backend="python",
    innovation_recovery=InnovationRecoveryConfig(enabled=True),
)
```

`SSJ2RelativeEKFEstimator`, its `from_chief_state` factory,
`THRelativeEKFEstimator`, `YARelativeEKFEstimator`, and `OrbitEKFEstimator`
accept the same recovery option. Keep their existing dynamics, frame, epoch
and measurement-origin requirements. Recovery does not extend the relative
models beyond their chief-orbit, separation or linearization assumptions.
Both Python and Rust numerical backends
use the shared policy.

The raw `update(belief, measurement, t_s)` interface is unchanged. Recovery
decisions appear in `estimator.last_update_diagnostics.recovery_diagnostics`.
The diagnostics distinguish `nominal`, `pending_discrepancy`,
`unresolved_discrepancy`, `transient_discrepancy` and
`persistent_discrepancy`, with actions `accepted`, `held` or `reinitialized`.
Acquisition and decision epochs are recorded separately. With the default
confirmation count, a resolvable change needs at least one later observation.

Pending history expires after the maximum observation gap. Replacing the
returned belief's state or covariance invalidates that history. After changing
frame/reference identity or applying a discontinuous reference correction,
call `estimator.reset_innovation_recovery()`. Routine epoch-dependent TH/YA
prediction remains owned by those estimators.

## Inspect recovery evidence

Knowledge consistency summaries retain holds, transient discrepancies,
reinitializations and the latest recovery acquisition/decision epochs. With
review output enabled, query the existing metrics table:

```bash
python -m sim.review outputs/my_recovery_study --query "SELECT metric_name, value, object_id, chief_id FROM metrics WHERE source = 'innovation_recovery' ORDER BY metric_name"
```

The maneuver detector retains raw NIS evidence even when recovery holds a
measurement. Recovery counts are separate from its maneuver hypotheses.

## Limits and checks

A persistent discrepancy is not proof of a maneuver. A dynamically plausible
sensor bias, wrong association, or model mismatch may produce similar
evidence. Independent errors between acquisition epochs are assumed, while
full within-epoch covariance is supported. Thresholds are consistency
heuristics, not calibrated maneuver probabilities.

During a hold, covariance describes the no-change prediction. It is not a
calibrated mixture over possible unknown maneuvers and can understate the
actual error during a change. Full-state reinitialization trusts the current
observation with the configured covariance inflation. Synthetic regression
evidence does not establish real-sensor accuracy or operational readiness.

From a public source checkout with development dependencies installed, run:

```bash
python -m pytest -q sim/tests/test_innovation_recovery.py sim/tests/test_relative_hcw_ekf.py sim/tests/test_relative_th_ekf.py sim/tests/test_ekf_maneuver_detection.py
```

These checks exercise enabled/disabled behavior, both numerical backends,
measurement origins, eccentric-chief references, acquisition/output timing,
dropouts, replaced beliefs, correlated covariance, persistence comparison and
review metrics.

The concept was inspired by [@sneed-and-feed's PR #97](https://github.com/adamcohen8/orbital-engineering-lab/pull/97).
The shared causal implementation and its tests were independently written;
the original patch was not incorporated.
