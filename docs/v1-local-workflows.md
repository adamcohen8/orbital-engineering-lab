# OEL v1.0 local workflows

Use the installed OEL environment and the matching public source release.
Local tools run without license activation. The legacy Python module names
remain part of the import and CLI contracts.

| Workflow | Entry point |
| --- | --- |
| Single scenario, Monte Carlo, sensitivity, or covariance study | `oel-sim --config SCENARIO.yaml` |
| Config queue | `oel-sim --campaign QUEUE.yaml` |
| Controller Bench and tuning | `oel-sim --controller-bench SUITE.yaml`; `python -m sim.controller_lab --help` |
| Catalog, screening, OD, and campaigns | `python -m sim.scale --help` |
| Conjunction, covariance sensitivity, and avoidance | `python -m sim.pro_conjunction --help` |
| Trajectory optimization | `python -m sim.pro_trajectory_optimization --help` |
| Constellation design search | `python -m sim.pro_constellation_optimization --help` |
| RF engineering | `sim.pro_communications` Python API |
| Local OD and satellite laser ranging | [Tracking OD](tracking-od.md); `python -m sim.estimation --help` for evidence and state products |
| Flight software authoring and FSWDK | `oel fswdk --help` |
| Generic spacecraft twin authoring | `oel-mcp` and `sim.spacecraft_twin` |
| cFS/SIL | [Bridge and mock endpoint](../integrations/cfs_sil/README.md) |
| Deterministic report preparation and audit | [Local MCP reporting](oel-mcp.md) |
| Experimental simulated operations adapter | `python -m sim.agent_operations --help` |

Use `--safe-validate` to inspect unfamiliar scenario or queue configs before
trusting plugins. Validate trusted inputs before execution and use new output
directories. For controlled execution budgets, declare the resource profile
and worker policy in the scenario. Compare runs using the same seed, inputs,
and model settings. Use `oel.describe_capabilities.v1` for tools available in
the current MCP session.

The included JSON examples are synthetic. Run the commands from the public
source root and use fresh output paths:

```bash
python -m sim.pro_conjunction assess examples/workflows/pro_conjunction/crossing.json --output outputs/conjunction
python -m sim.pro_conjunction replay outputs/conjunction
python -m sim.pro_constellation_optimization solve examples/workflows/pro_constellation_optimization/problem.json --output outputs/constellation
python -m sim.pro_trajectory_optimization solve examples/workflows/pro_trajectory_optimization/constrained_impulse.json --output outputs/trajectory
```

For the ITU-R P.676-13 Annex 2 slant gas mode, obtain the Part 1 table from
the [ITU recommendation's related files](https://www.itu.int/oth/R1101000002/en)
and set `OEL_P676_ANNEX2_PART1` to the local file. Use caller-supplied
antenna, atmosphere, and station inputs for engineering studies.

For the optional Schwarzschild comparison, obtain Orekit 13.1.7 and the
Hipparchus 4.0.3 jars named in
[the recipe](../tools/orekit_schwarzschild.py) from Maven Central. Use
JDK 11.0.25. The recipe verifies each jar and regenerates six fixed-state
accelerations and six six-hour trajectories in GCRF at J2000 TT. Outputs use
SI units and are written to a new local directory.

```bash
python -I tools/orekit_schwarzschild.py --jars /path/to/jars --output-dir outputs/schwarzschild-reference
```

The [offline suite](tracking-od.md#offline-release-validation) is the installed
runtime gate. The optional Orekit comparison reports unavailable prerequisites
separately. Generated output stays in the local output directories.
