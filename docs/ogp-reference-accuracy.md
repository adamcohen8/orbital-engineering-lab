# OGP backend reference accuracy

Use this comparison when selecting the Python or Rust backend for native TEME propagation. OEL computes each backend’s position and velocity differences against the same published verification rows. The command reports the default backend from the active dispatcher. Run it in the environment that will execute the study.

```sh
python -m sim.dynamics.orbit.reference_accuracy --output-dir outputs/ogp-accuracy
```

The included near-Earth cases come from Vallado, Crawford, Hujsak and Kelso’s SGP4 verification suite. Vanguard 1 spans 72 hours at six-hour intervals. CBERS‑2 spans 48 hours at two-hour intervals. Each fixture records its exact TLE, the commit-pinned source URLs and the SHA-256 of the authors’ original ephemeris. The reference uses TEME kilometers and kilometers per second. [Paper and verification materials](https://celestrak.org/publications/AIAA/2006-6753/)

| Case | Epochs | Backend | Maximum position difference (m) | Maximum velocity difference (mm/s) |
| --- | ---: | --- | ---: | ---: |
| Vanguard 1 | 13 | Python | 7.89084 | 6.26337 |
| Vanguard 1 | 13 | Rust | 0.00000682 | 0.00077049 |
| CBERS‑2 | 25 | Python | 0.15208 | 0.15165 |
| CBERS‑2 | 25 | Rust | 0.00000678 | 0.00068941 |

These results used the development source with the 0.18.4 native orbit wheel on Apple Silicon. The limits default to 1 m and 1 mm/s. Each backend must return a state at every requested epoch and satisfy both limits. The command exits 1 if any backend misses a limit or fails to return a state. This is the expected outcome for the included Vanguard case. The figures measure agreement with published numerical vectors. They do not quantify orbit prediction error against tracking observations.

Each case writes `backend_accuracy.json` and separate Python and Rust CSV and JSON reports with epoch-level differences and propagation errors. In the detailed report, `matched_sample_count` counts epochs with comparable states. Use the summary’s `passed` field for the tolerance verdict. To supply a different reference, use the following command with a positive or zero limit in each unit:

```sh
python -m sim.dynamics.orbit.reference_accuracy --reference my_case.json \
  --position-limit-m 1 --velocity-limit-mm-s 1 --output-dir outputs/my-case
```

Use the structure of either bundled fixture under `sim/dynamics/orbit/data/`. The required fields are `reference_frame` (`teme`), `tle.line1`, `tle.line2`, and a nonempty `samples` array. Each sample supplies elapsed minutes in `tsince_min` and three-component arrays in `position_teme_km` and `velocity_teme_km_s`.
