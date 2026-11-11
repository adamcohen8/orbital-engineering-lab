# RPO Trainer

The OEL RPO Trainer is an educational, single-player environment for building
intuition about relative orbital motion. It is intended for aerospace students,
early-career space operators, and instructors. It is not an operational mission
rehearsal or training-qualification system.

## Launch

Install and activate OEL as described in [Installation](installation.md), then
run:

```bash
python run_game.py
```

The Trainer selects the Rust physics and flight-software backend by default.
The `game` installation profile requires the `oel_rust_game` and
`oel_rust_orbit` native wheels.
Use `python run_game.py --backend python` to select the Python backend explicitly.

The selector offers Pilot and Operator modes. Use Up/Down or W/S to choose a
level, Left/Right to change assists, Enter or Space to launch, and Escape to
return to the selector.

## 3D View (Downloadable Trainer)

The 3D view is available in every public level except Cislunar Rendezvous.
Levels start with the two RIC plane views. Click **3D** at the top right of
those panels to open an orthographic 3D view; click **2D** to return. Switching
preserves the simulation and remembers the 3D camera within the session.

- Left-drag inside the plot to orbit around the camera focus.
- Shift-left-drag or middle-drag to pan the focus in the viewing plane.
- Scroll to zoom uniformly across all three axes.
- **Recenter** returns the focus to the target; **Fit both** frames both spacecraft
  using the current camera angle and viewport dimensions, with 20% padding.
  The initial 3D view uses the same framing. The camera automatically zooms out
  when necessary to keep both spacecraft visible, including during rotation,
  panning, window resizing, and spacecraft motion. It does not auto-zoom back in.
- **RI**, **RC**, and **IC** snap the camera to the corresponding plane.

Camera drags temporarily cap effective simulation speed at 1×. Releasing the
mouse restores the selected speed; scroll zoom restores it after 250 ms without
another wheel event. Paused simulation stays paused. Speed selections made
during adjustment are retained. The HUD shows active and selected coast speeds.
The grid, labeled RIC axes, trails, existing coast prediction, and kilometer
scale provide spatial references. RIC keyboard thrust controls are unchanged.
Level 3 uses a spherical forbidden shell (0.95–5.8 km from the target), with a
25° half-angle approach cone removed along −R and a spherical cavity inside
0.95 km. The separate 0.15 km keepout still applies inside that cavity.
The 3D view shows translucent red surfaces and wireframe boundaries, plus orange
keepout and green goal-range rings. The 2D views show central cuts of the same
shell and cone.
The initial view includes the volumes; **Fit both** focuses on the spacecraft.
Position and swept-segment violation checks use the same shell/cone geometry.
The view is available in desktop Pilot and Operator modes, not the hosted trainer.
Existing box, cylindrical, spherical, and sector constraints retain their scoring
geometry. Inspection gates, Sun-angle guidance, goal regions, and formation paths
are displayed in 3D. Initial framing includes nearby mission geometry.

## Interaction Modes

Pilot mode uses direct RIC translation controls:

- W/S: radial +/-R
- A/D: in-track +/-I
- Left/Right arrows: cross-track +/-C
- Space: pause or resume
- R: reset the current attempt
- Up/Down: adjust runtime speed
- C: switch the 2D Sandbox RI/RC camera between full-trajectory and current-spacecraft framing
- O/P: switch a supported RI or RC panel to an orbit-plane view
- D: open the debrief folder from the pass/fail screen, when available
- Escape: leave the active level

Operator mode uses the same R/I/C frame language but replaces continuous
keyboard inputs with time-tagged burn rows, trajectory preview, and view-only
playback.

## Training Content

Bundled lessons cover the controls tutorial, passive relative motion, V-bar and
R-bar approaches, terminal rendezvous, cross-track inspection, Sun-angle
inspection, elliptic-orbit approaches and natural-motion circumnavigation,
evasion, pursuit, and a bonus cislunar rendezvous lesson. Selector availability
varies by interaction mode.

The displays emphasize target-centered RIC axes, trajectory history, relative
velocity, commanded thrust, keepout regions, goal regions, approach or
inspection gates, and current objective status. Supported lessons also expose a
coast-from-here prediction and burn markers.

## Debriefs And Evidence

Structured training levels write attempt evidence below:

```text
outputs/game_debriefs/<scenario_id>/attempt_.../
```

The evidence includes a Markdown debrief, `summary.json`, a mission timeline,
RIC trajectory plots, relative range and velocity histories, cumulative
delta-v, and control-command plots. Open-ended or replayable modes may omit
reports, as documented by the selected scenario.

Typical debrief fields include closest approach, final range, final relative
speed, keepout time, approximate delta-v, objective success or miss reason, and
the level pass/fail result. Treat these artifacts as educational evidence for
the exact attempt, not as flight qualification.

For classroom setup and facilitation, see the
[RPO Trainer Instructor One-Pager](rpo-trainer-instructor-one-pager.md). For the
browser multiplayer experiment, see the
[RPO Duel Beta](../web/rpo-duel-prototype/README.md).
