# Orbital Engineering Lab RPO Trainer Preview

This is the browser-native OEL RPO Trainer Preview. It is a lightweight web app
intended for social-media clickthroughs, quick demos, and small Pursuit Arcade
competitions.

Open locally:

```bash
python -m http.server 8765 --directory web/rpo-trainer-preview
```

Then visit:

```text
http://localhost:8765
```

## Included

- Unified level selector for Tutorial, Sandbox, Pursuit Arcade, and the hosted
  RPO Duel Beta.
- Persistent Browser Preview framing plus a direct link to the full trainer's
  installation guide.
- Computer-preview Pilot/Operator Preview selector for Tutorial and Sandbox.
  Operator mode scripts impulsive RIC burns and shows the planned deterministic
  trajectory before playback.
- Optional 3D view for computer Sandbox play and Operator Sandbox playback.
  Use the `3D`/`2D` button, mouse drag to orbit, Shift-drag or middle-drag to
  pan, and the scroll wheel to zoom. Axes share one scale, +R starts upward,
  and automatic zoom keeps both satellites visible. Camera interaction
  temporarily limits playback to 1x, then restores the selected speed;
  paused playback stays paused. Recenter, Fit both, and RI/RC/IC presets are
  available. Mobile Sandbox supports 3D in landscape: one-finger rotation and
  pinch zoom stay locked on the target, with automatic zoom to retain the
  chaser. Portrait temporarily uses 2D and restores the 3D selection and angle
  on return to landscape. Other levels and the Operator burn editor retain 2D.
- Coordinate-frame convention toggle for OEL Default and Space Force-style
  positive in-track display, persisted in local browser storage and carried
  into RPO Duel through its launch and return URLs.
- Automatic mobile/computer layout detection with manual view switching.
- Tutorial mode based on the Level 0 RIC-control lesson.
- Sandbox preflight matching the downloadable field contract: all six target
  classical orbital elements plus all six chaser target-centered RIC position
  and relative-rate values. Pilot and Operator Sandbox apply the edited values
  before simulation or burn planning. Live motion uses deterministic two-body
  propagation; the coast overlay uses that target-aware path for a circular
  target and the numerical Tschauner-Hempel projection for any nonzero target
  eccentricity.
- Web-only Pursuit Arcade multi-round gameplay with deterministic replay
  validation. Pursuit Arcade is intentionally not listed in the downloadable
  trainer launcher.
- Mobile-friendly portrait and landscape controls with compact speed-multiple
  buttons, explicit camera toggling, and long-press selection suppression.
- Bounded fast-coast frame scheduling that preserves fractional pending work,
  reports any dropped simulated time, and shows the achieved simulation rate
  against active wall time. Long suspensions account for both clipped and
  capped work, and terminal completion does not count planned no-op steps.
- RI and RC canvas plots with HCW projection for the circular tutorial,
  target-orbit-aware Sandbox projection, and browser-native arcade projections
  for Pursuit Arcade, including stable goal rings and exact pass-tick clear
  range reporting.
- Keyboard controls for computer users and touch controls for mobile users.
- Browser-started music matched to each mode.
- Hosted leaderboard submission hooks with optional email ownership
  verification.
- Debrief and repository links for follow-up.

## Lightweight Analytics

The hosted preview can send privacy-focused Plausible and Vercel Web Analytics
events for product-funnel questions: preview views, tutorial starts, primer
completion, tutorial completion, sandbox starts, download clicks, music
toggles, and returns to the level selector.

Analytics are disabled for `file://`, `localhost`, and `127.0.0.1` runs. The
static page reads its analytics configuration from:

```html
<meta name="oel-analytics-provider" content="plausible,vercel" />
<meta name="oel-analytics-domain" content="adamcohen8.github.io" />
<meta name="oel-vercel-analytics-script" content="/_vercel/insights/script.js" />
<meta name="oel-vercel-analytics-hosts" content=".vercel.app,orbital-engineering-lab.vercel.app" />
```

Completion events use coarse buckets for time, delta-v, and closest range.
Browser analytics does not send raw trajectories, per-frame controls, names,
emails, or a player identifier. Vercel Analytics is only loaded on configured
Vercel-hosted domains so GitHub Pages and local runs do not request the Vercel
insights script. Pursuit Arcade leaderboard submissions are a separate explicit
form submit that sends the username, optional email, and attempt packet to the
hosted validation API.

## Not Included

This preview intentionally does not include the full OEL Python simulator,
scenario YAML support, all downloadable trainer levels, downloadable-game
recordings, or full debrief reports. Pursuit Arcade leaderboard attempts are
validated by replaying the browser-native deterministic arcade engine, not by
trusting client-submitted scores. See `docs/physics-contract.md` for the model
boundary.

## Contract Checks

The checked-in browser contract fixtures are generated from the downloadable
Level 0, Sandbox, and Pursuit Arcade scenario YAML. Level 0 reference
trajectories are generated with OEL's two-body engine and compared against the
browser HCW integrator at the tolerance documented in `docs/physics-contract.md`.
Sandbox contract checks also pin the downloadable setup field order, defaults,
numeric bounds, and RIC velocity unit conversion.

Run the complete preview check from this directory:

```bash
npm test
```

## Published Schemas

The `schemas/` directory serves the 23 public JSON Schema IDs at
`https://orbital-engineering-lab.vercel.app/schemas/`. Its files are byte-for-byte
copies of their authoritative public source schemas. Run
`node tools/sync-schemas.mjs --write` after changing a source schema; `npm test`
checks the exact public inventory, URLs, and content. Pro-only schemas are
distributed with Pro source and are not copied into the public site.

## Hosted RPO Duel release gate

The Pursuit Arcade API requires the `publish_unclaimed_arcade_attempt` and
`claim_arcade_attempt` functions in `supabase/schema.sql` before deployment.
Configure `OEL_ARCADE_PUBLIC_ORIGIN` as the exact HTTPS site origin. Email
verification links never use request Host or forwarded headers; if the origin
is missing or malformed, the API does not create or send a verification token.
Both functions run the player ownership check and leaderboard write in one
Postgres transaction. If the migration is absent, submissions and verification
fail closed. Anonymous usernames can publish provisional scores; a verified
username can be updated only by an unused, unexpired email token tied to the
exact validated attempt. A recipient-wide one-hour reservation bounds sends
across usernames and retries until the token is claimed; a successful claim
releases only its matching reservation so the owner can submit another score.
The migration resets legacy verified winners that lack a consumed token bound
to the current email and exact eligible attempt. Before enabling this change on a live
database, exercise two different email tokens claiming one username at once,
token replay, a lower verified attempt following a higher provisional score,
and a provisional submit racing a successful claim. Browser submissions with
a disallowed Origin are rejected before validation or writes; nonbrowser
requests still need provider or IP rate limiting.

Treat that as a production gate whenever email sending is enabled: configure a
Vercel edge rate rule for `POST /api/submit-attempt` keyed by the platform's
observed client IP, and an operator-controlled account-wide send ceiling or
budget that can stop mail, with monitoring. Choose thresholds for deployed
traffic and the mail provider plan; the repository does not set those
provider-console controls, and alerts alone do not bound sends.
The recipient reservation limits repeat sends to one normalized address, not
total sends across distinct addresses. Local mocked-fetch tests do not verify
edge rules or provider quotas. If either production control is unavailable,
leave `RESEND_API_KEY` unset; leaderboard submissions continue to work without
email.

For a release that includes RPO Duel, deploy the production Cloudflare Worker
first, confirm that `oel-rpo-duel-url` contains its exact stable HTTPS URL, and
then deploy this directory to the production Vercel project. After both
deployments, run:

```bash
npm run verify:hosted-duel
```

The command bypasses ordinary browser caches and fails unless the live Vercel
selector lists RPO Duel with a Beta label, launches the production Worker, and
the live Duel page links back to the production selector. The release remains
incomplete until this check and a browser launch-and-return smoke test pass.
