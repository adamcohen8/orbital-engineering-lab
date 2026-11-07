import { createSandbox3D } from "./sandbox-camera-3d.js?v=mobile-game-gestures-2026-09-15";
import {
  buildChallengeRecord,
  createPursuitArcadeSession,
  DEFAULT_PURSUIT_CHALLENGE,
  ellipticLinearCoastStates,
  gameTickDtS,
  relativeRicState,
  stateFromRelativeRic,
  stepControlledTwoBodyPair,
  validateAttemptPacket,
} from "./competition/arcade-engine.js";
import { PREVIEW_LEVEL_CONTRACTS } from "./preview-contract.js";
import {
  PREVIEW_FIXED_DT_S as FIXED_DT_S,
  PREVIEW_MAX_ACCEL_KM_S2 as MAX_ACCEL_KM_S2,
  PREVIEW_MEAN_MOTION_RAD_S as MEAN_MOTION,
  PREVIEW_MU_KM3_S2 as MU,
  PREVIEW_TARGET_A_KM as TARGET_A_KM,
  stepHcwStateInPlace,
} from "./preview-physics.js";
import {
  DEFAULT_SANDBOX_SETUP,
  SANDBOX_SETUP_FIELDS,
  sandboxOrbitPeriodS,
  sandboxProjectionModel,
  sandboxRelativeSeed,
  sandboxSetupInputValues,
  validateSandboxSetup,
} from "./sandbox-setup.js";
import {
  sandboxEllipticLinearCoastStates,
  sandboxTargetStateAt,
} from "./sandbox-projection.js";
import {
  advanceFrameTiming,
  createFrameTimingState,
  MAX_PENDING_SIMULATION_S,
  reconcileFrameTimingExecution,
  resetFrameTimingState,
} from "./frame-timing.js";

const ORBIT_PERIOD_S = (2 * Math.PI) / MEAN_MOTION;
const MAX_STEPS_PER_FRAME = 32;
const MAX_GHOST_DRAW_POINTS = 120;
const TUTORIAL_TARGET_PATH_POINTS = 181;
const SPEED_OPTIONS = [1, 2, 5, 10, 25, 50, 100, 200];
const MANEUVER_CONTROL_SPEED = 10;
const TRAIL_LIMIT = 1200;
const MIN_PLOT_SPAN_KM = 0.005;
const PLOT_SCALE_MARGIN = 1.2;
const OPERATOR_BURN_MAX_DV_M_S = 5;
const OPERATOR_BURN_SPACING_S = 10;
const OPERATOR_PREVIEW_POINTS = 240;
const OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE = 200;
const OPERATOR_TUTORIAL_STAGE_DURATION_S = 3000;
const OPERATOR_TUTORIAL_BURN_TIME_S = 50;
const OPERATOR_TUTORIAL_BURN_DELTA_V_M_S = 0.25;
const OPERATOR_BURN_CINEMATIC_SPEED_MULTIPLE = 10;
const OPERATOR_BURN_CINEMATIC_LOOKAHEAD_S = 5;
const OPERATOR_BURN_VISUAL_DURATION_BASE_S = 1.0;
const OPERATOR_BURN_VISUAL_DURATION_PER_M_S = 0.2;
const OPERATOR_BURN_VISUAL_DURATION_MIN_S = 1.0;
const OPERATOR_BURN_VISUAL_DURATION_MAX_S = 2.0;
const SATELLITE_SPRITE_DIAMETER_KM = 0.006;
const SATELLITE_ICON_SIZE_PX = 20;
const TARGET_MARKER = "#f55c5c";
const CHASER_MARKER = "#f5cd5c";
const OPERATOR_PROJECTION_COLOR = "rgba(238, 184, 92, 0.95)";
const OPERATOR_PROJECTION_HIGHLIGHT = "rgba(255, 224, 142, 0.9)";
const OPERATOR_BURN_MARKER_COLOR = "rgba(255, 146, 67, 0.96)";
const OPERATOR_PROBE_COLOR = "rgba(86, 202, 245, 0.96)";
const OPERATOR_PROBE_PICK_RADIUS_PX = 10;
const BUILD_ID = "web-preview-product-contract-2026-08-20";
const ARCADE_BUILD_ID = `${BUILD_ID}-competition-local`;
const ARCADE_CHALLENGE_RECORD = buildChallengeRecord(DEFAULT_PURSUIT_CHALLENGE);
const LEADERBOARD_REFRESH_MS = 30000;
const PLAUSIBLE_ANALYTICS_SCRIPT_SRC = "https://plausible.io/js/script.js";
const VERCEL_ANALYTICS_SCRIPT_SRC = "/_vercel/insights/script.js";
const RPO_DUEL_URL = window.location.pathname.startsWith("/trainer") &&
  ["", "localhost", "127.0.0.1", "::1"].includes(window.location.hostname)
  ? new URL("/", window.location.href).href
  : document.querySelector('meta[name="oel-rpo-duel-url"]')?.content.trim() || "";
const ANALYTICS_LOCAL_HOSTNAMES = new Set(["", "localhost", "127.0.0.1", "::1"]);
const PREVIEW_DEV_HOSTNAMES = new Set(["", "localhost", "127.0.0.1", "::1"]);
const PRIMER_AMPLITUDES_KM = { r: 0.65, i: 0.75, c: 0.65 };
const MUSIC_TRACKS = {
  selector: "./assets/01_insert_coin_to_orbit.wav",
  tutorial: "./assets/10_training_grid_sunrise.wav",
  sandbox: "./assets/06_casting_the_orbit_line.wav",
  arcade: "./assets/21_pursuit_arcade_overdrive_no_siren_demo.wav",
  arcadeBoss: "./assets/28_high_shred_boss_riff.wav",
};
const PLAY_MODE_KEY = "oelPreviewPlayMode";
const FRAME_CONVENTION_KEY = "oelPreviewFrameConvention";
const PLAY_MODES = ["pilot", "operator"];
const FRAME_CONVENTIONS = ["oel_default", "space_force"];

const el = {
  shell: document.querySelector(".trainer-shell"),
  levelSelector: document.querySelector("#levelSelector"),
  selectorMusicButton: document.querySelector("#selectorMusicButton"),
  selectorViewButton: document.querySelector("#selectorViewButton"),
  selectorInstallLink: document.querySelector("#selectorInstallLink"),
  selectorModeButton: document.querySelector("#selectorModeButton"),
  selectorFrameButton: document.querySelector("#selectorFrameButton"),
  selectorFrameButtons: Array.from(document.querySelectorAll("[data-selector-frame-button]")),
  selectorPreviewTitle: document.querySelector("#selectorPreviewTitle"),
  selectorPreviewBudget: document.querySelector("#selectorPreviewBudget"),
  selectorPreviewScope: document.querySelector("#selectorPreviewScope"),
  selectorPreviewObjective: document.querySelector("#selectorPreviewObjective"),
  selectorPreviewBrief: document.querySelector("#selectorPreviewBrief"),
  selectorPreviewCriteria: document.querySelector("#selectorPreviewCriteria"),
  selectorPreviewNotes: document.querySelector("#selectorPreviewNotes"),
  selectorPlayButton: document.querySelector("#selectorPlayButton"),
  leaderboardPanel: document.querySelector("#leaderboardPanel"),
  leaderboardList: document.querySelector("#leaderboardList"),
  leaderboardMeta: document.querySelector("#leaderboardMeta"),
  leaderboardRefresh: document.querySelector("#leaderboardRefresh"),
  pauseButton: document.querySelector("#pauseButton"),
  resetButton: document.querySelector("#resetButton"),
  levelSelectButton: document.querySelector("#levelSelectButton"),
  musicButton: document.querySelector("#musicButton"),
  viewButton: document.querySelector("#viewButton"),
  modeLabel: document.querySelector("#modeLabel"),
  objectiveTitle: document.querySelector("#objectiveTitle"),
  objectiveText: document.querySelector("#objectiveText"),
  riTitle: document.querySelector("#riTitle"),
  riSubtitle: document.querySelector("#riSubtitle"),
  rcTitle: document.querySelector("#rcTitle"),
  rcSubtitle: document.querySelector("#rcSubtitle"),
  riPanel: document.querySelector("#riPanel"),
  rcPanel: document.querySelector("#rcPanel"),
  riCanvas: document.querySelector("#riCanvas"),
  rcCanvas: document.querySelector("#rcCanvas"),
  rangeMetric: document.querySelector("#rangeMetric"),
  speedMetric: document.querySelector("#speedMetric"),
  dvMetric: document.querySelector("#dvMetric"),
  timeMetric: document.querySelector("#timeMetric"),
  topRangeMetric: document.querySelector("#topRangeMetric"),
  topSpeedMetric: document.querySelector("#topSpeedMetric"),
  topDvMetric: document.querySelector("#topDvMetric"),
  topTimingMetric: document.querySelector("#topTimingMetric"),
  hudLine: document.querySelector("#hudLine"),
  coachHint: document.querySelector("#coachHint"),
  commandLine: document.querySelector("#commandLine"),
  footerLine: document.querySelector("#footerLine"),
  levelLabel: document.querySelector("#levelLabel"),
  speedMultiple: document.querySelector("#speedMultiple"),
  rMeter: document.querySelector("#rMeter"),
  iMeter: document.querySelector("#iMeter"),
  cMeter: document.querySelector("#cMeter"),
  sandboxSetupScreen: document.querySelector("#sandboxSetupScreen"),
  sandboxSetupForm: document.querySelector("#sandboxSetupForm"),
  sandboxSetupLevelTitle: document.querySelector("#sandboxSetupLevelTitle"),
  sandboxSetupStatus: document.querySelector("#sandboxSetupStatus"),
  sandboxSetupContinue: document.querySelector("#sandboxSetupContinue"),
  sandboxSetupCancel: document.querySelector("#sandboxSetupCancel"),
  sandboxSetupBack: document.querySelector("#sandboxSetupBack"),
  sandboxSetupNext: document.querySelector("#sandboxSetupNext"),
  sandboxSetupStepButtons: [...document.querySelectorAll("[data-sandbox-setup-step]")],
  sandboxSetupPanels: [...document.querySelectorAll("[data-sandbox-setup-panel]")],
  sandboxSetupInputs: [...document.querySelectorAll("[data-sandbox-field]")],
  operatorPanel: document.querySelector("#operatorPanel"),
  operatorBurnRows: document.querySelector("#operatorBurnRows"),
  operatorAddBurn: document.querySelector("#operatorAddBurn"),
  operatorStatus: document.querySelector("#operatorStatus"),
  operatorError: document.querySelector("#operatorError"),
  equationSheet: document.querySelector("#equationSheet"),
  equationSheetButton: document.querySelector("#equationSheetButton"),
  equationSheetClose: document.querySelector("#equationSheetClose"),
  debriefPanel: document.querySelector("#debriefPanel"),
  debriefTitle: document.querySelector("#debriefTitle"),
  debriefText: document.querySelector("#debriefText"),
  leaderboardForm: document.querySelector("#leaderboardForm"),
  leaderboardUsername: document.querySelector("#leaderboardUsername"),
  leaderboardEmail: document.querySelector("#leaderboardEmail"),
  leaderboardSubmit: document.querySelector("#leaderboardSubmit"),
  leaderboardStatus: document.querySelector("#leaderboardStatus"),
  downloadLink: document.querySelector("#downloadLink"),
  mobileSpeedButtons: Array.from(document.querySelectorAll("[data-mobile-speed]")),
};

const levelOptions = [
  {
    id: "tutorial",
    mode: "primer",
    title: PREVIEW_LEVEL_CONTRACTS.tutorial.title,
    operatorTitle: PREVIEW_LEVEL_CONTRACTS.tutorial.operator_title,
    budget: `Time: ${PREVIEW_LEVEL_CONTRACTS.tutorial.max_time_s}s   Chaser dV: ${formatSpeedMS(PREVIEW_LEVEL_CONTRACTS.tutorial.max_delta_v_m_s)}   Speed Gate: ${formatSpeedMS(PREVIEW_LEVEL_CONTRACTS.tutorial.max_goal_speed_km_s * 1000)}`,
    operatorBudget: `Time: ${PREVIEW_LEVEL_CONTRACTS.tutorial.max_time_s}s   Max burn: ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)}   Scripted playback`,
    objective: PREVIEW_LEVEL_CONTRACTS.tutorial.learning_goal,
    operatorObjective:
      "Learn the RIC frame primer, then script impulsive RIC burns and watch the HCW projection execute without live thrust controls.",
    brief: PREVIEW_LEVEL_CONTRACTS.tutorial.player_brief,
    operatorBrief:
      "The yellow satellite is you. R is radial, I is in-track, and C is cross-track. After the primer, enter burns by time and R/I/C delta-v, then launch the script.",
    criteria: PREVIEW_LEVEL_CONTRACTS.tutorial.pass_criteria,
    notes: PREVIEW_LEVEL_CONTRACTS.tutorial.instructor_notes,
    scope: PREVIEW_LEVEL_CONTRACTS.tutorial.scope,
    operatorCriteria: [
      "Complete the RIC frame primer.",
      `Script burns no larger than ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)} each.`,
      `Separate burn times by at least ${OPERATOR_BURN_SPACING_S} seconds.`,
      "Launch the script and compare the playback against the projected path.",
    ],
    operatorNotes: [
      "Operator mode is view-only during playback: plan first, then observe the natural response.",
      "The preview path is a circular HCW approximation for the browser version.",
    ],
  },
  {
    id: "sandbox",
    mode: "sandbox",
    title: PREVIEW_LEVEL_CONTRACTS.sandbox.title,
    operatorTitle: PREVIEW_LEVEL_CONTRACTS.sandbox.operator_title,
    budget: `Time: ${PREVIEW_LEVEL_CONTRACTS.sandbox.max_time_s}s`,
    operatorBudget: `Time: ${PREVIEW_LEVEL_CONTRACTS.sandbox.max_time_s}s   Max burn: ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)}   Scripted playback`,
    objective: "Experiment with RIC translation controls and relative orbital motion without pass/fail goals.",
    operatorObjective:
      "Script impulsive burns from a configurable starting RIC state, then watch the predicted and executed trajectory.",
    brief:
      "Complete the Sandbox Setup screen, then maneuver freely. Delta-v used remains visible, but there is no delta-v budget.",
    operatorBrief:
      "Complete the Sandbox Setup screen before planning, then build a time-ordered burn script and observe the result.",
    criteria: ["No pass/fail objective; experiment freely."],
    operatorCriteria: [
      "No pass/fail objective; experiment freely.",
      `Each burn must be ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)} or less.`,
      `Burns must be at least ${OPERATOR_BURN_SPACING_S} seconds apart.`,
    ],
    notes: [
      "Use this mode to demonstrate how the target orbit and initial relative state change relative motion.",
      "The browser setup uses the same target COE and chaser RIC field contract as the downloadable Sandbox.",
    ],
    operatorNotes: [
      "The ghost path updates as the script changes.",
      "Use Reset during playback to return to the script screen.",
    ],
    scope: PREVIEW_LEVEL_CONTRACTS.sandbox.scope,
  },
  {
    id: "pursuitArcade",
    mode: "arcade",
    title: PREVIEW_LEVEL_CONTRACTS.pursuit_arcade.title,
    budget: `Time: 12000s   Chaser dV: ${formatSpeedMS(3.0)}   Goal: ${formatDistanceKm(0.1)}`,
    objective: "Chase an evading target using RIC translation controls in a browser-native two-body arcade model.",
    brief:
      "The arcade mode propagates target and chaser in ECI under central Earth gravity, maps your RIC commands into the target frame, records inputs by simulation tick, and validates the replay locally.",
    criteria: [
      `Reach the ${formatDistanceKm(0.1)} goal circle.`,
      "Clear as many pursuit rounds as possible.",
      `Stay inside the ${formatSpeedMS(3.0)} chaser delta-v budget.`,
    ],
    notes: [
      "Web-only competition prototype: browser play uses a deterministic two-body engine, not the full downloadable OEL engine.",
      "Standalone and multi-round arcade attempts can be replay-validated locally; hosted leaderboard submissions are validated before scoring.",
      "Static RI and RC plots can be generated from recomputed replay history.",
    ],
    scope: PREVIEW_LEVEL_CONTRACTS.pursuit_arcade.scope,
  },
  {
    id: "rpoDuel",
    mode: "external",
    title: "RPO Duel — Beta",
    budget: "Rounds: 2, 4, or 6   Chaser dV: 15.000 m/s   Target dV: 5.000 m/s",
    objective: "Outfly a second player or the deterministic OEL computer in a server-authoritative browser RPO match, then reverse roles on the same initial geometry.",
    brief:
      "Create an invite-only room, join with a room code, or choose Play computer. Chaser and Target roles alternate between rounds.",
    criteria: [
      "Chaser: enter the 100 m capture region before time expires.",
      "Target: survive until time expires.",
      "Use the shared automatic 200x coast and 10x maneuver time rails.",
    ],
    notes: [
      "Beta player-versus-player and player-versus-computer mode: an authoritative hosted room owns physics, scoring, and reconnect state.",
      "A disconnected spacecraft is neutralized and coasts while the remaining player stays connected.",
      "This browser-native two-body duel is not a replacement for the downloadable trainer's full OEL engine.",
    ],
    scope: "Hosted player-versus-player or player-versus-computer Beta. Opens the standalone RPO Duel service in this tab.",
    externalUrl: RPO_DUEL_URL,
  },
];

const tutorialStages = [
  {
    id: "plusI",
    title: "Pulse +I",
    text: "Hold D for a short in-track pulse, then coast and watch the RI curve.",
    axis: "i",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "speed",
    title: "Speed 10x",
    text: "Tap the up arrow until the speed readout reaches 10x.",
    speedTarget: 10,
  },
  {
    id: "minusI",
    title: "Pulse -I",
    text: "Hold A for the matching backward in-track pulse.",
    axis: "i",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "plusR",
    title: "Pulse +R",
    text: "Hold W to push radially away from the target reference line.",
    axis: "r",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "minusR",
    title: "Pulse -R",
    text: "Hold S to create the opposite radial response.",
    axis: "r",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "plusC",
    title: "Pulse +C",
    text: "Hold the right arrow to leave the orbital plane.",
    axis: "c",
    sign: 1,
    targetDv: 0.25,
  },
  {
    id: "minusC",
    title: "Pulse -C",
    text: "Hold the left arrow to make the cross-track mirror path.",
    axis: "c",
    sign: -1,
    targetDv: 0.25,
  },
  {
    id: "final",
    title: "Final Approach",
    text: `Use small pulses and coast into the green ${formatDistanceKm(0.25)} circle below ${formatSpeedMS(0.3)}.`,
    final: true,
  },
];

const operatorTutorialStages = [
  { id: "plusInTrack", displayLabel: "+I Burn", axis: "i", sign: 1 },
  { id: "minusInTrack", displayLabel: "-I Burn", axis: "i", sign: -1 },
  { id: "plusRadial", displayLabel: "+R Burn", axis: "r", sign: 1 },
  { id: "minusRadial", displayLabel: "-R Burn", axis: "r", sign: -1 },
  { id: "plusCrossTrack", displayLabel: "+C Burn", axis: "c", sign: 1 },
  { id: "minusCrossTrack", displayLabel: "-C Burn", axis: "c", sign: -1 },
];

const primerStages = [
  {
    id: "radial",
    axis: "r",
    title: "Radial Axis",
    text: "Away from Earth through the target.",
    hint: "Higher or lower circular orbits map to up/down motion on R.",
    eciPlane: "rc",
    localPlane: "ri",
    localSubtitle: "Radial offset in RI",
    eciSubtitle: "Orbit radius changes",
  },
  {
    id: "inTrack",
    axis: "i",
    title: "In-Track Axis",
    text: "Along the target's direction of motion.",
    hint: "Ahead or behind on the same circular orbit maps to left/right motion on I.",
    eciPlane: "rc",
    localPlane: "ri",
    localSubtitle: "Phase offset in RI",
    eciSubtitle: "Same orbit, phase shift",
  },
  {
    id: "crossTrack",
    axis: "c",
    title: "Cross-Track Axis",
    text: "Out of the target's orbital plane.",
    hint: "Different orbital-plane slopes map to left/right motion on C.",
    eciPlane: "ri",
    localPlane: "rc",
    localSubtitle: "Plane offset in RC",
    eciSubtitle: "Inclination sweep",
  },
];

const presets = {
  behind: { r: 0, i: -1.2, c: 0, rd: 0, id: 0, cd: 0 },
  radial: { r: 1.0, i: -0.25, c: 0, rd: 0, id: 0, cd: 0 },
  crossTrack: { r: 0, i: -0.6, c: 0.9, rd: 0, id: 0, cd: 0 },
  close: { r: 0.12, i: -0.32, c: 0.04, rd: 0, id: 0.00005, cd: 0 },
};

const keys = new Set();
const touch = new Set();
const keyPulses = new Set();
const touchPulses = new Set();

const state = {
  mode: "selector",
  running: false,
  passed: false,
  speedIndex: 0,
  sim: makeState(presets.behind),
  trail: [],
  ghost: [],
  tutorialTargetPath: [],
  activeStage: 0,
  stageStart: null,
  stageDv: 0,
  closestKm: Infinity,
  finalReason: "",
  lastFrameMs: 0,
  stepAccumulatorS: 0,
  frameTiming: createFrameTimingState(),
  timingResetToken: 0,
  cameraRuleMode: "full_trajectory",
  primerStage: 0,
  primerTimeS: 0,
  selectedLevel: 0,
  playMode: normalizePlayMode(readLocalPreference(PLAY_MODE_KEY) || "pilot"),
  activePlayMode: "pilot",
  activeLevelId: "",
  sandboxSetup: { ...DEFAULT_SANDBOX_SETUP },
  sandboxPair: null,
  frameConvention: normalizeFrameConvention(
    new URLSearchParams(window.location.search).get("frame_convention")
      || readLocalPreference(FRAME_CONVENTION_KEY)
      || "oel_default",
  ),
  operatorBurnRows: [],
  operatorPlan: [],
  operatorPlanPath: [],
  operatorBurnMarkers: [],
  operatorPlanKey: "",
  operatorTrajectoryProbe: null,
  operatorBurnIndex: 0,
  operatorPlanError: "",
  operatorPanelSignature: "",
  operatorTutorialStage: 0,
  operatorTutorialStageStartS: 0,
  operatorBurnCinematicActive: false,
  operatorBurnCinematicHoldUntilMs: 0,
  operatorBurnAnimation: null,
  equationSheetVisible: false,
  musicEnabled: true,
  musicStartRequested: false,
  arcadeSession: null,
  arcadeSnapshot: null,
  arcadeValidation: null,
  arcadeAttemptPacket: null,
  arcadeSeed: 4242,
  arcadeTransition: null,
  arcadeReferenceStateEci: null,
  arcadeTargetRel: { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: 0 },
  arcadeChaserTargetRel: { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: 0 },
  leaderboardEntries: [],
  leaderboardLastFetchMs: 0,
  leaderboardLoading: false,
  targetTrail: [],
  targetGhost: [],
  viewPreference: "auto",
  activeView: "desktop",
};

const music = createMusicPlayer(MUSIC_TRACKS.selector);
music.loop = true;
music.volume = 0.65;

const SHELL_GAME_MODE_CLASSES = [
  "mode-arcade",
  "mode-sandbox",
  "mode-tutorial",
  "mode-operator",
  "mode-operator-script",
  "primer-mode",
];

const analytics = {
  enabled: false,
  plausibleEnabled: false,
  vercelEnabled: false,
  provider: "",
  domain: "",
  trackedOnce: new Set(),
};

function initAnalytics() {
  const provider = metaContent("oel-analytics-provider").toLowerCase();
  const domain = metaContent("oel-analytics-domain");
  const providers = new Set(
    provider
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
  analytics.provider = provider;
  analytics.domain = domain;
  if (!analyticsAllowedOnCurrentHost()) return;
  if (providers.has("plausible") && domain) initPlausibleAnalytics(domain);
  if (providers.has("vercel") && vercelAnalyticsAllowedOnCurrentHost()) initVercelAnalytics();
  analytics.enabled = analytics.plausibleEnabled || analytics.vercelEnabled;
}

function initPlausibleAnalytics(domain) {
  analytics.plausibleEnabled = true;
  window.plausible =
    window.plausible ||
    function plausibleQueue() {
      window.plausible.q = window.plausible.q || [];
      window.plausible.q.push(arguments);
    };
  if (document.querySelector(`script[src="${PLAUSIBLE_ANALYTICS_SCRIPT_SRC}"]`)) return;
  const script = document.createElement("script");
  script.defer = true;
  script.dataset.domain = domain;
  script.src = PLAUSIBLE_ANALYTICS_SCRIPT_SRC;
  document.head.appendChild(script);
}

function initVercelAnalytics() {
  analytics.vercelEnabled = true;
  window.va =
    window.va ||
    function vercelAnalyticsQueue() {
      window.vaq = window.vaq || [];
      window.vaq.push(arguments);
    };
  const scriptSrc = metaContent("oel-vercel-analytics-script") || VERCEL_ANALYTICS_SCRIPT_SRC;
  if (document.querySelector(`script[src="${scriptSrc}"]`)) return;
  const script = document.createElement("script");
  script.defer = true;
  script.src = scriptSrc;
  document.head.appendChild(script);
}

function metaContent(name) {
  return document.querySelector(`meta[name="${name}"]`)?.content.trim() || "";
}

function analyticsAllowedOnCurrentHost() {
  if (!["http:", "https:"].includes(window.location.protocol)) return false;
  return !ANALYTICS_LOCAL_HOSTNAMES.has(window.location.hostname);
}

function vercelAnalyticsAllowedOnCurrentHost() {
  const hostname = window.location.hostname;
  const hostRules = metaContent("oel-vercel-analytics-hosts")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (hostRules.length <= 0) return hostname.endsWith(".vercel.app");
  const normalizedHost = hostname.toLowerCase();
  return hostRules.some((rule) => {
    if (rule === "*") return true;
    if (rule.startsWith(".")) return normalizedHost.endsWith(rule);
    return normalizedHost === rule;
  });
}

function trackEvent(name, props = {}) {
  if (!analytics.enabled) return;
  const cleanProps = analyticsProps(props);
  if (analytics.plausibleEnabled && typeof window.plausible === "function") {
    window.plausible(name, { props: cleanProps });
  }
  if (analytics.vercelEnabled && typeof window.va === "function") {
    window.va("event", { name, data: cleanProps });
  }
}

function trackEventOnce(name, props = {}) {
  if (analytics.trackedOnce.has(name)) return;
  analytics.trackedOnce.add(name);
  trackEvent(name, props);
}

function initializeViewPreference() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("view");
  const saved = readLocalPreference("oelPreviewViewPreference");
  state.viewPreference = normalizeViewPreference(requested || saved || "auto");
  applyViewPreference();
}

function readLocalPreference(key) {
  try {
    return window.localStorage?.getItem(key) || "";
  } catch {
    return "";
  }
}

function writeLocalPreference(key, value) {
  try {
    window.localStorage?.setItem(key, value);
  } catch {
    // Private/restricted browser contexts can block localStorage.
  }
}

function normalizePlayMode(value) {
  return PLAY_MODES.includes(value) ? value : "pilot";
}

function normalizeFrameConvention(value) {
  return FRAME_CONVENTIONS.includes(value) ? value : "oel_default";
}

function levelSupportsOperator(option) {
  return option?.id === "tutorial" || option?.id === "sandbox";
}

function operatorPreviewAvailable(option = selectedLevelOption()) {
  return state.activeView === "desktop" && levelSupportsOperator(option);
}

function selectedLevelOption() {
  return levelOptions[state.selectedLevel] || levelOptions[0];
}

function selectorOperatorModeRequested() {
  return state.playMode === "operator" && state.activeView === "desktop";
}

function levelVisibleInSelector(option) {
  return !selectorOperatorModeRequested() || levelSupportsOperator(option);
}

function visibleLevelIndex(index, direction = 1) {
  const count = levelOptions.length;
  if (count <= 0) return 0;
  const step = direction >= 0 ? 1 : -1;
  let idx = ((Math.floor(index) % count) + count) % count;
  for (let checked = 0; checked < count; checked += 1) {
    if (levelVisibleInSelector(levelOptions[idx])) return idx;
    idx = (idx + step + count) % count;
  }
  return 0;
}

function selectedPlayModeFor(option = selectedLevelOption()) {
  if (!operatorPreviewAvailable(option)) return "pilot";
  return state.playMode;
}

function selectorPlayModeLabel() {
  if (state.activeView !== "desktop") return "Pilot Only";
  return state.playMode === "operator" ? "Operator Preview" : "Pilot Preview";
}

function operatorModeActive() {
  return state.activePlayMode === "operator" && (state.mode === "operatorTutorial" || state.mode === "operatorSandbox");
}

function operatorScriptModeActive() {
  return (
    state.activePlayMode === "operator" &&
    (state.mode === "operatorScriptSandbox" || state.mode === "operatorScriptTutorial")
  );
}

function sandboxExperienceActive(mode = state.mode) {
  return mode === "sandbox" || mode === "operatorSandbox" || mode === "operatorScriptSandbox";
}

function operatorExperienceActive() {
  return operatorModeActive() || operatorScriptModeActive();
}

function displayAxisSign(axis) {
  if (state.frameConvention === "space_force" && axis === "i") return -1;
  return 1;
}

function frameConventionLabel() {
  return state.frameConvention === "space_force" ? "Frame: Space Force" : "Frame: OEL";
}

function displayTitleForOption(option = selectedLevelOption()) {
  return selectedPlayModeFor(option) === "operator" && option.operatorTitle ? option.operatorTitle : option.title;
}

function modeSpecificField(option, field) {
  const operatorField = `operator${field[0].toUpperCase()}${field.slice(1)}`;
  return selectedPlayModeFor(option) === "operator" && option[operatorField] ? option[operatorField] : option[field];
}

function normalizeViewPreference(value) {
  return ["auto", "mobile", "desktop"].includes(value) ? value : "auto";
}

function detectedView() {
  const narrow = window.matchMedia("(max-width: 760px), (max-height: 620px)").matches;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  return narrow || coarse ? "mobile" : "desktop";
}

function applyViewPreference() {
  const activeView = state.viewPreference === "auto" ? detectedView() : state.viewPreference;
  state.activeView = activeView;
  document.body.classList.toggle("mobile-view", activeView === "mobile");
  document.body.classList.toggle("desktop-view", activeView === "desktop");
  el.shell.classList.toggle("mobile-view", activeView === "mobile");
  el.shell.classList.toggle("desktop-view", activeView === "desktop");
  if (activeView !== "desktop" && operatorExperienceActive()) {
    showLevelSelector({ track: true, source: "view_change" });
  }
  syncViewButtons();
  syncMusicButton();
  renderLevelSelector();
  updateDebugState();
  draw();
}

function syncViewButtons() {
  const viewLabel = state.viewPreference === "desktop" ? "Computer" : state.viewPreference[0].toUpperCase() + state.viewPreference.slice(1);
  const label = `View: ${viewLabel}`;
  if (el.viewButton) {
    const mobileCameraButton = state.activeView === "mobile";
    el.viewButton.textContent = mobileCameraButton ? "Toggle Camera" : label;
    el.viewButton.disabled =
      mobileCameraButton && state.mode !== "sandbox" && state.mode !== "operatorSandbox" && state.mode !== "arcade";
    el.viewButton.setAttribute(
      "aria-label",
      mobileCameraButton ? "Toggle camera framing." : `${label}. Active layout ${state.activeView}.`,
    );
  }
  if (el.selectorViewButton) {
    el.selectorViewButton.textContent = label;
    el.selectorViewButton.setAttribute("aria-label", `${label}. Active layout ${state.activeView}.`);
  }
}

function cycleViewPreference() {
  const next = state.viewPreference === "auto" ? "mobile" : state.viewPreference === "mobile" ? "desktop" : "auto";
  state.viewPreference = next;
  writeLocalPreference("oelPreviewViewPreference", next);
  const url = new URL(window.location.href);
  if (next === "auto") url.searchParams.delete("view");
  else url.searchParams.set("view", next);
  window.history.replaceState(null, "", url);
  applyViewPreference();
  trackEvent("view_toggle", { preference: next, active_view: state.activeView });
}

function toggleSelectorPlayMode() {
  if (state.activeView !== "desktop") return;
  state.playMode = state.playMode === "operator" ? "pilot" : "operator";
  state.selectedLevel = visibleLevelIndex(state.selectedLevel, -1);
  writeLocalPreference(PLAY_MODE_KEY, state.playMode);
  renderLevelSelector();
  updateDebugState();
  trackEvent("preview_play_mode_toggle", { play_mode: state.playMode });
}

function toggleFrameConvention() {
  state.frameConvention = state.frameConvention === "space_force" ? "oel_default" : "space_force";
  writeLocalPreference(FRAME_CONVENTION_KEY, state.frameConvention);
  const url = new URL(window.location.href);
  url.searchParams.set("frame_convention", state.frameConvention);
  window.history.replaceState(null, "", url);
  renderLevelSelector();
  syncTouchControlLabels();
  updateGhost();
  draw();
  trackEvent("frame_convention_toggle", { frame_convention: state.frameConvention });
}

function launchInitialLevelFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("level") || params.get("mode");
  if (!requested) return false;
  const normalized = requested.toLowerCase();
  const idx = levelOptions.findIndex(
    (option) => option.id.toLowerCase() === normalized || option.mode.toLowerCase() === normalized,
  );
  if (idx < 0) return false;
  selectLevel(idx);
  launchSelectedLevel("url");
  return true;
}

function analyticsProps(props) {
  const clean = {};
  Object.entries(props).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    clean[key] = String(value);
  });
  return clean;
}

function createMusicPlayer(src) {
  if (typeof Audio === "function") {
    return new Audio(src);
  }
  return {
    currentSrc: "",
    loop: false,
    paused: true,
    src: new URL(src, window.location.href).href,
    volume: 0,
    load() {},
    pause() {
      this.paused = true;
    },
    play() {
      this.paused = false;
      return Promise.resolve();
    },
  };
}

function makeState(seed) {
  return {
    r: seed.r,
    i: seed.i,
    c: seed.c,
    rd: seed.rd,
    id: seed.id,
    cd: seed.cd,
    t: 0,
    dv: 0,
  };
}

function currentControls() {
  if (operatorExperienceActive()) return { r: 0, i: 0, c: 0 };
  const r = axisValue("w", "s", "rPlus", "rMinus");
  const i = displayAxisSign("i") * axisValue("d", "a", "iPlus", "iMinus");
  const c = axisValue("arrowright", "arrowleft", "cPlus", "cMinus");
  const mag = Math.hypot(r, i, c);
  if (mag > 1) {
    return { r: r / mag, i: i / mag, c: c / mag };
  }
  return { r, i, c };
}

function axisValue(plusKey, minusKey, plusTouch, minusTouch) {
  const plus = keys.has(plusKey) || touch.has(plusTouch) || keyPulses.has(plusKey) || touchPulses.has(plusTouch);
  const minus = keys.has(minusKey) || touch.has(minusTouch) || keyPulses.has(minusKey) || touchPulses.has(minusTouch);
  return Number(plus) - Number(minus);
}

function hasPendingControlPulse() {
  return keyPulses.size > 0 || touchPulses.size > 0;
}

function clearControlPulses() {
  keyPulses.clear();
  touchPulses.clear();
}

function resetSimulationTiming() {
  resetFrameTimingState(state.frameTiming);
  state.stepAccumulatorS = 0;
  state.timingResetToken += 1;
}

function resetState(seed = presets.behind) {
  clearControlPulses();
  clearOperatorBurnCinematic();
  state.sim = makeState(seed);
  state.sandboxPair = sandboxExperienceActive() ? sandboxPairFromSeed(state.sim) : null;
  state.trail = [samplePoint()];
  state.targetTrail = [];
  state.targetGhost = [];
  state.stageStart = { ...state.sim };
  state.closestKm = rangeKm();
  state.stageDv = 0;
  state.passed = false;
  state.finalReason = "";
  resetSimulationTiming();
  state.operatorBurnIndex = 0;
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  updateGhost();
  draw();
}

function clearOperatorBurnCinematic() {
  state.operatorBurnCinematicActive = false;
  state.operatorBurnCinematicHoldUntilMs = 0;
  state.operatorBurnAnimation = null;
}

function showLevelSelector(options = {}) {
  const previousMode = state.mode;
  state.mode = "selector";
  state.running = false;
  state.passed = false;
  resetSimulationTiming();
  el.topTimingMetric.textContent = "";
  el.topTimingMetric.classList.remove("timing-warning");
  keys.clear();
  touch.clear();
  clearControlPulses();
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  el.shell.classList.add("selector-mode");
  el.shell.classList.remove("sandbox-setup-mode", ...SHELL_GAME_MODE_CLASSES);
  el.sandboxSetupScreen.classList.add("hidden");
  setMusicTrackForMode("selector");
  renderLevelSelector();
  syncMusicButton();
  updateDebugState();
  if (options.track && previousMode !== "selector") {
    trackEvent("level_select_return", { from: previousMode, source: options.source || "unknown" });
  }
}

function renderLevelSelector() {
  state.selectedLevel = visibleLevelIndex(state.selectedLevel, 1);
  document.querySelectorAll("[data-level-option]").forEach((button) => {
    const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
    const active = idx === state.selectedLevel;
    const visible = idx >= 0 && levelVisibleInSelector(levelOptions[idx]);
    button.hidden = !visible;
    button.classList.toggle("active", active);
    button.setAttribute("aria-hidden", String(!visible));
    button.setAttribute("aria-current", active ? "true" : "false");
  });
  const option = selectedLevelOption();
  const selectorPlayMode = selectedPlayModeFor(option);
  const selectorOperatorMode = selectorPlayMode === "operator";
  el.shell.classList.toggle("selector-operator-mode", selectorOperatorMode);
  el.shell.classList.toggle("selector-pilot-mode", !selectorOperatorMode);
  el.levelSelector.classList.toggle("operator-mode", selectorOperatorMode);
  el.levelSelector.classList.toggle("pilot-mode", !selectorOperatorMode);
  if (el.selectorModeButton) {
    const availableOperator = state.activeView === "desktop";
    el.selectorModeButton.textContent = selectorPlayModeLabel();
    el.selectorModeButton.disabled = !availableOperator;
    el.selectorModeButton.classList.toggle("active", availableOperator && state.playMode === "operator");
    el.selectorModeButton.setAttribute("aria-pressed", String(availableOperator && state.playMode === "operator"));
  }
  el.selectorFrameButtons.forEach((button) => {
    button.textContent = frameConventionLabel();
    button.classList.toggle("active", state.frameConvention === "space_force");
    button.setAttribute("aria-pressed", String(state.frameConvention === "space_force"));
  });
  if (el.selectorPlayButton) {
    const externalUnavailable = option.mode === "external" && !option.externalUrl;
    el.selectorPlayButton.textContent = option.mode === "external" ? "Open Beta" : "Play Level";
    el.selectorPlayButton.disabled = externalUnavailable;
    el.selectorPlayButton.setAttribute(
      "aria-label",
      externalUnavailable ? "RPO Duel Beta hosting is not configured." : `Play ${displayTitleForOption(option)}.`,
    );
  }
  el.selectorPreviewTitle.textContent = displayTitleForOption(option);
  el.selectorPreviewBudget.textContent = modeSpecificField(option, "budget");
  el.selectorPreviewScope.textContent = modeSpecificField(option, "scope");
  el.selectorPreviewObjective.textContent = modeSpecificField(option, "objective");
  el.selectorPreviewBrief.textContent = modeSpecificField(option, "brief");
  replaceList(el.selectorPreviewCriteria, modeSpecificField(option, "criteria"));
  replaceList(el.selectorPreviewNotes, modeSpecificField(option, "notes"));
  const showLeaderboard = option.id === "pursuitArcade";
  el.leaderboardPanel.classList.toggle("hidden", !showLeaderboard);
  if (showLeaderboard) refreshLeaderboard();
}

function replaceList(listEl, items) {
  listEl.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function refreshLeaderboard(options = {}) {
  if (!el.leaderboardPanel || state.leaderboardLoading) return;
  const force = Boolean(options.force);
  const now = Date.now();
  if (!force && state.leaderboardLastFetchMs && now - state.leaderboardLastFetchMs < LEADERBOARD_REFRESH_MS) {
    renderLeaderboard();
    return;
  }
  if (!["http:", "https:"].includes(window.location.protocol)) {
    state.leaderboardEntries = [];
    renderLeaderboard("Leaderboard appears on the hosted site.");
    return;
  }
  state.leaderboardLoading = true;
  renderLeaderboard("Loading scores...");
  fetch(`/api/leaderboard?challenge=${encodeURIComponent(ARCADE_CHALLENGE_RECORD.challenge_id)}&limit=10`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      state.leaderboardEntries = Array.isArray(payload.entries) ? payload.entries : [];
      state.leaderboardLastFetchMs = Date.now();
      renderLeaderboard();
      trackEvent("arcade_leaderboard_view", { entries: state.leaderboardEntries.length });
    })
    .catch((error) => {
      renderLeaderboard(`Leaderboard unavailable: ${error instanceof Error ? error.message : String(error)}`);
    })
    .finally(() => {
      state.leaderboardLoading = false;
    });
}

function renderLeaderboard(message = "") {
  if (!el.leaderboardList || !el.leaderboardMeta) return;
  el.leaderboardList.replaceChildren();
  if (state.leaderboardEntries.length <= 0) {
    const li = document.createElement("li");
    li.className = "leaderboard-empty";
    li.textContent = "No scores yet.";
    el.leaderboardList.appendChild(li);
  } else {
    state.leaderboardEntries.forEach((entry, idx) => {
      const li = document.createElement("li");
      li.className = "leaderboard-row";
      const rank = document.createElement("span");
      rank.className = "leaderboard-rank";
      rank.textContent = String(idx + 1).padStart(2, "0");
      const name = document.createElement("span");
      name.className = "leaderboard-name";
      name.textContent = String(entry.username || "anonymous");
      const score = document.createElement("span");
      score.className = "leaderboard-score";
      score.textContent = Number(entry.score || 0).toLocaleString();
      const rounds = document.createElement("span");
      rounds.className = "leaderboard-rounds";
      rounds.textContent = `${Number(entry.metrics?.rounds_cleared || 0)} rd`;
      li.append(rank, name, score, rounds);
      el.leaderboardList.appendChild(li);
    });
  }
  if (message) {
    el.leaderboardMeta.textContent = message;
  } else if (state.leaderboardLastFetchMs) {
    el.leaderboardMeta.textContent = `Updated ${new Date(state.leaderboardLastFetchMs).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  } else {
    el.leaderboardMeta.textContent = "";
  }
}

function selectLevel(index) {
  const direction = index >= state.selectedLevel ? 1 : -1;
  state.selectedLevel = visibleLevelIndex(index, direction);
  renderLevelSelector();
  updateDebugState();
}

function launchSelectedLevel(source = "selector") {
  const option = selectedLevelOption();
  if (option.mode === "external") {
    if (!option.externalUrl) return;
    const destination = externalLevelUrl(option);
    trackEvent("rpo_duel_open", {
      source,
      destination: destination.hostname,
      frame_convention: state.frameConvention,
    });
    window.location.assign(destination.href);
    return;
  }
  state.activePlayMode = selectedPlayModeFor(option);
  state.activeLevelId = option.id;
  if (state.activePlayMode === "operator" && (option.id === "tutorial" || option.id === "sandbox")) {
    state.operatorTutorialStage = 0;
    state.operatorTutorialStageStartS = 0;
    state.operatorBurnRows = [];
    state.operatorPlanPath = [];
    state.operatorBurnMarkers = [];
    state.operatorPlanKey = "";
    state.operatorTrajectoryProbe = null;
    state.operatorPanelSignature = "";
  }
  if (option.id === "sandbox") {
    showSandboxSetup({ source });
    playMusicFromGesture();
    return;
  }
  const mode =
    state.activePlayMode === "operator" && option.id === "sandbox"
      ? "operatorScriptSandbox"
      : state.activePlayMode === "operator" && option.id === "tutorial"
        ? "primer"
        : option.mode;
  setMode(mode);
  if (option.id === "pursuitArcade") {
    trackEvent("arcade_start", { source });
  } else {
    trackEvent("tutorial_start", { source, entry: option.mode, play_mode: state.activePlayMode });
  }
  playMusicFromGesture();
}

function showSandboxSetup(options = {}) {
  state.mode = "sandboxSetup";
  state.running = false;
  state.passed = false;
  resetSimulationTiming();
  keys.clear();
  touch.clear();
  clearControlPulses();
  el.shell.classList.remove("selector-mode", ...SHELL_GAME_MODE_CLASSES);
  el.shell.classList.add("sandbox-setup-mode");
  el.sandboxSetupScreen.classList.remove("hidden");
  el.sandboxSetupLevelTitle.textContent =
    state.activePlayMode === "operator" ? "Operator Sandbox" : "Pilot Sandbox";
  const values = sandboxSetupInputValues(state.sandboxSetup);
  el.sandboxSetupInputs.forEach((input) => {
    input.value = values[input.dataset.sandboxField] ?? "";
    input.removeAttribute("aria-invalid");
  });
  setSandboxSetupStep("target");
  syncSandboxSetupValidation();
  setMusicTrackForMode("sandbox");
  trackEvent("sandbox_setup_open", {
    source: options.source || "selector",
    play_mode: state.activePlayMode,
  });
  updateDebugState();
  requestAnimationFrame(() => el.sandboxSetupInputs[0]?.focus());
}

function sandboxPortraitSetupActive() {
  return (
    document.body.classList.contains("mobile-view") &&
    window.matchMedia("(orientation: portrait) and (max-width: 760px)").matches
  );
}

function setSandboxSetupStep(step, options = {}) {
  const selectedStep = step === "chaser" ? "chaser" : "target";
  el.sandboxSetupScreen.dataset.sandboxStep = selectedStep;
  el.sandboxSetupStepButtons.forEach((button) => {
    if (button.dataset.sandboxSetupStep === selectedStep) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  if (options.focus) {
    requestAnimationFrame(() => {
      el.sandboxSetupPanels
        .find((panel) => panel.dataset.sandboxSetupPanel === selectedStep)
        ?.querySelector("input")
        ?.focus();
    });
  }
}

function sandboxSetupInstruction() {
  if (!sandboxPortraitSetupActive()) {
    return "Tab changes fields. Continue applies this setup before simulation or planning.";
  }
  return el.sandboxSetupScreen.dataset.sandboxStep === "chaser"
    ? "Step 2 of 2. Set the chaser RIC state."
    : "Step 1 of 2. Set the target orbit.";
}

function readSandboxSetupForm() {
  const rawValues = Object.fromEntries(
    el.sandboxSetupInputs.map((input) => [input.dataset.sandboxField, input.value]),
  );
  return validateSandboxSetup(rawValues);
}

function syncSandboxSetupValidation() {
  const result = readSandboxSetupForm();
  const message = result.error || sandboxSetupInstruction();
  el.sandboxSetupStatus.textContent = message;
  el.sandboxSetupStatus.dataset.error = String(Boolean(result.error));
  el.sandboxSetupContinue.disabled = Boolean(result.error);
  el.sandboxSetupInputs.forEach((input) => input.removeAttribute("aria-invalid"));
  if (result.error) {
    const invalidField = SANDBOX_SETUP_FIELDS.find((field) => result.error.includes(field.label));
    el.sandboxSetupInputs
      .find((input) => input.dataset.sandboxField === invalidField?.key)
      ?.setAttribute("aria-invalid", "true");
  }
  return result;
}

function advanceSandboxSetupStep() {
  const result = syncSandboxSetupValidation();
  const invalidInput = el.sandboxSetupInputs.find((input) => input.getAttribute("aria-invalid") === "true");
  if (result.error && invalidInput?.closest('[data-sandbox-setup-panel="target"]')) {
    invalidInput.focus();
    return;
  }
  setSandboxSetupStep("chaser", { focus: true });
  syncSandboxSetupValidation();
}

function continueSandboxSetup() {
  const result = syncSandboxSetupValidation();
  if (!result.value) return;
  state.sandboxSetup = result.value;
  el.sandboxSetupScreen.classList.add("hidden");
  el.shell.classList.remove("sandbox-setup-mode");
  const mode = state.activePlayMode === "operator" ? "operatorScriptSandbox" : "sandbox";
  setMode(mode);
  trackEvent("sandbox_start", {
    source: "sandbox_setup_continue",
    play_mode: state.activePlayMode,
    target_ecc: state.sandboxSetup.target_ecc,
    target_a_km: state.sandboxSetup.target_a_km,
  });
}

function externalLevelUrl(option) {
  const destination = new URL(option.externalUrl, window.location.href);
  if (option.id === "rpoDuel") destination.searchParams.set("frame_convention", state.frameConvention);
  return destination;
}

function setMode(mode) {
  state.mode = mode;
  el.shell.classList.remove("selector-mode", "sandbox-setup-mode");
  el.sandboxSetupScreen.classList.add("hidden");
  state.running = false;
  state.speedIndex = 0;
  state.cameraRuleMode =
    mode === "sandbox" || mode === "operatorSandbox" || operatorScriptModeActive() || mode === "arcade"
      ? "full_trajectory"
      : "default";
  setMusicTrackForMode(initialMusicTrackForMode(mode));
  state.activeStage = 0;
  state.stageStart = null;
  state.stageDv = 0;
  state.passed = false;
  state.primerTimeS = 0;
  if (mode === "primer") state.primerStage = 0;
  if (mode === "arcade") {
    startArcadeSession();
  } else {
    state.arcadeSession = null;
    state.arcadeSnapshot = null;
    state.arcadeValidation = null;
    const seed =
      mode === "sandbox" || mode === "operatorSandbox" || mode === "operatorScriptSandbox"
        ? sandboxSeed()
        : mode === "operatorScriptTutorial" || mode === "operatorTutorial"
          ? presets.behind
        : mode === "primer"
          ? primerSample()
          : presets.behind;
    resetState(seed);
    if (operatorExperienceActive()) {
      ensureOperatorRows(mode);
      updateOperatorPlan();
    }
  }
  updateMissionText();
}

function launchOperatorPlayback() {
  updateOperatorPlan();
  if (state.operatorPlanError) {
    renderOperatorPlanStatus();
    return;
  }
  const nextMode = state.mode === "operatorScriptTutorial" ? "operatorTutorial" : "operatorSandbox";
  setMode(nextMode);
  state.running = true;
  if (nextMode === "operatorTutorial") {
    state.speedIndex = speedOptionIndex(OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE);
    state.operatorTutorialStageStartS = state.sim.t;
  }
  updateMissionText();
}

function activePrimerStage() {
  return primerStages[state.primerStage] || primerStages[primerStages.length - 1];
}

function primerSample() {
  const stage = activePrimerStage();
  const phase = Math.sin(state.primerTimeS * 1.05);
  const sample = { r: 0, i: 0, c: 0, rd: 0, id: 0, cd: 0, t: state.primerTimeS };
  sample[stage.axis] = (PRIMER_AMPLITUDES_KM[stage.axis] || 0.6) * phase;
  return sample;
}

function advancePrimer() {
  if (state.mode !== "primer") return false;
  if (state.primerStage < primerStages.length - 1) {
    state.primerStage += 1;
    state.primerTimeS = 0;
    updateMissionText();
    updateGhost();
    draw();
    return true;
  }
  trackEvent("primer_complete", { step_count: primerStages.length });
  setMode(state.activePlayMode === "operator" ? "operatorScriptTutorial" : "tutorial");
  return true;
}

function sandboxSeed() {
  return sandboxRelativeSeed(state.sandboxSetup);
}

function sandboxPairFromSeed(seed) {
  const timeS = Math.max(Number(seed?.t || 0), 0);
  const target = sandboxTargetStateAt(state.sandboxSetup, timeS, MU);
  const chaser = stateFromRelativeRic(target, {
    r_km: Number(seed?.r || 0),
    i_km: Number(seed?.i || 0),
    c_km: Number(seed?.c || 0),
    rd_km_s: Number(seed?.rd || 0),
    id_km_s: Number(seed?.id || 0),
    cd_km_s: Number(seed?.cd || 0),
  });
  return { target, chaser, timeS };
}

function stepSandboxPair(pair, durationS, chaserAccelRic = [0, 0, 0]) {
  let next = { target: pair.target, chaser: pair.chaser };
  let remainingS = Math.max(Number(durationS || 0), 0);
  while (remainingS > 1.0e-9) {
    const dtS = Math.min(remainingS, 10);
    next = stepControlledTwoBodyPair({
      target: next.target,
      chaser: next.chaser,
      mu_km3_s2: MU,
      chaser_accel_ric_km_s2: chaserAccelRic,
      dt_s: dtS,
    });
    remainingS -= dtS;
  }
  return { ...next, timeS: Number(pair.timeS || 0) + Math.max(Number(durationS || 0), 0) };
}

function sandboxPairRelativeState(pair, previous = {}) {
  const rel = relativeRicState(pair.target, pair.chaser);
  return {
    r: rel.r_km,
    i: rel.i_km,
    c: rel.c_km,
    rd: rel.rd_km_s,
    id: rel.id_km_s,
    cd: rel.cd_km_s,
    t: Number(pair.timeS || 0),
    dv: Number(previous.dv || 0),
  };
}

function sandboxCoastPoint(seed, durationS) {
  const pair = sandboxPairFromSeed(seed);
  const nextPair = stepSandboxPair(pair, durationS);
  return sandboxPairRelativeState(nextPair, seed);
}

function sandboxEllipticProjectionActive() {
  return sandboxProjectionModel(state.sandboxSetup) === "tschauner_hempel";
}

function sandboxProjectionCoastPoint(seed, durationS) {
  if (!sandboxEllipticProjectionActive()) return sandboxCoastPoint(seed, durationS);
  return sandboxEllipticLinearCoastStates(state.sandboxSetup, seed, [durationS], MU)[0] || { ...seed };
}

function currentOrbitPeriodS() {
  return sandboxExperienceActive() ? sandboxOrbitPeriodS(state.sandboxSetup, MU) : ORBIT_PERIOD_S;
}

function startArcadeSession() {
  state.arcadeSeed = (state.arcadeSeed + 101) % 4294967296;
  state.arcadeSession = createPursuitArcadeSession(ARCADE_CHALLENGE_RECORD.config, {
    seed: state.arcadeSeed,
    ...arcadeDevStartOptions(),
  });
  state.arcadeValidation = null;
  state.arcadeAttemptPacket = null;
  state.arcadeTransition = null;
  state.passed = false;
  state.finalReason = "";
  resetSimulationTiming();
  el.debriefPanel.classList.add("hidden");
  setLeaderboardFormVisible(false);
  syncArcadeSnapshot();
}

function arcadeDevStartOptions() {
  if (!previewDevParamsAllowed()) return {};
  const params = new URLSearchParams(window.location.search);
  const firstBossRound = Math.max(Math.floor(Number(ARCADE_CHALLENGE_RECORD.config.arcade?.boss_round_interval || 0)), 1);
  const requestedRound = params.has("boss") ? firstBossRound : Math.floor(Number(params.get("round") || 1));
  if (!Number.isFinite(requestedRound) || requestedRound <= 1) return {};
  return { startRoundIndex: requestedRound };
}

function previewDevParamsAllowed() {
  return window.location.protocol === "file:" || PREVIEW_DEV_HOSTNAMES.has(window.location.hostname);
}

function syncArcadeSnapshot() {
  if (!state.arcadeSession) return;
  state.arcadeSnapshot = state.arcadeSession.snapshot();
  state.arcadeTransition = state.arcadeSnapshot.round_transition || null;
  state.arcadeReferenceStateEci = arcadeEciBlockToState(state.arcadeSnapshot.target_reference_state_eci);
  state.arcadeChaserTargetRel = arcadeRicBlockToSim(state.arcadeSnapshot.relative_ric, state.arcadeSnapshot.time_s);
  state.arcadeTargetRel = arcadeRicBlockToSim(
    lastArcadeHistoryBlock(state.arcadeSnapshot.history, "target_reference_ric"),
    state.arcadeSnapshot.time_s,
  );
  const chaserReference = lastArcadeHistoryBlock(state.arcadeSnapshot.history, "chaser_reference_ric");
  state.sim = arcadeRicBlockToSim(chaserReference || state.arcadeSnapshot.relative_ric, state.arcadeSnapshot.time_s);
  state.closestKm = Number(state.arcadeSnapshot.closest_range_km || state.arcadeSnapshot.range_km || Infinity);
  state.trail = arcadeHistoryToTrail(state.arcadeSnapshot.history || [], "chaser_reference_ric");
  state.targetTrail = arcadeHistoryToTrail(state.arcadeSnapshot.history || [], "target_reference_ric");
  state.passed = Boolean(state.arcadeSnapshot.terminal);
  if (state.arcadeTransition) {
    state.running = false;
  }
  if (state.arcadeSnapshot.terminal) {
    state.running = false;
    state.finalReason = state.arcadeSnapshot.terminal_reason || "";
  }
  setMusicTrackForMode(arcadeMusicTrackKey());
}

function lastArcadeHistoryBlock(history, key) {
  if (!Array.isArray(history) || history.length === 0) return null;
  return history[history.length - 1]?.[key] || null;
}

function arcadeRicBlockToSim(rel, timeS = 0) {
  rel = rel || {};
  return {
    r: Number(rel.r_km || 0),
    i: Number(rel.i_km || 0),
    c: Number(rel.c_km || 0),
    rd: Number(rel.rd_km_s || 0),
    id: Number(rel.id_km_s || 0),
    cd: Number(rel.cd_km_s || 0),
    t: Number(timeS || 0),
    dv: Number(state.arcadeSnapshot?.player_delta_v_m_s || 0),
  };
}

function arcadeEciBlockToState(block) {
  if (!block || !Array.isArray(block.r_km) || !Array.isArray(block.v_km_s)) return null;
  return {
    r: block.r_km.map(Number),
    v: block.v_km_s.map(Number),
  };
}

function arcadeHistoryToTrail(history, key = "relative_ric") {
  return history.slice(-TRAIL_LIMIT).map((sample) => {
    const rel = sample[key] || sample.relative_ric || {};
    return {
      r: Number(rel.r_km || 0),
      i: Number(rel.i_km || 0),
      c: Number(rel.c_km || 0),
      rd: Number(rel.rd_km_s || 0),
      id: Number(rel.id_km_s || 0),
      cd: Number(rel.cd_km_s || 0),
      t: Number(sample.time_s || 0),
    };
  });
}

function arcadeStep(ticks = 1) {
  if (!state.arcadeSession || !state.running || state.passed) return;
  state.arcadeSession.setControls(currentControls());
  state.arcadeSession.step(Math.max(1, Math.floor(Number(ticks) || 1)));
  syncArcadeSnapshot();
  if (state.arcadeTransition) {
    showArcadeRoundTransition();
  } else if (state.arcadeSnapshot?.terminal) {
    showArcadeDebrief();
  }
}

function step(dt, forceRun = false) {
  if (state.mode === "arcade") {
    const baseDtS = ARCADE_CHALLENGE_RECORD.config.dt_s;
    const tickCount = Math.max(1, Math.round(Number(dt || baseDtS) / Math.max(Number(baseDtS || 1), 1.0e-9)));
    arcadeStep(tickCount);
    return;
  }
  if (operatorModeActive()) {
    stepOperator(dt, forceRun);
    return;
  }
  if ((!state.running && !forceRun) || state.passed) return;
  const u = currentControls();
  if (state.mode === "sandbox") {
    stepSandboxLive(dt, u);
  } else {
    stepHcwStateInPlace(state.sim, u, dt);
  }
  state.sim.dv += Math.hypot(u.r, u.i, u.c) * MAX_ACCEL_KM_S2 * dt * 1000;
  state.closestKm = Math.min(state.closestKm, rangeKm());
  state.trail.push(samplePoint());
  if (state.trail.length > TRAIL_LIMIT) state.trail.shift();
  updateTutorial(dt, u);
}

function stepSandboxLive(dtS, controls) {
  if (!state.sandboxPair || Math.abs(Number(state.sandboxPair.timeS || 0) - Number(state.sim.t || 0)) > 1.0e-6) {
    state.sandboxPair = sandboxPairFromSeed(state.sim);
  }
  const accelRic = [controls.r, controls.i, controls.c].map((value) => Number(value || 0) * MAX_ACCEL_KM_S2);
  state.sandboxPair = stepSandboxPair(state.sandboxPair, dtS, accelRic);
  state.sim = sandboxPairRelativeState(state.sandboxPair, state.sim);
}

function operatorBurnVisualDurationS(deltaVMps) {
  const magnitude = Number.isFinite(Number(deltaVMps)) ? Math.max(Number(deltaVMps), 0) : 0;
  const duration = OPERATOR_BURN_VISUAL_DURATION_BASE_S + OPERATOR_BURN_VISUAL_DURATION_PER_M_S * magnitude;
  return Math.min(Math.max(duration, OPERATOR_BURN_VISUAL_DURATION_MIN_S), OPERATOR_BURN_VISUAL_DURATION_MAX_S);
}

function updateOperatorBurnCinematic(nowMs, frameHorizonS = 0) {
  if (!operatorModeActive()) {
    clearOperatorBurnCinematic();
    return;
  }
  if (
    state.operatorBurnCinematicActive &&
    state.operatorBurnCinematicHoldUntilMs > 0 &&
    nowMs > state.operatorBurnCinematicHoldUntilMs
  ) {
    clearOperatorBurnCinematic();
  }
  if (state.operatorBurnCinematicActive) return;
  const nextBurn = state.operatorPlan[state.operatorBurnIndex];
  if (!nextBurn) return;
  const timeToBurnS = Number(nextBurn.timeS || 0) - Number(state.sim.t || 0);
  const triggerWindowS = Math.max(OPERATOR_BURN_CINEMATIC_LOOKAHEAD_S, Number(frameHorizonS || 0));
  if (timeToBurnS >= -1.0e-9 && timeToBurnS <= triggerWindowS + 1.0e-9) {
    state.operatorBurnCinematicActive = true;
    state.operatorBurnCinematicHoldUntilMs = 0;
  }
}

function beginOperatorBurnAnimation(preBurnState, postBurnState, burn, nowMs) {
  const durationS = operatorBurnVisualDurationS(burn?.dvMps || 0);
  state.operatorBurnAnimation = {
    pre: { ...preBurnState },
    post: { ...postBurnState },
    startMs: Number(nowMs || performance.now()),
    durationMs: durationS * 1000,
  };
  state.operatorBurnCinematicActive = true;
  state.operatorBurnCinematicHoldUntilMs = Number(nowMs || performance.now()) + durationS * 1000;
}

function operatorBurnProjectionSeed(nowMs = performance.now()) {
  const animation = state.operatorBurnAnimation;
  if (!animation) return { ...state.sim };
  const durationMs = Math.max(Number(animation.durationMs || 0), 1);
  const alpha = Math.min(Math.max((Number(nowMs) - Number(animation.startMs || 0)) / durationMs, 0), 1);
  if (alpha >= 1 && (!state.operatorBurnCinematicActive || state.operatorBurnCinematicHoldUntilMs <= 0)) {
    state.operatorBurnAnimation = null;
    return { ...state.sim };
  }
  const blended = {};
  ["r", "i", "c", "rd", "id", "cd"].forEach((key) => {
    blended[key] = Number(animation.pre[key] || 0) + (Number(animation.post[key] || 0) - Number(animation.pre[key] || 0)) * alpha;
  });
  blended.t = Number(state.sim.t || 0);
  blended.dv = Number(state.sim.dv || 0);
  return blended;
}

function stepOperator(dt, forceRun = false) {
  if ((!state.running && !forceRun) || state.passed) return;
  const targetTimeS = state.sim.t + dt;
  const nowMs = performance.now();
  while (targetTimeS - state.sim.t > 1.0e-9) {
    const nextBurn = state.operatorPlan[state.operatorBurnIndex];
    const nextStopS = nextBurn ? Math.min(targetTimeS, nextBurn.timeS) : targetTimeS;
    const coastDt = Math.max(nextStopS - state.sim.t, 0);
    if (coastDt > 1.0e-9) {
      const nextState = coastPoint(state.sim, coastDt);
      state.sim.r = nextState.r;
      state.sim.i = nextState.i;
      state.sim.c = nextState.c;
      state.sim.rd = nextState.rd;
      state.sim.id = nextState.id;
      state.sim.cd = nextState.cd;
      state.sim.t = nextState.t;
    }
    if (nextBurn && Math.abs(state.sim.t - nextBurn.timeS) <= 1.0e-6) {
      const preBurnState = { ...state.sim };
      state.sim.rd += nextBurn.rMps / 1000;
      state.sim.id += nextBurn.iMps / 1000;
      state.sim.cd += nextBurn.cMps / 1000;
      state.sim.dv += nextBurn.dvMps;
      beginOperatorBurnAnimation(preBurnState, { ...state.sim }, nextBurn, nowMs);
      state.operatorBurnIndex += 1;
    } else {
      break;
    }
  }
  state.closestKm = Math.min(state.closestKm, rangeKm());
  state.trail.push(samplePoint());
  if (state.trail.length > TRAIL_LIMIT) state.trail.shift();
  if (state.mode === "operatorTutorial") {
    const stageStartS = Number(state.operatorTutorialStageStartS || 0);
    if (state.sim.t - stageStartS >= OPERATOR_TUTORIAL_STAGE_DURATION_S) {
      state.operatorTutorialStage += 1;
      state.operatorTutorialStageStartS = 0;
      state.running = false;
      if (state.operatorTutorialStage >= operatorTutorialStages.length) {
        state.passed = true;
        state.finalReason = "Operator tutorial complete.";
        showDebrief(true);
      } else {
        state.operatorBurnRows = [];
        state.operatorPanelSignature = "";
        setMode("operatorScriptTutorial");
      }
    }
  }
}

function updateTutorial(dt, u) {
  if (state.mode !== "tutorial" || state.passed) return;
  const stage = tutorialStages[state.activeStage];
  if (!stage) return;
  if (stage.speedTarget) {
    maybeCompleteSpeedStage(stage);
    return;
  }
  if (stage.final) {
    const slowEnough = relativeSpeedKmS() <= 0.0003;
    if (rangeKm() <= 0.25 && slowEnough) {
      state.passed = true;
      state.running = false;
      state.finalReason = "Goal reached under the speed limit.";
      showDebrief(true);
    }
    return;
  }
  const value = u[stage.axis] || 0;
  if (Math.sign(value) === stage.sign) {
    state.stageDv += Math.abs(value) * MAX_ACCEL_KM_S2 * dt * 1000;
  }
  if (state.stageDv >= stage.targetDv) {
    completeStage();
  }
}

function maybeCompleteSpeedStage(stage = activeTutorialStage()) {
  if (!stage || !stage.speedTarget) return false;
  if (SPEED_OPTIONS[state.speedIndex] < stage.speedTarget) return false;
  completeStage();
  return true;
}

function activeTutorialStage() {
  if (state.mode !== "tutorial") return null;
  return tutorialStages[state.activeStage] || null;
}

function tutorialInputMatches(stage = activeTutorialStage(), controls = currentControls()) {
  if (!stage || !stage.axis || stage.final || stage.speedTarget) return false;
  const value = controls[stage.axis] || 0;
  return Math.sign(value) === stage.sign && Math.abs(value) > 0.5;
}

function simulationShouldRun() {
  if (state.mode === "primer") return false;
  if (operatorScriptModeActive()) return false;
  if (operatorModeActive()) return state.running;
  if (state.mode === "arcade") return state.running;
  return state.running || tutorialInputMatches();
}

function currentStepDtS(controls = currentControls()) {
  const baseDtS = state.mode === "arcade" ? ARCADE_CHALLENGE_RECORD.config.dt_s : FIXED_DT_S;
  return gameTickDtS({ baseDtS, speedMultiple: effectiveSpeedMultiple(controls) });
}

const sandbox3d = createSandbox3D({
  getState: () => state, fitCanvas, drawMarker: drawSpacecraftMarker, drawPath,
  onChange: () => draw(),
});

function currentSpeedMultiple() {
  return SPEED_OPTIONS[state.speedIndex];
}

function speedOptionIndex(value) {
  let bestIdx = 0;
  let bestError = Infinity;
  SPEED_OPTIONS.forEach((option, idx) => {
    const error = Math.abs(option - value);
    if (error < bestError) {
      bestIdx = idx;
      bestError = error;
    }
  });
  return bestIdx;
}

function hasManeuverInput(controls = currentControls()) {
  return Math.hypot(controls.r, controls.i, controls.c) > 1.0e-9;
}

function operatorBurnCinematicShouldClamp() {
  return operatorModeActive() && state.operatorBurnCinematicActive;
}

function effectiveSpeedMultiple(controls = currentControls()) {
  if (sandbox3d.interacting()) return Math.min(currentSpeedMultiple(), 1);
  if (operatorBurnCinematicShouldClamp()) {
    return Math.min(currentSpeedMultiple(), OPERATOR_BURN_CINEMATIC_SPEED_MULTIPLE);
  }
  if (currentSpeedMultiple() <= MANEUVER_CONTROL_SPEED || !hasManeuverInput(controls)) return currentSpeedMultiple();
  return MANEUVER_CONTROL_SPEED;
}

function speedBadgeText(controls = currentControls()) {
  const selected = currentSpeedMultiple();
  const effective = effectiveSpeedMultiple(controls);
  return effective === selected ? `${selected}x` : `${effective}x ${sandbox3d.interacting() ? "camera" : "burn"}`;
}

function speedFooterText(controls = currentControls()) {
  const selected = currentSpeedMultiple();
  const effective = effectiveSpeedMultiple(controls);
  return effective === selected ? `${selected.toFixed(0)}x` : `${effective.toFixed(0)}x ${sandbox3d.interacting() ? "Camera" : "Burn"} (${selected.toFixed(0)}x Coast)`;
}

function timingSecondsText(value) {
  const seconds = Math.max(Number(value) || 0, 0);
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 100) return `${seconds.toFixed(0)}s`;
  if (seconds >= 10) return `${seconds.toFixed(1)}s`;
  return `${seconds.toFixed(2)}s`;
}

function timingRateText(value) {
  const rate = Math.max(Number(value) || 0, 0);
  return rate >= 100 ? `${rate.toFixed(0)}x` : `${rate.toFixed(1)}x`;
}

function simulationTimingStatusText() {
  const timing = state.frameTiming;
  const pending = Math.max(Number(state.stepAccumulatorS || 0), 0);
  const droppedNow = Math.max(Number(timing.droppedSimulationThisFrameS || 0), 0);
  const droppedTotal = Math.max(Number(timing.droppedSimulationS || 0), 0);
  const status = `Rate ${timingRateText(timing.actualRateMultiple)} · Pending ${timingSecondsText(pending)}`;
  if (droppedTotal <= 1.0e-9) return status;
  if (droppedNow > 1.0e-9) {
    return `${status} · Drop ${timingSecondsText(droppedNow)} now/${timingSecondsText(droppedTotal)} total`;
  }
  return `${status} · Dropped ${timingSecondsText(droppedTotal)} total`;
}

function refreshInputState() {
  if (state.mode === "arcade" && state.arcadeSession && !state.arcadeSnapshot?.terminal) {
    state.arcadeSession.setControls(currentControls());
    syncArcadeSnapshot();
  }
  updateGhost();
  draw();
}

function completeStage() {
  state.activeStage += 1;
  state.stageDv = 0;
  const enteringFinalApproach =
    state.mode === "tutorial" && Boolean(tutorialStages[state.activeStage] && tutorialStages[state.activeStage].final);
  if (state.activeStage < tutorialStages.length - 1) {
    resetState(presets.behind);
  } else {
    resetState({ r: 0, i: -0.8, c: 0, rd: 0, id: 0, cd: 0 });
  }
  state.running = enteringFinalApproach;
  updateMissionText();
}

function samplePoint() {
  return {
    r: state.sim.r,
    i: state.sim.i,
    c: state.sim.c,
    rd: state.sim.rd,
    id: state.sim.id,
    cd: state.sim.cd,
    t: state.sim.t,
  };
}

function rangeKm() {
  if (state.mode === "arcade" && state.arcadeTransition?.clear_range_km !== undefined) {
    return Number(state.arcadeTransition.clear_range_km || 0);
  }
  if (state.mode === "arcade" && state.arcadeSnapshot) {
    return Number(state.arcadeSnapshot.range_km || 0);
  }
  return Math.hypot(state.sim.r, state.sim.i, state.sim.c);
}

function currentArcadeGoalRangeKm() {
  return Number(
    state.arcadeSnapshot?.goal_range_km ??
      state.arcadeSession?.config?.goal_range_km ??
      ARCADE_CHALLENGE_RECORD.config.goal_range_km,
  );
}

function relativeSpeedKmS() {
  if (state.mode === "arcade" && state.arcadeSnapshot) {
    return Number(state.arcadeSnapshot.relative_speed_km_s || 0);
  }
  return Math.hypot(state.sim.rd, state.sim.id, state.sim.cd);
}

function updateMissionText() {
  if (state.mode === "selector") {
    el.shell.classList.add("selector-mode");
    el.shell.classList.remove(...SHELL_GAME_MODE_CLASSES);
    renderLevelSelector();
    return;
  }
  if (state.mode === "sandboxSetup") return;
  el.shell.classList.remove("selector-mode");
  el.shell.classList.toggle("primer-mode", state.mode === "primer");
  el.shell.classList.toggle("mode-arcade", state.mode === "arcade");
  el.shell.classList.toggle(
    "mode-sandbox",
    state.mode === "sandbox" || state.mode === "operatorSandbox" || operatorScriptModeActive(),
  );
  el.shell.classList.toggle("mode-tutorial", state.mode === "tutorial" || state.mode === "operatorTutorial");
  el.shell.classList.toggle("mode-operator", operatorModeActive());
  el.shell.classList.toggle("mode-operator-script", operatorScriptModeActive());
  el.modeLabel.textContent = operatorModeActive()
    ? "Operator Mode"
    : operatorScriptModeActive()
      ? "Operator Script"
    : state.mode === "sandbox"
      ? "Sandbox"
      : state.mode === "arcade"
        ? "Arcade"
        : "Tutorial";
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    el.levelLabel.textContent = state.activePlayMode === "operator" ? "OPERATOR RIC FRAME PRIMER" : "RIC FRAME PRIMER";
    el.objectiveTitle.textContent = stage.title;
    if (el.objectiveText) {
      el.objectiveText.textContent = stage.text;
    }
  } else if (operatorScriptModeActive()) {
    el.levelLabel.textContent =
      state.mode === "operatorScriptTutorial" ? `Operator Mode   ${operatorTutorialDemoTitle()}` : "Operator Mode   Sandbox";
    el.objectiveTitle.textContent = state.mode === "operatorScriptTutorial" ? "Launch Demo" : "Script Burns";
    if (el.objectiveText) {
      el.objectiveText.textContent =
        state.mode === "operatorScriptTutorial"
          ? operatorTutorialStatusText()
          : "Build a burn script, preview the projected path, then launch playback.";
    }
  } else if (state.mode === "operatorSandbox") {
    el.levelLabel.textContent = "OPERATOR SANDBOX";
    el.objectiveTitle.textContent = "Playback";
    if (el.objectiveText) {
      el.objectiveText.textContent = nextOperatorBurnText();
    }
  } else if (state.mode === "sandbox") {
    el.levelLabel.textContent = "RPO SANDBOX";
    el.objectiveTitle.textContent = "Free Flight";
    if (el.objectiveText) {
      el.objectiveText.textContent = "";
    }
  } else if (state.mode === "arcade") {
    el.levelLabel.textContent = "PURSUIT ARCADE";
    const snap = state.arcadeSnapshot;
    const roundLabel = snap?.is_boss_round ? `Boss Round ${snap.round_index}` : `Round ${snap?.round_index || 1}`;
    el.objectiveTitle.textContent = roundLabel;
    if (el.objectiveText) {
      el.objectiveText.textContent = `Reach ${formatDistanceKm(currentArcadeGoalRangeKm())}`;
    }
  } else if (state.mode === "operatorTutorial") {
    el.levelLabel.textContent = "LEVEL 0 - OPERATOR TUTORIAL";
    el.objectiveTitle.textContent = operatorTutorialDemoTitle();
    if (el.objectiveText) {
      el.objectiveText.textContent = state.running ? nextOperatorBurnText() : operatorTutorialStatusText();
    }
  } else {
    el.levelLabel.textContent = "LEVEL 0 - TUTORIAL";
    const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
    el.objectiveTitle.textContent = stage.title;
    if (el.objectiveText) {
      el.objectiveText.textContent = "";
    }
  }
  el.pauseButton.disabled = false;
  if (state.mode === "arcade") {
    el.pauseButton.textContent = state.arcadeTransition ? "Next Round" : state.running ? "Running" : "Start";
    el.pauseButton.disabled = Boolean(state.running && !state.arcadeTransition);
  } else if (operatorScriptModeActive()) {
    el.pauseButton.textContent = state.mode === "operatorScriptTutorial" ? "Launch Demo" : "Launch";
    el.pauseButton.disabled = Boolean(state.operatorPlanError);
  } else if (operatorModeActive()) {
    el.pauseButton.textContent = state.running ? "Pause" : state.sim.t > 0 ? "Resume" : "Launch";
    el.pauseButton.disabled = Boolean(!state.running && state.operatorPlanError);
  } else {
    el.pauseButton.textContent = state.mode === "primer" ? primerAdvanceLabel() : state.running ? "Pause" : "Start";
  }
  el.resetButton.textContent =
    state.mode === "primer"
      ? "Replay"
      : state.mode === "operatorScriptTutorial"
        ? "Level Select"
        : operatorScriptModeActive()
          ? "Cancel"
          : operatorModeActive()
            ? "Script"
            : "Reset";
  if (!operatorScriptModeActive()) state.equationSheetVisible = false;
  renderOperatorPanel();
  syncEquationSheet();
  updatePlotTitles();
  syncViewButtons();
  syncMusicButton();
}

function primerAdvanceLabel() {
  return state.primerStage >= primerStages.length - 1
    ? state.activePlayMode === "operator"
      ? "Script Burns"
      : "Start Tutorial"
    : "Next";
}

function updatePlotTitles() {
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    const riIsEci = stage.eciPlane === "ri";
    el.riTitle.textContent = riIsEci ? "ECI Orbit" : "RI Plane";
    el.riSubtitle.textContent = riIsEci ? stage.eciSubtitle : stage.localSubtitle;
    el.rcTitle.textContent = riIsEci ? "RC Plane" : "ECI Orbit";
    el.rcSubtitle.textContent = riIsEci ? stage.localSubtitle : stage.eciSubtitle;
    return;
  }
  if (operatorScriptModeActive()) {
    el.riTitle.textContent = "Initial RI";
    el.riSubtitle.textContent = "";
    el.rcTitle.textContent = "Initial RC";
    el.rcSubtitle.textContent = "";
    return;
  }
  el.riTitle.textContent = "RI Plane";
  el.riSubtitle.textContent = "In-Track vs Radial";
  el.rcTitle.textContent = "RC Plane";
  el.rcSubtitle.textContent = "Cross-Track vs Radial";
}

function syncMusicButton() {
  const musicPrefix = state.activeView === "mobile" ? "" : "M ";
  el.musicButton.textContent = state.musicEnabled ? `${musicPrefix}Music: ON` : `${musicPrefix}Music: OFF`;
  el.musicButton.classList.remove("active");
  el.musicButton.setAttribute("aria-pressed", String(state.musicEnabled));
  el.selectorMusicButton.textContent = state.musicEnabled ? "Music: ON" : "Music: OFF";
  el.selectorMusicButton.classList.remove("active");
  el.selectorMusicButton.setAttribute("aria-pressed", String(state.musicEnabled));
}

function playMusicFromGesture() {
  if (!state.musicEnabled || !music.paused) return;
  state.musicStartRequested = true;
  music.play().catch(() => {
    // Browser audio policies require a click or key press; the next gesture will retry.
  });
}

function setMusicTrackForMode(mode) {
  const nextSrc = new URL(MUSIC_TRACKS[mode] || MUSIC_TRACKS.tutorial, window.location.href).href;
  if (music.currentSrc === nextSrc || music.src === nextSrc) return;
  const shouldResume = state.musicEnabled && state.musicStartRequested && !music.paused;
  music.pause();
  music.src = nextSrc;
  music.load();
  if (shouldResume) {
    playMusicFromGesture();
  }
}

function initialMusicTrackForMode(mode) {
  if (mode === "sandbox" || mode === "operatorSandbox" || mode === "operatorScriptSandbox") return "sandbox";
  if (mode === "arcade") return "arcade";
  return "tutorial";
}

function arcadeMusicTrackKey() {
  return state.arcadeSnapshot?.is_boss_round ? "arcadeBoss" : "arcade";
}

function toggleMusic() {
  if (state.musicEnabled && music.paused && !state.musicStartRequested) {
    playMusicFromGesture();
    syncMusicButton();
    return;
  }
  state.musicEnabled = !state.musicEnabled;
  trackEvent("music_toggle", { enabled: state.musicEnabled, mode: state.mode });
  if (state.musicEnabled) {
    playMusicFromGesture();
  } else {
    music.pause();
  }
  syncMusicButton();
}

function isEditableControlTarget(target) {
  return target instanceof Element && Boolean(target.closest("input, select, textarea, [contenteditable='true']"));
}

function updateGhost() {
  if (state.mode === "selector" || state.mode === "primer") {
    state.ghost = [];
    state.tutorialTargetPath = [];
    return;
  }
  if (operatorScriptModeActive()) {
    state.ghost = state.operatorPlanPath;
    state.tutorialTargetPath = [];
    return;
  }
  if (operatorModeActive()) {
    state.ghost = predictGhost(operatorBurnProjectionSeed(), currentOrbitPeriodS(), MAX_GHOST_DRAW_POINTS);
    state.tutorialTargetPath = [];
    return;
  }
  if (state.mode === "arcade") {
    updateArcadeGhosts();
    state.tutorialTargetPath = [];
    return;
  }
  state.ghost = predictGhost(livePredictionSeed(), currentOrbitPeriodS(), MAX_GHOST_DRAW_POINTS);
  state.tutorialTargetPath = [];
  if (state.mode !== "tutorial") return;
  const stage = tutorialStages[state.activeStage];
  if (!stage || !stage.axis || stage.final) return;
  if (!state.stageStart) state.stageStart = { ...state.sim };
  const seed = { ...state.stageStart };
  const dvKmS = (stage.targetDv / 1000) * stage.sign;
  seed[`${stage.axis}d`] += dvKmS;
  state.tutorialTargetPath = predictGhost(seed, ORBIT_PERIOD_S, TUTORIAL_TARGET_PATH_POINTS);
}

function livePredictionSeed() {
  return { ...state.sim };
}

function updateArcadeGhosts() {
  const targetSeed = { ...state.arcadeTargetRel };
  const chaserTargetSeed = { ...state.arcadeChaserTargetRel };
  if (state.arcadeSnapshot?.is_boss_round && state.arcadeReferenceStateEci) {
    const times = ghostTimes(arcadeProjectionHorizonS(), MAX_GHOST_DRAW_POINTS);
    state.targetGhost = ellipticLinearCoastStates(simToArcadeRicBlock(targetSeed), times, state.arcadeReferenceStateEci, MU)
      .map(arcadeCoastPointToSim);
    state.ghost = ellipticLinearCoastStates(simToArcadeRicBlock(state.sim), times, state.arcadeReferenceStateEci, MU)
      .map(arcadeCoastPointToSim);
    return;
  }
  state.targetGhost = predictGhost(targetSeed, ORBIT_PERIOD_S, MAX_GHOST_DRAW_POINTS);
  const relativeGhost = predictGhost(chaserTargetSeed, ORBIT_PERIOD_S, MAX_GHOST_DRAW_POINTS);
  state.ghost = relativeGhost.map((point, idx) => addRicPoint(point, state.targetGhost[idx] || targetSeed));
}

function arcadeProjectionHorizonS() {
  const aKm = Number(state.arcadeSession?.config?.target_coes?.a_km || TARGET_A_KM);
  const mu = Number(state.arcadeSession?.config?.mu_km3_s2 || MU);
  return (2 * Math.PI) * Math.sqrt(Math.max(aKm, 1) ** 3 / mu);
}

function ghostTimes(horizonS, samples) {
  const count = Math.max(Math.floor(samples), 2);
  return Array.from({ length: count }, (_, idx) => (horizonS * idx) / (count - 1));
}

function simToArcadeRicBlock(seed) {
  return {
    r_km: Number(seed?.r || 0),
    i_km: Number(seed?.i || 0),
    c_km: Number(seed?.c || 0),
    rd_km_s: Number(seed?.rd || 0),
    id_km_s: Number(seed?.id || 0),
    cd_km_s: Number(seed?.cd || 0),
  };
}

function arcadeCoastPointToSim(point) {
  return {
    r: Number(point?.r || 0),
    i: Number(point?.i || 0),
    c: Number(point?.c || 0),
    rd: Number(point?.rd || 0),
    id: Number(point?.id || 0),
    cd: Number(point?.cd || 0),
    t: Number(point?.t || 0),
  };
}

function addRicPoint(a, b) {
  return {
    r: Number(a.r || 0) + Number(b.r || 0),
    i: Number(a.i || 0) + Number(b.i || 0),
    c: Number(a.c || 0) + Number(b.c || 0),
    rd: Number(a.rd || 0) + Number(b.rd || 0),
    id: Number(a.id || 0) + Number(b.id || 0),
    cd: Number(a.cd || 0) + Number(b.cd || 0),
    t: Number(a.t || 0),
  };
}

function ensureOperatorRows(mode = state.mode) {
  if (mode === "operatorScriptTutorial" || mode === "operatorTutorial") {
    state.operatorBurnRows = operatorTutorialRowsForStage();
    return;
  }
  if (state.operatorBurnRows.length > 0) return;
  state.operatorBurnRows =
    [{ t: "0", r: "0", i: "0.1", c: "0" }];
}

function activeOperatorTutorialStage() {
  return operatorTutorialStages[state.operatorTutorialStage] || null;
}

function operatorTutorialRowsForStage() {
  const stage = activeOperatorTutorialStage() || operatorTutorialStages[0];
  const row = { t: String(OPERATOR_TUTORIAL_BURN_TIME_S), r: "0", i: "0", c: "0" };
  if (stage?.axis) {
    row[stage.axis] = formatOperatorTutorialBurnComponent(stage.sign * OPERATOR_TUTORIAL_BURN_DELTA_V_M_S);
  }
  return [row];
}

function formatOperatorTutorialBurnComponent(value) {
  return Number(value).toFixed(2).replace(/\.?0+$/, "") || "0";
}

function operatorTutorialDemoTitle() {
  const stage = activeOperatorTutorialStage();
  if (!stage) return "Operator Tutorial";
  return `Demo ${state.operatorTutorialStage + 1}/${operatorTutorialStages.length}: ${stage.displayLabel}`;
}

function operatorTutorialStatusText() {
  const stage = activeOperatorTutorialStage();
  if (!stage) return "Operator tutorial complete.";
  return `${operatorTutorialDemoTitle()}. Observe ${OPERATOR_TUTORIAL_STAGE_DURATION_S}s at ${OPERATOR_TUTORIAL_PLAYBACK_SPEED_MULTIPLE}x, then the next scripted burn will load.`;
}

function renderOperatorPanel() {
  if (!el.operatorPanel || !el.operatorBurnRows) return;
  const visible = operatorScriptModeActive();
  el.operatorPanel.classList.toggle("hidden", !visible);
  if (!visible) return;
  ensureOperatorRows();
  const readOnly = state.mode === "operatorScriptTutorial";
  if (el.operatorAddBurn) el.operatorAddBurn.classList.toggle("hidden", readOnly);
  const signature = operatorPanelSignature(readOnly);
  if (state.operatorPanelSignature === signature && el.operatorBurnRows.children.length > 0) {
    renderOperatorPlanStatus();
    return;
  }
  el.operatorBurnRows.replaceChildren();
  el.operatorBurnRows.append(operatorHeaderRow());
  state.operatorBurnRows.forEach((row, idx) => {
    const rowEl = document.createElement("div");
    rowEl.className = "operator-row";
    rowEl.append(operatorRowLabel(idx));
    ["t", "r", "i", "c"].forEach((field) => {
      const input = document.createElement("input");
      input.type = "number";
      input.step = field === "t" ? "1" : "0.01";
      input.value = row[field] ?? "";
      input.dataset.operatorField = field;
      input.dataset.operatorIndex = String(idx);
      input.disabled = readOnly;
      input.setAttribute("aria-label", operatorInputLabel(field, idx));
      input.addEventListener("input", () => {
        if (readOnly) return;
        state.operatorBurnRows[idx][field] = input.value;
        state.operatorPanelSignature = operatorPanelSignature(readOnly);
        updateOperatorPlan();
      });
      rowEl.append(input);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "x";
    remove.disabled = readOnly;
    remove.classList.toggle("hidden", readOnly);
    remove.setAttribute("aria-label", `Remove burn ${idx + 1}`);
    remove.addEventListener("click", () => {
      if (readOnly) return;
      state.operatorBurnRows.splice(idx, 1);
      if (state.operatorBurnRows.length <= 0) state.operatorBurnRows.push({ t: "", r: "", i: "", c: "" });
      state.operatorPanelSignature = "";
      updateOperatorPlan();
      renderOperatorPanel();
    });
    rowEl.append(remove);
    el.operatorBurnRows.append(rowEl);
  });
  state.operatorPanelSignature = signature;
  renderOperatorPlanStatus();
}

function operatorPanelSignature(readOnly = state.mode === "operatorScriptTutorial") {
  return `${state.mode}:${readOnly ? "read-only" : "editable"}:${JSON.stringify(state.operatorBurnRows)}`;
}

function operatorHeaderRow() {
  const rowEl = document.createElement("div");
  rowEl.className = "operator-row operator-header";
  ["#", "T (s)", "R (m/s)", "I (m/s)", "C (m/s)", ""].forEach((text) => {
    const label = document.createElement("span");
    label.textContent = text;
    rowEl.append(label);
  });
  return rowEl;
}

function operatorRowLabel(idx) {
  const label = document.createElement("span");
  label.textContent = String(idx + 1);
  return label;
}

function operatorInputLabel(field, idx) {
  const labels = { t: "time seconds", r: "radial meters per second", i: "in-track meters per second", c: "cross-track meters per second" };
  return `Burn ${idx + 1} ${labels[field]}`;
}

function operatorBurnTimeInputValue(timeS) {
  const value = Math.max(Number(timeS || 0), 0);
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value - Math.round(value)) < 1.0e-9) return String(Math.round(value));
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function addOperatorBurnRow() {
  if (state.mode === "operatorScriptTutorial") return;
  ensureOperatorRows();
  const probeTime = state.operatorTrajectoryProbe ? operatorBurnTimeInputValue(state.operatorTrajectoryProbe.timeS) : "";
  const lastTime = Math.max(
    0,
    ...state.operatorBurnRows.map((row) => Number(row.t)).filter((value) => Number.isFinite(value)),
  );
  state.operatorBurnRows.push({ t: probeTime || String(lastTime + 600), r: "0", i: "0", c: "0" });
  state.operatorTrajectoryProbe = null;
  state.operatorPanelSignature = "";
  updateOperatorPlan();
  renderOperatorPanel();
}

function updateOperatorPlan() {
  const { burns, error } = parseOperatorBurns();
  const planKey = operatorPlanKey(burns);
  state.operatorPlan = burns;
  state.operatorPlanError = error;
  if (state.operatorPlanKey !== planKey) {
    state.operatorTrajectoryProbe = null;
    state.operatorPlanKey = planKey;
  }
  const preview = buildOperatorPreviewArtifacts(operatorInitialSeed(), burns);
  state.operatorPlanPath = preview.path;
  state.operatorBurnMarkers = preview.markers;
  state.ghost = state.operatorPlanPath;
  state.tutorialTargetPath = [];
  renderOperatorPlanStatus();
  draw();
}

function operatorPlanKey(burns) {
  return (Array.isArray(burns) ? burns : [])
    .map((burn) =>
      [
        Number(burn.timeS || 0).toFixed(3),
        Number(burn.rMps || 0).toFixed(6),
        Number(burn.iMps || 0).toFixed(6),
        Number(burn.cMps || 0).toFixed(6),
      ].join(","),
    )
    .join("|");
}

function renderOperatorPlanStatus() {
  if (!el.operatorStatus || !el.operatorError) return;
  const plannedDv = state.operatorPlan.reduce((sum, burn) => sum + burn.dvMps, 0);
  if (state.mode === "operatorScriptTutorial") {
    el.operatorStatus.textContent = `${operatorTutorialDemoTitle()} | ${formatSpeedMS(plannedDv)} planned`;
    el.operatorError.textContent = state.operatorPlanError || "Read-only tutorial script. Press Launch Demo to observe the burn.";
    return;
  }
  el.operatorStatus.textContent = `${state.operatorPlan.length} burns | ${formatSpeedMS(plannedDv)} planned`;
  el.operatorError.textContent =
    state.operatorPlanError || `Max ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)} per burn | ${OPERATOR_BURN_SPACING_S}s minimum spacing`;
}

function syncEquationSheet() {
  const visible = operatorScriptModeActive() && state.equationSheetVisible;
  if (el.equationSheet) el.equationSheet.classList.toggle("hidden", !visible);
  if (el.equationSheetButton) {
    el.equationSheetButton.textContent = visible ? "Hide Equation Sheet" : "Show Equation Sheet";
    el.equationSheetButton.setAttribute("aria-pressed", String(visible));
  }
}

function toggleEquationSheet() {
  if (!operatorScriptModeActive()) return;
  state.equationSheetVisible = !state.equationSheetVisible;
  syncEquationSheet();
}

function parseOperatorBurns() {
  const errors = [];
  const burns = [];
  state.operatorBurnRows.forEach((row, idx) => {
    const values = {
      t: String(row.t ?? "").trim(),
      r: String(row.r ?? "").trim(),
      i: String(row.i ?? "").trim(),
      c: String(row.c ?? "").trim(),
    };
    if (!values.t && !values.r && !values.i && !values.c) return;
    const timeS = Number(values.t);
    const rMps = Number(values.r || 0);
    const iMps = Number(values.i || 0);
    const cMps = Number(values.c || 0);
    if (!Number.isFinite(timeS) || timeS < 0) errors.push(`Burn ${idx + 1}: enter a time >= 0.`);
    if (![rMps, iMps, cMps].every(Number.isFinite)) errors.push(`Burn ${idx + 1}: enter numeric R/I/C values.`);
    const dvMps = Math.hypot(rMps, iMps, cMps);
    if (dvMps > OPERATOR_BURN_MAX_DV_M_S + 1.0e-9) {
      errors.push(`Burn ${idx + 1}: ${formatSpeedMS(dvMps)} exceeds ${formatSpeedMS(OPERATOR_BURN_MAX_DV_M_S)}.`);
    }
    if (errors.length <= 0 || Number.isFinite(timeS)) {
      burns.push({ timeS, rMps, iMps, cMps, dvMps });
    }
  });
  burns.sort((a, b) => a.timeS - b.timeS);
  for (let idx = 1; idx < burns.length; idx += 1) {
    if (burns[idx].timeS - burns[idx - 1].timeS < OPERATOR_BURN_SPACING_S) {
      errors.push(`Burns ${idx} and ${idx + 1}: separate by at least ${OPERATOR_BURN_SPACING_S}s.`);
      break;
    }
  }
  return { burns: errors.length > 0 ? [] : burns, error: errors[0] || "" };
}

function operatorInitialSeed() {
  if (operatorExperienceActive()) return state.stageStart ? { ...state.stageStart, t: 0, dv: 0 } : { ...state.sim, t: 0, dv: 0 };
  return { ...state.sim };
}

function buildOperatorPreviewArtifacts(seed, burns) {
  if (!operatorExperienceActive()) return { path: [], markers: [] };
  if (sandboxExperienceActive()) return buildSandboxOperatorPreviewArtifacts(seed, burns);
  const validBurns = Array.isArray(burns) ? burns : [];
  const horizonS = operatorPlaybackEndS(validBurns);
  const sampleCount = Math.max(2, OPERATOR_PREVIEW_POINTS);
  const path = [];
  const markers = [];
  let segmentSeed = { ...seed, t: 0 };
  let segmentStartS = 0;
  let burnIdx = 0;
  for (let idx = 0; idx < sampleCount; idx += 1) {
    const absoluteTimeS = (horizonS * idx) / (sampleCount - 1);
    while (burnIdx < validBurns.length && validBurns[burnIdx].timeS <= absoluteTimeS) {
      const burn = validBurns[burnIdx];
      segmentSeed = cwCoastPoint(segmentSeed, burn.timeS - segmentStartS);
      segmentSeed.rd += burn.rMps / 1000;
      segmentSeed.id += burn.iMps / 1000;
      segmentSeed.cd += burn.cMps / 1000;
      segmentSeed.t = burn.timeS;
      markers.push({ ...segmentSeed, burnIndex: burnIdx + 1, timeS: burn.timeS });
      segmentStartS = burn.timeS;
      burnIdx += 1;
    }
    const point = cwCoastPoint(segmentSeed, absoluteTimeS - segmentStartS);
    point.t = absoluteTimeS;
    path.push(point);
  }
  return { path, markers };
}

function buildSandboxOperatorPreviewArtifacts(seed, burns) {
  if (sandboxEllipticProjectionActive()) {
    return buildSandboxEllipticOperatorPreviewArtifacts(seed, burns);
  }
  const validBurns = Array.isArray(burns) ? burns : [];
  const horizonS = operatorPlaybackEndS(validBurns);
  const sampleCount = Math.max(2, OPERATOR_PREVIEW_POINTS);
  const path = [];
  const markers = [];
  let pair = sandboxPairFromSeed({ ...seed, t: 0 });
  let burnIdx = 0;
  for (let idx = 0; idx < sampleCount; idx += 1) {
    const absoluteTimeS = (horizonS * idx) / (sampleCount - 1);
    while (burnIdx < validBurns.length && validBurns[burnIdx].timeS <= absoluteTimeS) {
      const burn = validBurns[burnIdx];
      pair = stepSandboxPair(pair, burn.timeS - pair.timeS);
      const burnState = sandboxPairRelativeState(pair);
      burnState.rd += burn.rMps / 1000;
      burnState.id += burn.iMps / 1000;
      burnState.cd += burn.cMps / 1000;
      pair.chaser = stateFromRelativeRic(pair.target, {
        r_km: burnState.r,
        i_km: burnState.i,
        c_km: burnState.c,
        rd_km_s: burnState.rd,
        id_km_s: burnState.id,
        cd_km_s: burnState.cd,
      });
      markers.push({ ...burnState, burnIndex: burnIdx + 1, timeS: burn.timeS });
      burnIdx += 1;
    }
    pair = stepSandboxPair(pair, absoluteTimeS - pair.timeS);
    path.push(sandboxPairRelativeState(pair));
  }
  return { path, markers };
}

function buildSandboxEllipticOperatorPreviewArtifacts(seed, burns) {
  const validBurns = Array.isArray(burns) ? burns : [];
  const horizonS = operatorPlaybackEndS(validBurns);
  const sampleCount = Math.max(2, OPERATOR_PREVIEW_POINTS);
  const path = [];
  const markers = [];
  let segmentSeed = { ...seed, t: 0 };
  let burnIdx = 0;
  for (let idx = 0; idx < sampleCount; idx += 1) {
    const absoluteTimeS = (horizonS * idx) / (sampleCount - 1);
    while (burnIdx < validBurns.length && validBurns[burnIdx].timeS <= absoluteTimeS) {
      const burn = validBurns[burnIdx];
      segmentSeed = sandboxProjectionCoastPoint(segmentSeed, burn.timeS - segmentSeed.t);
      segmentSeed.rd += burn.rMps / 1000;
      segmentSeed.id += burn.iMps / 1000;
      segmentSeed.cd += burn.cMps / 1000;
      segmentSeed.t = burn.timeS;
      markers.push({ ...segmentSeed, burnIndex: burnIdx + 1, timeS: burn.timeS });
      burnIdx += 1;
    }
    path.push(sandboxProjectionCoastPoint(segmentSeed, absoluteTimeS - segmentSeed.t));
  }
  return { path, markers };
}

function operatorPlaybackEndS(burns = state.operatorPlan) {
  const lastBurnS = burns.length > 0 ? Math.max(...burns.map((burn) => burn.timeS)) : 0;
  const orbitPeriodS = currentOrbitPeriodS();
  return Math.max(lastBurnS + orbitPeriodS, orbitPeriodS);
}

function predictGhost(seed, horizonS, samples) {
  if (sandboxExperienceActive()) return predictSandboxGhost(seed, horizonS, samples);
  const ghost = [];
  const count = Math.max(Math.floor(samples), 2);
  for (let idx = 0; idx < count; idx += 1) {
    const t = count <= 1 ? 0 : (horizonS * idx) / (count - 1);
    ghost.push(cwCoastPoint(seed, t));
  }
  return ghost;
}

function predictSandboxGhost(seed, horizonS, samples) {
  if (sandboxEllipticProjectionActive()) {
    return sandboxEllipticLinearCoastStates(
      state.sandboxSetup,
      seed,
      ghostTimes(horizonS, samples),
      MU,
    );
  }
  const ghost = [];
  const count = Math.max(Math.floor(samples), 2);
  let pair = sandboxPairFromSeed(seed);
  let previousTimeS = 0;
  for (let idx = 0; idx < count; idx += 1) {
    const timeS = count <= 1 ? 0 : (Number(horizonS) * idx) / (count - 1);
    pair = stepSandboxPair(pair, timeS - previousTimeS);
    ghost.push(sandboxPairRelativeState(pair, seed));
    previousTimeS = timeS;
  }
  return ghost;
}

function coastPoint(seed, tS) {
  return sandboxExperienceActive() ? sandboxCoastPoint(seed, tS) : cwCoastPoint(seed, tS);
}

function cwCoastPoint(seed, tS) {
  const x = seed.r;
  const y = seed.i;
  const z = seed.c;
  const xd = seed.rd;
  const yd = seed.id;
  const zd = seed.cd;
  const n = MEAN_MOTION;
  const t = Number(tS);
  const absoluteTimeS = Number(seed.t || 0) + t;
  if (Math.abs(n) <= 1.0e-12) {
    return { r: x + xd * t, i: y + yd * t, c: z + zd * t, rd: xd, id: yd, cd: zd, t: absoluteTimeS };
  }
  const nt = n * t;
  const cosNt = Math.cos(nt);
  const sinNt = Math.sin(nt);
  return {
    r: (4 - 3 * cosNt) * x + (sinNt / n) * xd + ((2 * (1 - cosNt)) / n) * yd,
    i: 6 * (sinNt - nt) * x + y - ((2 * (1 - cosNt)) / n) * xd + (((4 * sinNt - 3 * nt) / n) * yd),
    c: cosNt * z + (sinNt / n) * zd,
    rd: 3 * n * sinNt * x + cosNt * xd + 2 * sinNt * yd,
    id: -6 * n * (1 - cosNt) * x - 2 * sinNt * xd + (4 * cosNt - 3) * yd,
    cd: -n * sinNt * z + cosNt * zd,
    t: absoluteTimeS,
  };
}

function integrateCopy(s, u, dt) {
  stepHcwStateInPlace(s, u, dt);
}

function draw() {
  const use3D = sandbox3d.sync();
  if (state.mode === "selector" || state.mode === "sandboxSetup") {
    updateDebugState();
    return;
  }
  updateHud();
  updateDebugState();
  if (use3D) sandbox3d.render();
  else {
    drawPlot(el.riCanvas, "i", "r", "ri");
    drawPlot(el.rcCanvas, "c", "r", "rc");
  }
}

function updateDebugState() {
  window.__OEL_RPO_PREVIEW_DEBUG__ = {
    buildId: BUILD_ID,
    mode: state.mode,
    running: state.running,
    activeStage: state.activeStage,
    primerStage: state.primerStage,
    selectedLevel: state.selectedLevel,
    playMode: state.playMode,
    activePlayMode: state.activePlayMode,
    activeLevelId: state.activeLevelId,
    sandboxSetup: { ...state.sandboxSetup },
    sandboxPhysics: sandboxExperienceActive() ? "two_body_target_relative" : "hcw",
    frameConvention: state.frameConvention,
    viewPreference: state.viewPreference,
    activeView: state.activeView,
    speedMultiple: currentSpeedMultiple(),
    effectiveSpeedMultiple: effectiveSpeedMultiple(),
    camera3d: sandbox3d.debug(),
    cameraRuleMode: state.cameraRuleMode,
    musicSrc: music.currentSrc || music.src,
    analytics: {
      enabled: analytics.enabled,
      provider: analytics.provider,
      domain: analytics.domain,
    },
    controls: currentControls(),
    sim: { ...state.sim },
    timing: {
      ...state.frameTiming,
      pendingSimulationS: state.stepAccumulatorS,
      maxPendingSimulationS: MAX_PENDING_SIMULATION_S,
      maxStepsPerFrame: MAX_STEPS_PER_FRAME,
    },
    arcade: state.arcadeSnapshot
      ? {
          score: state.arcadeSnapshot.score,
          goalRangeKm: state.arcadeSnapshot.goal_range_km,
          roundIndex: state.arcadeSnapshot.round_index,
          terminal: state.arcadeSnapshot.terminal,
          inputEventCount: state.arcadeSnapshot.input_events.length,
          validation: state.arcadeValidation?.status || "",
          challengeHash: ARCADE_CHALLENGE_RECORD.config_hash,
        }
      : null,
    livePredictionSeed: livePredictionSeed(),
    operatorPlan: state.operatorPlan.map((burn) => ({ ...burn })),
    operatorPlanError: state.operatorPlanError,
    operatorBurnMarkers: state.operatorBurnMarkers.map((marker) => ({ ...marker })),
    operatorTrajectoryProbe: state.operatorTrajectoryProbe
      ? { timeS: state.operatorTrajectoryProbe.timeS, state: { ...state.operatorTrajectoryProbe.state } }
      : null,
    ghostHead: state.ghost.slice(0, 8).map((point) => ({ ...point })),
    tutorialTargetHead: state.tutorialTargetPath.slice(0, 8).map((point) => ({ ...point })),
  };
}

function updateHud() {
  if (state.mode === "selector") return;
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    const stepText = `Step ${state.primerStage + 1}/${primerStages.length}`;
    el.rangeMetric.textContent = stepText;
    el.speedMetric.textContent = stage.title;
    el.dvMetric.textContent = "RIC";
    el.timeMetric.textContent = `${Math.round(state.primerTimeS)} s`;
    el.topRangeMetric.textContent = `INFO ${stepText}`;
    el.topSpeedMetric.textContent = `INFO ${stage.title}`;
    el.topDvMetric.textContent = "";
    el.topTimingMetric.textContent = "";
    el.topTimingMetric.classList.remove("timing-warning");
    el.hudLine.textContent = `${stepText}   ${stage.title}`;
    el.coachHint.textContent = currentCoachHint();
    el.commandLine.textContent = "";
    el.footerLine.textContent = `${primerAdvanceLabel()} to continue   Esc Level Select`;
    el.speedMultiple.textContent = "1x";
    syncMobileSpeedButtons();
    el.rMeter.value = 0;
    el.iMeter.value = 0;
    el.cMeter.value = 0;
    updateMissionText();
    return;
  }
  const u = currentControls();
  const currentRangeKm = rangeKm();
  const currentRelativeSpeedKmS = relativeSpeedKmS();
  const rangeText = formatDistanceKm(currentRangeKm);
  const speedText = formatSpeedKmS(currentRelativeSpeedKmS);
  const hudSigFigs = mobileLandscapeHudSigFigs();
  const compactLandscapeHud = hudSigFigs === 2;
  const hudRangeText = formatDistanceKm(currentRangeKm, hudSigFigs);
  const hudSpeedText = formatSpeedKmS(currentRelativeSpeedKmS, hudSigFigs);
  const arcadePlayerDvUsed = Number(state.arcadeSnapshot?.player_delta_v_m_s || state.sim.dv || 0);
  const arcadeTargetDvUsed = Number(state.arcadeSnapshot?.target_delta_v_m_s || 0);
  const arcadePlayerDvBudget = Number(state.arcadeSnapshot?.max_delta_v_m_s ?? DEFAULT_PURSUIT_CHALLENGE.max_delta_v_m_s ?? 0);
  const arcadeTargetDvBudget = Number(
    state.arcadeSnapshot?.max_target_delta_v_m_s ?? DEFAULT_PURSUIT_CHALLENGE.max_target_delta_v_m_s ?? 0,
  );
  const arcadePlayerDvRemaining = Math.max(arcadePlayerDvBudget - arcadePlayerDvUsed, 0);
  const arcadeTargetDvRemaining = Math.max(arcadeTargetDvBudget - arcadeTargetDvUsed, 0);
  const arcadeTimeRemainingS = Math.max(
    Number(state.arcadeSnapshot?.remaining_time_s || 0) - Number(state.arcadeSnapshot?.time_s || 0),
    0,
  );
  const dvText = state.mode === "arcade" ? formatSpeedMS(arcadePlayerDvRemaining) : formatSpeedMS(state.sim.dv);
  const targetDvText = formatSpeedMS(arcadeTargetDvRemaining);
  const arcadeTimeText = formatClockTime(arcadeTimeRemainingS);
  const timeText = `${Math.round(state.sim.t)} s`;
  el.rangeMetric.textContent = rangeText;
  el.speedMetric.textContent = speedText;
  el.dvMetric.textContent = dvText;
  el.timeMetric.textContent = timeText;
  el.topTimingMetric.textContent = simulationTimingStatusText();
  el.topTimingMetric.classList.toggle(
    "timing-warning",
    Number(state.frameTiming.droppedSimulationS || 0) > 1.0e-9,
  );
  if (state.mode === "arcade") {
    const metricGap = "\u00a0".repeat(state.activeView === "mobile" ? 3 : 6);
    const timeLabel = state.activeView === "mobile" ? "Time" : "Time Remaining";
    el.topRangeMetric.textContent = `Score ${state.arcadeSnapshot?.score || 0}${metricGap}${timeLabel} ${arcadeTimeText}`;
    el.topSpeedMetric.textContent = `Target dV ${targetDvText}`;
    el.topDvMetric.textContent = `Chaser dV ${dvText}`;
  } else if (operatorExperienceActive()) {
    el.topRangeMetric.textContent = `OPERATOR Range ${rangeText}`;
    el.topSpeedMetric.textContent = `OPERATOR Rel Speed ${speedText}`;
    el.topDvMetric.textContent = `OPERATOR Delta-v ${dvText}`;
  } else {
    el.topRangeMetric.textContent = `INFO Range ${rangeText}`;
    el.topSpeedMetric.textContent = `INFO Rel Speed ${speedText}`;
    el.topDvMetric.textContent = `INFO Delta-v ${dvText}`;
  }
  el.hudLine.textContent = compactLandscapeHud
    ? `T=${state.sim.t.toFixed(1)}s R=${hudRangeText} Vrel=${hudSpeedText}`
    : `T=${state.sim.t.toFixed(1).padStart(7, " ")}s   Range=${hudRangeText}   Rel Speed=${hudSpeedText}`;
  el.coachHint.textContent = currentCoachHint();
  el.commandLine.textContent = commandStatusLine();
  const spaceAction =
    state.mode === "arcade" ? "Space Start" : operatorScriptModeActive() ? "Space Launch" : operatorModeActive() ? "Space Pause" : "Space Pause";
  el.footerLine.textContent = `Speed ${speedFooterText(u)}  Up/Down Speed  ${spaceAction}  R Reset  Esc Level Select`;
  el.speedMultiple.textContent = speedBadgeText(u);
  syncMobileSpeedButtons();
  el.rMeter.value = u.r;
  el.iMeter.value = u.i;
  el.cMeter.value = u.c;
  updateMissionText();
}

function currentCoachHint() {
  if (state.mode === "primer") {
    return activePrimerStage().hint;
  }
  if (state.mode === "sandbox") {
    if (state.activeView === "mobile" && sandbox3d.active()) return "Camera: Target Locked";
    const label = state.cameraRuleMode === "full_trajectory" ? "Full Trajectory" : "Satellites Only";
    const cameraLabel = state.activeView === "mobile" ? "Camera" : "C Camera";
    if (state.activeView === "mobile") return `${cameraLabel}: ${label}`;
    return `Use small pulses, then coast and watch the target-centered RIC motion. ${cameraLabel}: ${label}.`;
  }
  if (operatorScriptModeActive()) {
    if (state.mode === "operatorScriptTutorial") return operatorTutorialStatusText();
    return "Build a time-ordered burn script. The dashed path previews the planned trajectory.";
  }
  if (operatorModeActive()) {
    return nextOperatorBurnText();
  }
  if (state.mode === "arcade") {
    const snap = state.arcadeSnapshot;
    const label = state.cameraRuleMode === "full_trajectory" ? "Full Trajectory" : "Satellites Only";
    if (state.arcadeTransition) {
      const tr = state.arcadeTransition;
      return `Round ${tr.cleared_round_index} clear. +${tr.round_score.toLocaleString()} points, ${Math.round(
        tr.next_time_budget_s,
      )} s for round ${tr.next_round_index}.`;
    }
    if (snap?.terminal) return `${snap.terminal_reason} Local validator score: ${snap.score}.`;
    const cameraLabel = state.activeView === "mobile" ? "Camera" : "C Camera";
    if (state.activeView === "mobile") return `${cameraLabel}: ${label}`;
    return `${snap?.is_boss_round ? "Boss round. " : ""}Clear rounds to tighten the goal and grow the score. ${cameraLabel}: ${label}.`;
  }
  const stage = tutorialStages[state.activeStage] || tutorialStages[tutorialStages.length - 1];
  if (stage.final) {
    return `Guided burns complete. Settle gently into the green ${formatDistanceKm(0.25)} circle. Keep pulses short.`;
  }
  if (stage.speedTarget) {
    if (state.activeView === "mobile") {
      return `Want to go faster? Tap ${stage.speedTarget}x in the speed row. Current speed: ${SPEED_OPTIONS[state.speedIndex]}x.`;
    }
    return `Want to go faster? Hit the up arrow key to increase the speed multiple. Current speed: ${SPEED_OPTIONS[state.speedIndex]}x.`;
  }
  const progress = Math.min(state.stageDv, stage.targetDv || 0);
  if (state.activeView === "mobile") {
    return `${tutorialStageInstruction(stage)} Burn ${formatBurnProgressMS(progress, stage.targetDv || 0)}.`;
  }
  return `${tutorialStageInstruction(stage)} Burn progress: ${formatSpeedMS(progress)}/${formatSpeedMS(stage.targetDv || 0)}.`;
}

function commandStatusLine() {
  if (state.mode === "primer") {
    return "";
  }
  if (operatorScriptModeActive()) return state.mode === "operatorScriptTutorial" ? "Review the scripted burn, then press Launch Demo." : "Script RIC burns, then press Launch.";
  if (operatorModeActive()) return "";
  const inTrackKeys = displayAxisSign("i") > 0 ? "A -I / D +I" : "A +I / D -I";
  if (state.mode === "arcade") return `W/S R  ${inTrackKeys}  Left/Right C  Space Start  R Reset`;
  return `W/S R  ${inTrackKeys}  Left/Right C  C Camera  M Music`;
}

function nextOperatorBurnText() {
  const nextBurn = state.operatorPlan[state.operatorBurnIndex];
  if (!nextBurn) {
    return state.running ? "Next Burn: none scheduled. Coasting through preview horizon." : "Next Burn: script is ready.";
  }
  const remainingS = Math.max(nextBurn.timeS - state.sim.t, 0);
  const components = [`R ${formatSpeedMS(nextBurn.rMps)}`, `I ${formatSpeedMS(nextBurn.iMps)}`, `C ${formatSpeedMS(nextBurn.cMps)}`];
  return `Next Burn T=${Math.round(nextBurn.timeS)}s (${Math.round(remainingS)}s): ${components.join("  ")}`;
}

function tutorialStageInstruction(stage) {
  if (state.activeView !== "mobile") {
    if (stage.axis !== "i") return stage.text;
    const key = stage.sign * displayAxisSign("i") > 0 ? "D" : "A";
    return String(stage.text || "").replace(/^Hold [AD]/, `Hold ${key}`);
  }
  const label = `${stage.sign > 0 ? "+" : "-"}${String(stage.axis || "").toUpperCase()}`;
  if (stage.id === "plusI") return `Hold ${label}, then coast.`;
  if (stage.id === "minusI") return `Hold ${label}, then coast.`;
  if (stage.id === "plusR") return `Hold ${label}, then coast.`;
  if (stage.id === "minusR") return `Hold ${label}, then coast.`;
  if (stage.id === "plusC") return `Hold ${label}, then coast.`;
  if (stage.id === "minusC") return `Hold ${label}, then coast.`;
  return stage.text;
}

function syncTouchControlLabels() {
  const positiveOnRight = displayAxisSign("i") > 0;
  const left = document.querySelector('[data-touch="iMinus"]');
  const right = document.querySelector('[data-touch="iPlus"]');
  if (left) left.textContent = positiveOnRight ? "-I" : "+I";
  if (right) right.textContent = positiveOnRight ? "+I" : "-I";
}

function formatBurnProgressMS(progressMps, targetMps) {
  return `${Math.round(progressMps * 1000)}/${Math.round(targetMps * 1000)} mm/s`;
}

function syncMobileSpeedButtons() {
  el.mobileSpeedButtons.forEach((button) => {
    const active = Number(button.dataset.mobileSpeed || 0) === currentSpeedMultiple();
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function drawPlot(canvas, xAxis, yAxis, plane) {
  const ctx = canvas.getContext("2d");
  const { width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  if (state.mode === "primer") {
    const stage = activePrimerStage();
    if (stage.eciPlane === plane) {
      drawPrimerEci(ctx, width, height, stage);
    } else {
      drawPrimerRic(ctx, width, height, xAxis, yAxis, stage);
    }
    return;
  }
  const cameraCenter = cameraCenterFor(xAxis, yAxis);
  const scale = plotScale(width, height, xAxis, yAxis, cameraCenter);
  const xSign = displayAxisSign(xAxis);
  const ySign = displayAxisSign(yAxis);
  const toPx = (p) => ({
    x: width / 2 + ((p[xAxis] - cameraCenter[xAxis]) * xSign) * scale,
    y: height / 2 - ((p[yAxis] - cameraCenter[yAxis]) * ySign) * scale,
  });

  const targetState = state.mode === "arcade" ? state.arcadeTargetRel : { r: 0, i: 0, c: 0 };
  drawGrid(ctx, width, height, scale);
  drawRings(ctx, toPx, scale, xAxis, yAxis, targetState);
  drawPath(ctx, state.tutorialTargetPath, toPx, "rgba(92, 240, 132, 0.92)", true, 3);
  if (state.mode === "arcade") {
    drawPath(ctx, state.targetGhost, toPx, "rgba(245, 92, 92, 0.55)", true, 2);
    drawPath(ctx, state.targetTrail, toPx, "rgba(245, 92, 92, 0.9)", false, 2);
  }
  if (operatorScriptModeActive()) {
    drawOperatorScriptPlotOverlays(ctx, width, height, xAxis, yAxis, toPx);
  } else {
    drawPath(ctx, state.ghost, toPx, "rgba(135, 150, 172, 0.95)", true, 2);
  }
  drawPath(ctx, state.trail, toPx, "rgba(245, 205, 92, 0.95)", false);

  const target = toPx(targetState);
  const chaser = toPx(state.sim);
  drawVector(ctx, chaser, state.sim, xAxis, yAxis, "velocity");
  drawThrustVector(ctx, chaser, xAxis, yAxis);
  drawSpacecraftMarker(ctx, target, "target", { scale, fallbackRadius: 6 });
  drawSpacecraftMarker(ctx, chaser, "chaser", { scale, fallbackRadius: 7 });
  ctx.fillStyle = "rgba(170, 180, 195, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  drawAxisDirectionLabels(ctx, width, height, xAxis, yAxis);
}

function drawOperatorScriptPlotOverlays(ctx, width, height, xAxis, yAxis, toPx) {
  drawPath(ctx, state.operatorPlanPath, toPx, OPERATOR_PROJECTION_COLOR, false, 2);
  drawPath(ctx, state.operatorPlanPath, toPx, OPERATOR_PROJECTION_HIGHLIGHT, true, 1);
  state.operatorBurnMarkers.forEach((marker) => {
    const markerPx = toPx(marker);
    drawVelocityVector(ctx, markerPx, marker, xAxis, yAxis, {
      color: OPERATOR_PROBE_COLOR,
      lengthPx: 30,
      unitLength: true,
    });
    ctx.save();
    ctx.fillStyle = OPERATOR_BURN_MARKER_COLOR;
    ctx.strokeStyle = "rgba(8, 11, 16, 0.95)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(markerPx.x, markerPx.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText(String(marker.burnIndex || ""), markerPx.x + 7, markerPx.y - 14);
    ctx.restore();
  });
  if (state.operatorTrajectoryProbe) {
    drawOperatorProbe(ctx, toPx(state.operatorTrajectoryProbe.state), state.operatorTrajectoryProbe.timeS);
  }
  drawOperatorStateReadout(ctx, width, height, xAxis, yAxis);
}

function drawOperatorProbe(ctx, center, timeS) {
  ctx.save();
  ctx.fillStyle = OPERATOR_PROBE_COLOR;
  ctx.strokeStyle = "rgba(8, 11, 16, 0.95)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(operatorProbeTimeLabel(timeS), center.x + 9, center.y - 18);
  ctx.restore();
}

function drawOperatorStateReadout(ctx, width, height, xAxis, yAxis) {
  const stateForReadout = state.operatorTrajectoryProbe?.state || operatorInitialSeed();
  const velocityReadout = xAxis === "c" && yAxis === "r";
  const labels = velocityReadout ? ["dR", "dI", "dC"] : ["R", "I", "C"];
  const values = velocityReadout
    ? [stateForReadout.rd * 1000, stateForReadout.id * 1000, stateForReadout.cd * 1000]
    : [stateForReadout.r, stateForReadout.i, stateForReadout.c];
  const unit = velocityReadout ? "m/s" : "km";
  ctx.save();
  ctx.fillStyle = state.operatorTrajectoryProbe ? OPERATOR_PROBE_COLOR : "rgba(162, 178, 198, 0.96)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(
    `${labels[0]} ${values[0].toFixed(2)} ${unit}   ${labels[1]} ${values[1].toFixed(2)} ${unit}   ${labels[2]} ${values[2].toFixed(2)} ${unit}`,
    12,
    height - 12,
  );
  ctx.restore();
}

function operatorProbeTimeLabel(timeS) {
  const value = Number.isFinite(Number(timeS)) ? Math.max(Number(timeS), 0) : 0;
  return `T=${value.toFixed(0)}s`;
}

function handleOperatorTrajectoryProbeClick(event, plane) {
  if (!operatorScriptModeActive()) return false;
  const canvas = plane === "ri" ? el.riCanvas : el.rcCanvas;
  if (!canvas) return false;
  const rect = canvas.getBoundingClientRect();
  if (
    event.clientX < rect.left ||
    event.clientX > rect.right ||
    event.clientY < rect.top ||
    event.clientY > rect.bottom
  ) {
    return true;
  }
  const xAxis = plane === "ri" ? "i" : "c";
  const yAxis = "r";
  const { width, height } = fitCanvas(canvas);
  const click = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
  const toPx = operatorPlotTransform(width, height, xAxis, yAxis);
  if (state.operatorTrajectoryProbe) {
    const selectedPx = toPx(state.operatorTrajectoryProbe.state);
    if (distancePx(click, selectedPx) <= OPERATOR_PROBE_PICK_RADIUS_PX) {
      state.operatorTrajectoryProbe = null;
      draw();
      updateDebugState();
      return true;
    }
  }
  const nearest = nearestOperatorTrajectoryPoint(click, toPx);
  if (nearest && nearest.distancePx <= OPERATOR_PROBE_PICK_RADIUS_PX) {
    state.operatorTrajectoryProbe = {
      state: { ...nearest.point },
      timeS: Number(nearest.point.t || nearest.point.timeS || 0),
    };
    draw();
    updateDebugState();
  }
  return true;
}

function operatorPlotTransform(width, height, xAxis, yAxis) {
  const cameraCenter = cameraCenterFor(xAxis, yAxis);
  const scale = plotScale(width, height, xAxis, yAxis, cameraCenter);
  const xSign = displayAxisSign(xAxis);
  const ySign = displayAxisSign(yAxis);
  return (p) => ({
    x: width / 2 + ((Number(p[xAxis] || 0) - cameraCenter[xAxis]) * xSign) * scale,
    y: height / 2 - ((Number(p[yAxis] || 0) - cameraCenter[yAxis]) * ySign) * scale,
  });
}

function nearestOperatorTrajectoryPoint(click, toPx) {
  let nearest = null;
  state.operatorPlanPath.forEach((point) => {
    const px = toPx(point);
    const d = distancePx(click, px);
    if (!nearest || d < nearest.distancePx) nearest = { point, distancePx: d };
  });
  return nearest;
}

function distancePx(a, b) {
  return Math.hypot(Number(a.x || 0) - Number(b.x || 0), Number(a.y || 0) - Number(b.y || 0));
}

function drawAxisDirectionLabels(ctx, width, height, xAxis, yAxis) {
  const xPositiveRight = displayAxisSign(xAxis) > 0;
  const yPositiveUp = displayAxisSign(yAxis) > 0;
  ctx.fillStyle = "rgba(170, 180, 195, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(`${xPositiveRight ? "+" : "-"}${axisLabel(xAxis)}`, width - 36, height / 2 + 22);
  ctx.fillText(`${xPositiveRight ? "-" : "+"}${axisLabel(xAxis)}`, 12, height / 2 + 22);
  ctx.save();
  ctx.fillText(`${yPositiveUp ? "+" : "-"}${axisLabel(yAxis)}`, width / 2 + 8, 24);
  ctx.fillText(`${yPositiveUp ? "-" : "+"}${axisLabel(yAxis)}`, width / 2 + 8, height - 12);
  ctx.restore();
}

function drawPrimerRic(ctx, width, height, xAxis, yAxis, stage) {
  const sample = primerSample();
  const scale = Math.min(width, height) / 2.5;
  const xSign = displayAxisSign(xAxis);
  const ySign = displayAxisSign(yAxis);
  const toPx = (p) => ({
    x: width / 2 + ((p[xAxis] || 0) * xSign) * scale,
    y: height / 2 - ((p[yAxis] || 0) * ySign) * scale,
  });

  drawGrid(ctx, width, height, scale);
  drawPrimerAxis(ctx, width, height, xAxis, yAxis, stage.axis);

  const target = toPx({ r: 0, i: 0, c: 0 });
  const chaser = toPx(sample);
  ctx.strokeStyle = "rgba(245, 205, 92, 0.38)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(target.x, target.y, 0.25 * scale, 0, Math.PI * 2);
  ctx.stroke();

  drawSpacecraftMarker(ctx, target, "target", { scale, fallbackRadius: 6, forceIcon: true });
  drawSpacecraftMarker(ctx, chaser, "chaser", { scale, fallbackRadius: 7, forceIcon: true });

  ctx.fillStyle = "rgba(245, 92, 92, 0.95)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText("Target", target.x + 10, target.y - 10);
  ctx.fillStyle = "rgba(245, 205, 92, 0.95)";
  ctx.fillText("Chaser", chaser.x + 10, chaser.y + 18);
}

function drawPrimerAxis(ctx, width, height, xAxis, yAxis, activeAxis) {
  const axisColors = {
    r: "rgba(150, 235, 170, 0.94)",
    i: "rgba(245, 205, 92, 0.94)",
    c: "rgba(96, 190, 245, 0.94)",
  };
  const xColor = activeAxis === xAxis ? axisColors[xAxis] : "rgba(90, 104, 124, 0.95)";
  const yColor = activeAxis === yAxis ? axisColors[yAxis] : "rgba(90, 104, 124, 0.95)";
  ctx.save();
  ctx.lineWidth = 2;
  if (displayAxisSign(xAxis) > 0) drawArrow(ctx, 36, height / 2, width - 36, height / 2, xColor);
  else drawArrow(ctx, width - 36, height / 2, 36, height / 2, xColor);
  if (displayAxisSign(yAxis) > 0) drawArrow(ctx, width / 2, height - 32, width / 2, 32, yColor);
  else drawArrow(ctx, width / 2, 32, width / 2, height - 32, yColor);
  ctx.font = "13px Menlo, Consolas, monospace";
  ctx.fillStyle = xColor;
  ctx.fillText(`+${axisLabel(xAxis)}`, displayAxisSign(xAxis) > 0 ? width - 54 : 24, height / 2 - 10);
  ctx.fillStyle = yColor;
  ctx.fillText(`+${axisLabel(yAxis)}`, width / 2 + 10, displayAxisSign(yAxis) > 0 ? 42 : height - 42);
  ctx.restore();
}

function drawPrimerEci(ctx, width, height, stage) {
  const center = { x: width / 2, y: height / 2 + 4 };
  const orbitScale = Math.min(width, height) * 0.34;
  const t = state.primerTimeS;
  const phase = Math.sin(t * 1.05);
  const targetTheta = -0.35;
  const radialOffset = stage.id === "radial" ? 0.14 * phase : 0;
  const phaseOffset = stage.id === "inTrack" ? 0.34 * phase : 0;
  const inclinationDeg = stage.id === "crossTrack" ? 10 * phase : 0;

  if (stage.id === "crossTrack") {
    drawCrossTrackSideView(ctx, width, height, inclinationDeg);
    return;
  }

  drawEciCircle(ctx, center, orbitScale, "rgba(96, 174, 224, 0.62)");
  drawEarth(ctx, center, orbitScale);
  const chaserRadius = 1 + radialOffset;
  const chaserTheta = targetTheta + phaseOffset;
  if (stage.id === "radial") {
    drawEciCircle(ctx, center, orbitScale * chaserRadius, "rgba(245, 205, 92, 0.66)");
  } else {
    drawEciCircle(ctx, center, orbitScale, "rgba(245, 205, 92, 0.34)", true);
  }
  const target = projectEciCircular(1, targetTheta, center, orbitScale);
  const chaser = projectEciCircular(chaserRadius, chaserTheta, center, orbitScale);
  drawSatellite(ctx, target, "target", "Target", -56, 22);
  drawSatellite(ctx, chaser, "chaser", "Chaser", 10, -14);

  if (stage.id === "radial") {
    ctx.fillStyle = "rgba(170, 184, 204, 0.92)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText(`Chaser radius ${(1 + radialOffset).toFixed(2)}x target orbit`, 18, height - 18);
  }
}

function drawEciCircle(ctx, center, radiusPx, color, dashed = false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radiusPx, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function projectEciCircular(radius, theta, center, scale) {
  return {
    x: center.x + radius * Math.cos(theta) * scale,
    y: center.y + radius * Math.sin(theta) * scale,
  };
}

function drawCrossTrackSideView(ctx, width, height, inclinationDeg) {
  const center = { x: width / 2, y: height / 2 + 6 };
  const earthRadius = Math.max(22, Math.min(width, height) * 0.07);
  const halfSpan = Math.min(width, height) * 0.36;
  const baseSlope = -0.5;
  const chaserSlope = baseSlope - inclinationDeg / 22;
  const targetLine = lineSegment(center, halfSpan, baseSlope);
  const chaserLine = lineSegment(center, halfSpan, chaserSlope);
  const target = pointAlongLine(targetLine, 0.86);
  const chaser = pointAlongLine(chaserLine, 0.86);

  drawEarth(ctx, center, earthRadius / 0.16);
  drawOrbitLine(ctx, targetLine, "rgba(96, 174, 224, 0.72)");
  drawOrbitLine(ctx, chaserLine, "rgba(245, 205, 92, 0.78)");
  drawSatellite(ctx, target, "target", "Target", -60, 24);
  drawSatellite(ctx, chaser, "chaser", "Chaser", 12, -12);

  ctx.fillStyle = "rgba(170, 184, 204, 0.92)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(`Side view: chaser inclination ${inclinationDeg.toFixed(1)} deg`, 18, height - 18);
}

function lineSegment(center, halfSpan, slope) {
  const dx = halfSpan / Math.sqrt(1 + slope * slope);
  const dy = slope * dx;
  return {
    a: { x: center.x - dx, y: center.y - dy },
    b: { x: center.x + dx, y: center.y + dy },
  };
}

function pointAlongLine(line, fraction) {
  return {
    x: line.a.x + (line.b.x - line.a.x) * fraction,
    y: line.a.y + (line.b.y - line.a.y) * fraction,
  };
}

function drawOrbitLine(ctx, line, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(line.a.x, line.a.y);
  ctx.lineTo(line.b.x, line.b.y);
  ctx.stroke();
  ctx.restore();
}

function drawEarth(ctx, center, orbitScale) {
  const radius = Math.max(24, orbitScale * 0.16);
  const gradient = ctx.createRadialGradient(center.x - radius * 0.35, center.y - radius * 0.35, 4, center.x, center.y, radius);
  gradient.addColorStop(0, "rgb(95, 170, 210)");
  gradient.addColorStop(0.65, "rgb(28, 82, 132)");
  gradient.addColorStop(1, "rgb(9, 28, 58)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(220, 240, 255, 0.55)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawSatellite(ctx, point, role, label, labelOffsetX, labelOffsetY) {
  drawSpacecraftMarker(ctx, point, role, { scale: 1, fallbackRadius: 7, forceIcon: true });
  ctx.fillStyle = "rgba(230, 235, 242, 0.95)";
  ctx.font = "12px Menlo, Consolas, monospace";
  ctx.fillText(label, point.x + labelOffsetX, point.y + labelOffsetY);
}

function satelliteMarkerSizePx(scalePxPerKm) {
  const rawPx = Math.abs(Number(scalePxPerKm || 0)) * SATELLITE_SPRITE_DIAMETER_KM;
  if (!Number.isFinite(rawPx) || rawPx <= 0) return 0;
  return Math.max(1, Math.round(rawPx));
}

function drawSpacecraftMarker(ctx, point, role, options = {}) {
  const color = role === "target" ? TARGET_MARKER : CHASER_MARKER;
  const fallbackRadius = options.fallbackRadius || 7;
  const size = options.forceIcon ? options.iconSize || SATELLITE_ICON_SIZE_PX : satelliteMarkerSizePx(options.scale);
  if (size <= 0) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, fallbackRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(245, 235, 242, 0.64)";
    ctx.lineWidth = 1;
    ctx.stroke();
    return;
  }

  ctx.save();
  ctx.translate(point.x, point.y);
  const s = size / 128;
  ctx.scale(s, s);
  ctx.translate(-64, -64);
  drawSpacecraftSprite(ctx, color);
  ctx.restore();

  const dotRadius = size < 30 ? 2 : 3;
  ctx.fillStyle = "rgba(235, 248, 255, 0.96)";
  ctx.beginPath();
  ctx.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(point.x, point.y, dotRadius + 2, 0, Math.PI * 2);
  ctx.stroke();
}

function drawSpacecraftSprite(ctx, accent) {
  ctx.save();
  ctx.shadowColor = accent;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(7, 91);
  ctx.lineTo(50, 72);
  ctx.moveTo(78, 56);
  ctx.lineTo(122, 41);
  ctx.stroke();
  ctx.shadowBlur = 0;

  drawPanel(ctx, [
    [18, 38],
    [48, 51],
    [44, 78],
    [14, 65],
  ]);
  drawPanel(ctx, [
    [80, 51],
    [112, 64],
    [106, 91],
    [76, 78],
  ]);

  ctx.fillStyle = "rgba(8, 18, 26, 0.92)";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  roundRectPath(ctx, 49, 48, 30, 32, 8);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(64, 64, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(220, 242, 250, 0.82)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(64, 64, 7, 0.2 * Math.PI, 1.8 * Math.PI);
  ctx.stroke();
  ctx.strokeStyle = "rgba(80, 110, 130, 0.72)";
  ctx.beginPath();
  ctx.moveTo(49, 58);
  ctx.lineTo(79, 58);
  ctx.moveTo(49, 69);
  ctx.lineTo(79, 69);
  ctx.stroke();
  ctx.fillStyle = "rgba(235, 248, 255, 0.94)";
  ctx.beginPath();
  ctx.arc(64, 64, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = accent;
  ctx.beginPath();
  ctx.moveTo(64, 56);
  ctx.lineTo(64, 72);
  ctx.moveTo(56, 64);
  ctx.lineTo(72, 64);
  ctx.stroke();
  ctx.restore();
}

function drawPanel(ctx, points) {
  ctx.fillStyle = "rgba(8, 30, 42, 0.76)";
  ctx.strokeStyle = "rgba(70, 190, 245, 0.82)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach(([x, y], idx) => {
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function roundRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(Math.floor(rect.width * dpr), 1);
  const height = Math.max(Math.floor(rect.height * dpr), 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rect.width, height: rect.height };
}

function cameraCenterFor(xAxis, yAxis) {
  if (state.mode === "primer" || state.mode === "tutorial" || operatorScriptModeActive() || state.mode === "operatorTutorial") {
    return { r: 0, i: 0, c: 0 };
  }
  if (state.mode === "operatorSandbox") {
    return { r: 0, i: 0, c: 0 };
  }
  if (state.mode === "arcade") {
    if (state.cameraRuleMode === "full_trajectory") return { r: 0, i: 0, c: 0 };
    return {
      r: (Number(state.sim.r || 0) + Number(state.arcadeTargetRel.r || 0)) / 2,
      i: (Number(state.sim.i || 0) + Number(state.arcadeTargetRel.i || 0)) / 2,
      c: (Number(state.sim.c || 0) + Number(state.arcadeTargetRel.c || 0)) / 2,
    };
  }
  if (state.mode === "sandbox" && new Set([xAxis, yAxis]).has("r")) {
    return { r: 0, i: 0, c: 0 };
  }
  if (new Set([xAxis, yAxis]).has("c") && new Set([xAxis, yAxis]).has("r")) {
    return { r: 0, i: 0, c: 0 };
  }
  return {
    r: state.sim.r / 2,
    i: state.sim.i / 2,
    c: state.sim.c / 2,
  };
}

function plotScale(width, height, xAxis, yAxis, cameraCenter) {
  let values;
  if (state.mode === "arcade" && state.cameraRuleMode === "current_pair") {
    values = [state.sim, state.arcadeTargetRel];
  } else if ((state.mode === "sandbox" || state.mode === "operatorSandbox") && state.cameraRuleMode === "current_pair") {
    values = [state.sim, { r: 0, i: 0, c: 0 }];
  } else {
    values = [...state.trail, ...state.ghost, ...state.tutorialTargetPath, { r: 0, i: 0, c: 0 }];
  }
  if (state.mode === "arcade") {
    if (state.cameraRuleMode === "current_pair") {
      values.push(state.arcadeTargetRel);
    } else {
      values.push(...state.targetTrail, ...state.targetGhost, state.arcadeTargetRel);
    }
  }
  if (state.mode === "tutorial") values.push({ r: 0.25, i: 0.25, c: 0.25 });
  const span = values.reduce(
    (max, p) =>
      Math.max(
        max,
        Math.abs((p[xAxis] || 0) - (cameraCenter[xAxis] || 0)),
        Math.abs((p[yAxis] || 0) - (cameraCenter[yAxis] || 0)),
      ),
    MIN_PLOT_SPAN_KM,
  );
  const padded = Math.max(span * PLOT_SCALE_MARGIN, MIN_PLOT_SPAN_KM);
  return Math.min(width, height) / (2 * padded);
}

function drawGrid(ctx, width, height, scale) {
  ctx.strokeStyle = "rgba(30, 38, 50, 0.95)";
  ctx.lineWidth = 1;
  const step = niceStep(80 / scale);
  for (let xKm = -20; xKm <= 20; xKm += step) {
    const x = width / 2 + xKm * scale;
    if (x < 0 || x > width) continue;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let yKm = -20; yKm <= 20; yKm += step) {
    const y = height / 2 - yKm * scale;
    if (y < 0 || y > height) continue;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(90, 104, 124, 0.95)";
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function niceStep(raw) {
  if (raw <= 0.05) return 0.05;
  if (raw <= 0.1) return 0.1;
  if (raw <= 0.25) return 0.25;
  if (raw <= 0.5) return 0.5;
  if (raw <= 1) return 1;
  if (raw <= 2) return 2;
  return 5;
}

function drawRings(ctx, toPx, scale, xAxis, yAxis, targetState = { r: 0, i: 0, c: 0 }) {
  const target = toPx(targetState);
  ctx.strokeStyle = "rgba(190, 68, 68, 0.72)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(target.x, target.y, 0.025 * scale, 0, Math.PI * 2);
  ctx.stroke();
  if (state.mode === "tutorial" || state.mode === "arcade") {
    const goalRange = state.mode === "arcade" ? currentArcadeGoalRangeKm() : 0.25;
    ctx.strokeStyle = "rgba(78, 178, 112, 0.86)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(target.x, target.y, goalRange * scale, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (xAxis === "i" && yAxis === "r") {
    ctx.fillStyle = "rgba(245, 205, 92, 0.9)";
    ctx.font = "12px Menlo, Consolas, monospace";
    ctx.fillText("Target", target.x + 10, target.y - 10);
  }
}

function drawPath(ctx, points, toPx, color, dashed, width = 2) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([8, 8]);
  ctx.beginPath();
  points.forEach((p, idx) => {
    const px = toPx(p);
    if (idx === 0) ctx.moveTo(px.x, px.y);
    else ctx.lineTo(px.x, px.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawVector(ctx, origin, sim, xAxis, yAxis, kind) {
  const scale = kind === "velocity" ? 75000 : 1;
  const vx = sim[`${xAxis}d`] * scale * displayAxisSign(xAxis);
  const vy = sim[`${yAxis}d`] * scale * displayAxisSign(yAxis);
  drawArrow(ctx, origin.x, origin.y, origin.x + vx, origin.y - vy, "rgba(245, 205, 92, 0.9)");
}

function drawVelocityVector(ctx, origin, sim, xAxis, yAxis, options = {}) {
  const color = options.color || "rgba(245, 205, 92, 0.9)";
  const rawX = Number(sim[`${xAxis}d`] || 0) * displayAxisSign(xAxis);
  const rawY = Number(sim[`${yAxis}d`] || 0) * displayAxisSign(yAxis);
  let vx = rawX;
  let vy = rawY;
  if (options.unitLength) {
    const norm = Math.hypot(vx, vy);
    if (!Number.isFinite(norm) || norm <= 1.0e-12) return;
    const lengthPx = Number(options.lengthPx || 30);
    vx = (vx / norm) * lengthPx;
    vy = (vy / norm) * lengthPx;
  } else {
    const scale = Number(options.scale || 75000);
    vx *= scale;
    vy *= scale;
  }
  if (Math.hypot(vx, vy) < 1) return;
  drawArrow(ctx, origin.x, origin.y, origin.x + vx, origin.y - vy, color);
}

function drawThrustVector(ctx, origin, xAxis, yAxis) {
  const u = currentControls();
  const scale = 42;
  const vx = u[xAxis] * scale * displayAxisSign(xAxis);
  const vy = u[yAxis] * scale * displayAxisSign(yAxis);
  if (Math.hypot(vx, vy) < 1) return;
  drawArrow(ctx, origin.x, origin.y, origin.x + vx, origin.y - vy, "rgba(92, 220, 160, 0.95)");
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 8 * Math.cos(angle - 0.45), y2 - 8 * Math.sin(angle - 0.45));
  ctx.lineTo(x2 - 8 * Math.cos(angle + 0.45), y2 - 8 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
}

function axisLabel(axis) {
  if (axis === "r") return "R";
  if (axis === "i") return "I";
  return "C";
}

function formatDistanceKm(valueKm, sigFigs = 4) {
  const value = Number(valueKm);
  if (!Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude >= 1.0) return `${formatSigFig(value, sigFigs)} km`;
  if (magnitude >= 1.0e-3) return `${formatSigFig(value * 1000.0, sigFigs)} m`;
  return `${formatSigFig(value * 1.0e6, sigFigs)} mm`;
}

function mobileLandscapeHudSigFigs() {
  return state.activeView === "mobile" && window.matchMedia("(orientation: landscape) and (max-height: 520px)").matches
    ? 2
    : 4;
}

function formatSpeedKmS(valueKmS, sigFigs = 4) {
  const value = Number(valueKmS);
  if (!Number.isFinite(value)) return "--";
  return formatSpeedMS(value * 1000.0, sigFigs);
}

function formatSpeedMS(valueMS, sigFigs = 4) {
  const value = Number(valueMS);
  if (!Number.isFinite(value)) return "--";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000.0) return `${formatSigFig(value / 1000.0, sigFigs)} km/s`;
  if (magnitude >= 1.0) return `${formatSigFig(value, sigFigs)} m/s`;
  return `${formatSigFig(value * 1000.0, sigFigs)} mm/s`;
}

function formatClockTime(valueS) {
  const totalSeconds = Math.max(Math.ceil(Number(valueS) || 0), 0);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatSigFig(value, sigFigs = 4) {
  const normalizedSigFigs = Math.max(Math.floor(Number(sigFigs) || 4), 1);
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  if (numeric === 0) return "0";
  const decimals = Math.max(normalizedSigFigs - Math.floor(Math.log10(Math.abs(numeric))) - 1, 0);
  return numeric.toFixed(decimals);
}

function showDebrief(passed) {
  setLeaderboardFormVisible(false);
  el.debriefPanel.classList.remove("hidden");
  el.debriefTitle.textContent = passed ? "Tutorial complete." : "Attempt ended.";
  el.debriefText.textContent = `${state.finalReason} Closest approach ${formatDistanceKm(
    state.closestKm,
  )}, delta-v ${formatSpeedMS(state.sim.dv)}.`;
  trackEvent(passed ? "tutorial_complete" : "tutorial_end", completionAnalyticsProps(passed));
}

function showArcadeDebrief() {
  if (!state.arcadeSession) return;
  const snap = state.arcadeSnapshot || state.arcadeSession.snapshot();
  const attempt = state.arcadeSession.attemptPacket({
    challengeRecord: ARCADE_CHALLENGE_RECORD,
    username: "LOCAL_PLAYER",
    client_build_hash: ARCADE_BUILD_ID,
  });
  const validation = validateAttemptPacket(attempt, ARCADE_CHALLENGE_RECORD);
  state.arcadeValidation = validation;
  state.arcadeAttemptPacket = attempt;
  prepareLeaderboardSubmission(validation);
  if ((snap.round_summaries || []).length > 0) {
    el.debriefPanel.classList.remove("hidden");
    el.debriefTitle.textContent = "Arcade run ended.";
    el.debriefText.textContent = `${snap.terminal_reason || "Attempt complete."} Total score ${Number(
      snap.score || 0,
    ).toLocaleString()} after ${(snap.round_summaries || []).length} cleared rounds. Final range ${formatDistanceKm(
      Number(snap.range_km || 0),
    )}, chaser delta-v ${formatSpeedMS(Number(snap.player_delta_v_m_s || 0))}. Local validation: ${validation.status}.`;
    trackEvent("arcade_attempt_complete", {
      result: "ended",
      validation: validation.status,
      score_bucket: snap.score > 0 ? "positive" : "zero",
    });
    return;
  }
  el.debriefPanel.classList.remove("hidden");
  const valid = validation.status === "valid";
  el.debriefTitle.textContent = snap.passed ? "Arcade rendezvous complete." : "Arcade attempt ended.";
  el.debriefText.textContent = `${snap.terminal_reason || "Attempt complete."} Local validation: ${
    validation.status
  }. Score ${validation.canonical_score || 0}, closest ${formatDistanceKm(
    validation.canonical_metrics?.closest_range_km || snap.closest_range_km || 0,
  )}, delta-v ${formatSpeedMS(
    Number(validation.canonical_metrics?.player_delta_v_m_s || snap.player_delta_v_m_s || 0),
  )}.${valid ? "" : ` ${validation.errors.join(" ")}`}`;
  trackEvent("arcade_attempt_complete", {
    result: snap.passed ? "success" : "ended",
    validation: validation.status,
    score_bucket: validation.canonical_score > 0 ? "positive" : "zero",
  });
}

function setLeaderboardFormVisible(visible, statusText = "") {
  if (!el.leaderboardForm) return;
  el.leaderboardForm.classList.toggle("hidden", !visible);
  el.leaderboardStatus.textContent = statusText;
  el.leaderboardSubmit.disabled = false;
}

function prepareLeaderboardSubmission(validation) {
  const canSubmit = ["valid", "suspicious"].includes(validation.status);
  setLeaderboardFormVisible(state.mode === "arcade", canSubmit ? "" : "This attempt cannot be submitted.");
  el.leaderboardSubmit.disabled = !canSubmit;
}

async function submitLeaderboardAttempt(event) {
  event.preventDefault();
  if (!state.arcadeAttemptPacket || !state.arcadeValidation) return;
  if (!["http:", "https:"].includes(window.location.protocol)) {
    el.leaderboardStatus.textContent = "Leaderboard submission is available after the hosted deploy.";
    return;
  }
  const username = String(el.leaderboardUsername.value || "LOCAL_PLAYER").trim() || "LOCAL_PLAYER";
  const email = String(el.leaderboardEmail.value || "").trim();
  const attempt = { ...state.arcadeAttemptPacket, username, email };
  el.leaderboardSubmit.disabled = true;
  el.leaderboardStatus.textContent = "Submitting...";
  try {
    const response = await fetch("/api/submit-attempt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, attempt }),
    });
    const payload = await response.json();
    if (!response.ok) {
      el.leaderboardStatus.textContent = payload.status
        ? `Rejected: ${payload.status}. ${(payload.errors || []).join(" ")}`
        : `Submission failed: ${payload.error || response.statusText}`;
      trackEvent("arcade_leaderboard_submit", { result: "rejected", status: payload.status || response.status });
      return;
    }
    const emailNote =
      email && payload.email_status === "sent"
        ? " Verification email sent."
        : email && payload.email_status === "not_configured"
          ? " Email receipts are not configured yet."
          : email && payload.email_status === "failed"
            ? ` Email failed: ${payload.email_error || "provider error"}.`
            : "";
    const ownershipNote =
      payload.ownership_status === "pending_verification"
        ? " Verify your email to reserve this username."
        : payload.ownership_status === "verified_owner"
          ? " Username verified."
          : payload.ownership_status === "locked"
            ? " Username reserved; attempt saved but leaderboard not updated."
            : "";
    el.leaderboardStatus.textContent = payload.leaderboard_updated
      ? `Submitted. Score ${Number(payload.score || 0).toLocaleString()} is on the leaderboard.`
      : `Submitted. Score ${Number(payload.score || 0).toLocaleString()} did not beat your best.`;
    el.leaderboardStatus.textContent += emailNote + ownershipNote;
    refreshLeaderboard({ force: true });
    trackEvent("arcade_leaderboard_submit", { result: "accepted", status: payload.status || "valid" });
  } catch (error) {
    el.leaderboardStatus.textContent = `Submission failed: ${error instanceof Error ? error.message : String(error)}`;
    trackEvent("arcade_leaderboard_submit", { result: "error" });
  } finally {
    el.leaderboardSubmit.disabled = false;
  }
}

function showArcadeRoundTransition() {
  const tr = state.arcadeTransition;
  if (!tr) return;
  setLeaderboardFormVisible(false);
  el.debriefPanel.classList.remove("hidden");
  el.debriefTitle.textContent = `Round ${tr.cleared_round_index} cleared.`;
  el.debriefText.textContent = `Round score ${tr.round_score.toLocaleString()}. Total score ${tr.total_score.toLocaleString()}. Bonus time ${Math.round(
    tr.bonus_time_s,
  )} s. Clear range ${formatDistanceKm(Number(tr.clear_range_km || 0))} against a ${formatDistanceKm(
    Number(tr.goal_range_km || 0),
  )} goal. Round ${tr.next_round_index}${tr.next_is_boss ? " is a boss round" : ""} starts with ${Math.round(
    tr.next_time_budget_s,
  )} s and a ${formatDistanceKm(tr.next_goal_range_km)} goal.`;
}

function completionAnalyticsProps(passed) {
  return {
    result: passed ? "success" : "ended",
    time_bucket: bucketSeconds(state.sim.t),
    dv_bucket: bucketDvMps(state.sim.dv),
    closest_range_bucket: bucketRangeKm(state.closestKm),
  };
}

function bucketSeconds(value) {
  if (value < 60) return "under_1min";
  if (value < 300) return "1_5min";
  if (value < 600) return "5_10min";
  if (value < 1200) return "10_20min";
  return "20min_plus";
}

function bucketDvMps(value) {
  if (value < 2) return "under_2mps";
  if (value < 5) return "2_5mps";
  if (value < 10) return "5_10mps";
  return "10mps_plus";
}

function bucketRangeKm(value) {
  if (value < 0.05) return "under_50m";
  if (value < 0.1) return "50_100m";
  if (value < 0.25) return "100_250m";
  if (value < 1) return "250m_1km";
  return "1km_plus";
}

function frame(nowMs) {
  const cameraAdjusting = sandbox3d.interacting();
  state.cameraWasAdjusting = cameraAdjusting;
  if (!state.lastFrameMs) state.lastFrameMs = nowMs;
  const wallElapsedS = Math.max((nowMs - state.lastFrameMs) / 1000, 0);
  const elapsedS = Math.min(wallElapsedS, 0.2);
  state.lastFrameMs = nowMs;
  if (state.mode === "selector") {
    updateDebugState();
    queueFrame(frame);
    return;
  }
  if (state.mode === "primer") {
    state.primerTimeS += elapsedS;
    updateGhost();
    draw();
    queueFrame(frame);
    return;
  }
  const controls = currentControls();
  const pendingPulse = hasPendingControlPulse();
  const shouldRun = simulationShouldRun();
  if (operatorModeActive() && !state.passed) {
    updateOperatorBurnCinematic(nowMs, shouldRun ? elapsedS * currentSpeedMultiple() : 0);
  }
  // The timing helper below owns accumulation, bounding, and accounting.
  // Keep the pending fraction across frames instead of discarding it when the
  // bounded per-frame work budget is reached.
  const stepDtS = currentStepDtS(controls);
  let pendingSimulationS = state.stepAccumulatorS;
  if (shouldRun && !state.passed && pendingPulse && hasManeuverInput(controls) && pendingSimulationS < stepDtS) {
    pendingSimulationS = stepDtS;
  }
  const timingPlan = advanceFrameTiming(state.frameTiming, {
    pendingSimulationS,
    elapsedS,
    wallElapsedS,
    speedMultiple: effectiveSpeedMultiple(controls),
    stepDtS,
    maxStepsPerFrame: MAX_STEPS_PER_FRAME,
    maxPendingSimulationS: MAX_PENDING_SIMULATION_S,
    shouldRun: shouldRun && !state.passed,
  });
  state.stepAccumulatorS = timingPlan.pendingSimulationS;
  const timingResetToken = state.timingResetToken;
  let actualSteps = 0;
  for (let steps = 0; steps < timingPlan.steps; steps += 1) {
    if (state.passed || state.timingResetToken !== timingResetToken) break;
    step(stepDtS, shouldRun);
    actualSteps += 1;
    if (state.passed || state.timingResetToken !== timingResetToken) break;
  }
  if (state.timingResetToken === timingResetToken) {
    const execution = reconcileFrameTimingExecution(state.frameTiming, {
      plannedSteps: timingPlan.steps,
      actualSteps,
      stepDtS,
    });
    state.stepAccumulatorS = execution.pendingSimulationS;
  }
  if (timingPlan.steps > 0 || pendingPulse) clearControlPulses();
  updateGhost();
  draw();
  queueFrame(frame);
}

function queueFrame(callback) {
  if (typeof window.requestAnimationFrame === "function") {
    window.requestAnimationFrame(callback);
    return;
  }
  window.setTimeout(() => callback(Date.now()), 16);
}

function togglePause() {
  if (advancePrimer()) return;
  if (state.passed) return;
  if (state.mode === "arcade" && state.arcadeTransition && state.arcadeSession) {
    state.arcadeSession.continueNextRound();
    state.speedIndex = speedOptionIndex(1);
    state.cameraRuleMode = "full_trajectory";
    resetSimulationTiming();
    state.arcadeTransition = null;
    el.debriefPanel.classList.add("hidden");
    syncArcadeSnapshot();
    state.running = true;
    updateMissionText();
    return;
  }
  if (state.mode === "arcade") {
    if (!state.running) state.running = true;
    updateMissionText();
    return;
  }
  if (operatorScriptModeActive()) {
    launchOperatorPlayback();
    return;
  }
  if (operatorModeActive()) {
    if (!state.running && state.operatorPlanError) {
      renderOperatorPlanStatus();
      return;
    }
    state.running = !state.running;
    updateMissionText();
    return;
  }
  state.running = !state.running;
  updateMissionText();
}

function toggleCameraRuleMode() {
  if (state.mode !== "sandbox" && state.mode !== "operatorSandbox" && state.mode !== "arcade") return;
  state.cameraRuleMode = state.cameraRuleMode === "full_trajectory" ? "current_pair" : "full_trajectory";
  updateGhost();
  syncViewButtons();
  draw();
}

function resetCurrent() {
  if (state.mode === "primer") {
    state.primerStage = 0;
    state.primerTimeS = 0;
    resetState(primerSample());
  } else if (state.mode === "tutorial") {
    state.activeStage = 0;
    state.speedIndex = 0;
    resetState(presets.behind);
  } else if (state.mode === "operatorTutorial") {
    state.speedIndex = 0;
    setMode("operatorScriptTutorial");
    return;
  } else if (state.mode === "operatorSandbox") {
    setMode("operatorScriptSandbox");
    return;
  } else if (state.mode === "operatorScriptTutorial") {
    showLevelSelector({ track: true, source: "operator_tutorial_script_cancel" });
    return;
  } else if (state.mode === "operatorScriptSandbox") {
    showLevelSelector({ track: true, source: "operator_script_cancel" });
    return;
  } else if (state.mode === "arcade") {
    startArcadeSession();
  } else {
    resetState(sandboxSeed());
  }
  state.running = false;
  updateMissionText();
}

function bindCommandButton(button, handler) {
  if (!button) return;
  let suppressClickUntil = 0;
  button.addEventListener("pointerdown", (event) => {
    if (typeof event.button === "number" && event.button !== 0) return;
    event.preventDefault();
    suppressClickUntil = Date.now() + 500;
    handler();
  });
  button.addEventListener("click", (event) => {
    if (Date.now() < suppressClickUntil) {
      event.preventDefault();
      return;
    }
    handler();
  });
}

function suppressMobileSelectionEvents() {
  const shouldSuppress = (event) => {
    if (state.activeView !== "mobile") return false;
    if (state.mode === "selector" || state.mode === "sandboxSetup") return false;
    if (isEditableControlTarget(event.target)) return false;
    return true;
  };
  ["selectstart", "dragstart", "contextmenu"].forEach((type) => {
    document.addEventListener(
      type,
      (event) => {
        if (!shouldSuppress(event)) return;
        event.preventDefault();
      },
      { capture: true },
    );
  });
}

function bindEvents() {
  suppressMobileSelectionEvents();
  const touchPointers = new Map();
  const clearTouchPointer = (event) => {
    const entry = touchPointers.get(event.pointerId);
    if (!entry) return;
    event.preventDefault();
    touchPointers.delete(event.pointerId);
    const stillPressed = [...touchPointers.values()].some((candidate) => candidate.button === entry.button);
    if (stillPressed) return;
    touch.delete(entry.value);
    entry.button.classList.remove("active");
    delete entry.button.dataset.pressed;
    refreshInputState();
  };
  const clearManeuverInputs = () => {
    keys.clear();
    keyPulses.clear();
    touch.clear();
    touchPulses.clear();
    touchPointers.clear();
    document.querySelectorAll("[data-touch]").forEach((button) => {
      button.classList.remove("active");
      delete button.dataset.pressed;
    });
    refreshInputState();
  };
  window.addEventListener("blur", clearManeuverInputs);
  window.addEventListener("pointerup", clearTouchPointer, { capture: true });
  window.addEventListener("pointercancel", clearTouchPointer, { capture: true });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") clearManeuverInputs();
  });
  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (key === "escape") {
      event.preventDefault();
      if (state.mode !== "selector") showLevelSelector({ track: true, source: "keyboard" });
      return;
    }
    if (isEditableControlTarget(event.target)) return;
    if (
      [
        "w",
        "a",
        "s",
        "d",
        "arrowleft",
        "arrowright",
        "arrowup",
        "arrowdown",
        " ",
        "enter",
        "r",
        "m",
        "c",
        "v",
      ].includes(key)
    ) {
      event.preventDefault();
    }
    if (key === "v") {
      cycleViewPreference();
      return;
    }
    if (key === "m") {
      toggleMusic();
      return;
    }
    if (state.mode === "selector") {
      if (key === "arrowdown" || key === "s") selectLevel(state.selectedLevel + 1);
      else if (key === "arrowup" || key === "w") selectLevel(state.selectedLevel - 1);
      else if (key === " " || key === "enter") launchSelectedLevel("keyboard");
      return;
    }
    if (state.mode === "primer") {
      playMusicFromGesture();
      if (key === " " || key === "enter" || key === "arrowright") advancePrimer();
      else if (key === "r") resetCurrent();
      return;
    }
    if (operatorScriptModeActive() && key === "enter") {
      playMusicFromGesture();
      launchOperatorPlayback();
      return;
    }
    if (key === "c" && (state.mode === "sandbox" || state.mode === "operatorSandbox" || state.mode === "arcade")) {
      playMusicFromGesture();
      toggleCameraRuleMode();
      return;
    } else {
      playMusicFromGesture();
    }
    if (key === " ") togglePause();
    else if (key === "r") resetCurrent();
    else if (key === "arrowup") {
      state.speedIndex = Math.min(state.speedIndex + 1, SPEED_OPTIONS.length - 1);
      maybeCompleteSpeedStage();
    } else if (key === "arrowdown") {
      state.speedIndex = Math.max(state.speedIndex - 1, 0);
      maybeCompleteSpeedStage();
    } else {
      keys.add(key);
      keyPulses.add(key);
      refreshInputState();
    }
    if (key === "arrowup" || key === "arrowdown" || key === " " || key === "r") refreshInputState();
  });
  document.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (keys.delete(key) || ["w", "a", "s", "d", "arrowleft", "arrowright"].includes(key)) {
      refreshInputState();
    }
  });
  document.querySelectorAll("[data-touch]").forEach((button) => {
    const value = button.dataset.touch;
    const start = (event) => {
      event.preventDefault();
      if (button.disabled) return;
      playMusicFromGesture();
      touchPointers.set(event.pointerId, { button, value });
      button.setPointerCapture?.(event.pointerId);
      button.dataset.pressed = "true";
      button.classList.add("active");
      touch.add(value);
      touchPulses.add(value);
      refreshInputState();
    };
    button.addEventListener("pointerdown", start);
    button.addEventListener("lostpointercapture", clearTouchPointer);
  });
  bindCommandButton(el.pauseButton, () => {
    playMusicFromGesture();
    togglePause();
  });
  bindCommandButton(el.resetButton, () => {
    playMusicFromGesture();
    resetCurrent();
  });
  bindCommandButton(el.levelSelectButton, () => {
    playMusicFromGesture();
    showLevelSelector({ track: true, source: "button" });
  });
  bindCommandButton(el.musicButton, toggleMusic);
  bindCommandButton(el.selectorMusicButton, toggleMusic);
  bindCommandButton(el.selectorPlayButton, () => {
    launchSelectedLevel("selector_play_button");
  });
  bindCommandButton(el.selectorModeButton, toggleSelectorPlayMode);
  el.selectorFrameButtons.forEach((button) => bindCommandButton(button, toggleFrameConvention));
  bindCommandButton(el.operatorAddBurn, addOperatorBurnRow);
  bindCommandButton(el.equationSheetButton, toggleEquationSheet);
  bindCommandButton(el.equationSheetClose, toggleEquationSheet);
  bindCommandButton(el.viewButton, () => {
    if (state.activeView === "mobile") {
      playMusicFromGesture();
      toggleCameraRuleMode();
      return;
    }
    cycleViewPreference();
  });
  bindCommandButton(el.selectorViewButton, cycleViewPreference);
  [el.riPanel, el.rcPanel].forEach((panel) => {
    if (!panel) return;
    let suppressClickUntil = 0;
    const toggle = (event) => {
      if (operatorScriptModeActive()) {
        event.preventDefault();
        event.stopPropagation();
        handleOperatorTrajectoryProbeClick(event, panel === el.riPanel ? "ri" : "rc");
        return;
      }
      if (state.mode !== "sandbox" && state.mode !== "operatorSandbox" && state.mode !== "arcade") return;
      if (state.activeView === "mobile") return;
      event.preventDefault();
      event.stopPropagation();
      playMusicFromGesture();
      toggleCameraRuleMode();
    };
    const suppressFollowUp = (event) => {
      event.preventDefault();
      event.stopPropagation();
    };
    panel.addEventListener("pointerdown", (event) => {
      if (typeof event.button === "number" && event.button !== 0) return;
      suppressClickUntil = Date.now() + 700;
      toggle(event);
    });
    panel.addEventListener("mousedown", (event) => {
      if (typeof event.button === "number" && event.button !== 0) return;
      if (Date.now() < suppressClickUntil) {
        suppressFollowUp(event);
        return;
      }
      suppressClickUntil = Date.now() + 700;
      toggle(event);
    });
    panel.addEventListener("click", (event) => {
      if (Date.now() < suppressClickUntil) {
        suppressFollowUp(event);
        return;
      }
      toggle(event);
    });
  });
  el.mobileSpeedButtons.forEach((button) => {
    button.addEventListener("click", () => {
      playMusicFromGesture();
      state.speedIndex = speedOptionIndex(Number(button.dataset.mobileSpeed || currentSpeedMultiple()));
      maybeCompleteSpeedStage();
      refreshInputState();
    });
  });
  document.querySelectorAll("[data-level-option]").forEach((button) => {
    button.addEventListener("pointerenter", () => {
      if (button.hidden) return;
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
    });
    button.addEventListener("focus", () => {
      if (button.hidden) return;
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
    });
    button.addEventListener("click", () => {
      if (button.hidden) return;
      const idx = levelOptions.findIndex((option) => option.id === button.dataset.levelOption);
      if (idx >= 0) selectLevel(idx);
      if (state.activeView === "mobile") return;
      launchSelectedLevel("selector_click");
    });
  });
  el.downloadLink.addEventListener("click", () => {
    trackEvent("download_click", { source: "debrief", mode: state.mode });
  });
  el.selectorInstallLink.addEventListener("click", () => {
    trackEvent("download_click", { source: "selector", mode: state.playMode });
  });
  el.leaderboardForm.addEventListener("submit", submitLeaderboardAttempt);
  el.leaderboardRefresh.addEventListener("click", () => refreshLeaderboard({ force: true }));
  el.sandboxSetupInputs.forEach((input) => input.addEventListener("input", syncSandboxSetupValidation));
  el.sandboxSetupForm.addEventListener("submit", (event) => {
    event.preventDefault();
    playMusicFromGesture();
    if (sandboxPortraitSetupActive() && el.sandboxSetupScreen.dataset.sandboxStep === "target") {
      advanceSandboxSetupStep();
      return;
    }
    continueSandboxSetup();
  });
  el.sandboxSetupNext.addEventListener("click", advanceSandboxSetupStep);
  el.sandboxSetupBack.addEventListener("click", () => {
    setSandboxSetupStep("target", { focus: true });
    syncSandboxSetupValidation();
  });
  el.sandboxSetupStepButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.sandboxSetupStep === "chaser") advanceSandboxSetupStep();
      else {
        setSandboxSetupStep("target", { focus: true });
        syncSandboxSetupValidation();
      }
    });
  });
  el.sandboxSetupCancel.addEventListener("click", () => {
    showLevelSelector({ track: true, source: "sandbox_setup_cancel" });
  });
  window.addEventListener("resize", draw);
  const viewQuery = window.matchMedia("(max-width: 760px), (max-height: 620px)");
  const handleViewQueryChange = () => {
    if (state.viewPreference === "auto") applyViewPreference();
  };
  if (viewQuery.addEventListener) viewQuery.addEventListener("change", handleViewQueryChange);
  else if (viewQuery.addListener) viewQuery.addListener(handleViewQueryChange);
}

bindEvents();
initializeViewPreference();
syncTouchControlLabels();
initAnalytics();
trackEventOnce("preview_view", { build: BUILD_ID });
if (!launchInitialLevelFromUrl()) showLevelSelector();
queueFrame(frame);
