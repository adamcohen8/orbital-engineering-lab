# ruff: noqa: F401,F821,I001
from __future__ import annotations

import sys

import os
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.relative_th_ekf import ya_closed_form_transition_matrix
from sim.game.fonts import game_font
from sim.game.formatting import format_distance_km, format_speed_km_s, format_speed_m_s
from sim.game.frame_convention import (
    FRAME_CONVENTION_PRESET_OEL_DEFAULT,
    FRAME_CONVENTION_PRESET_SPACE_FORCE,
    FrameConvention,
    frame_convention_display_axis_sign,
    frame_convention_from_preset,
    frame_convention_preset,
    frame_convention_to_yaml,
    normalize_frame_convention,
)
from sim.game.operator import OperatorBurn, OperatorBurnPlan, parse_operator_burn_plan, validate_operator_burn_plan
from sim.game.presentation import PRESENTATION_MODES, normalize_presentation_mode
from sim.game.pygame_dashboard import (
    CHASER_SPRITE_PATH,
    ELLIPTIC_PREDICTION_MODELS,
    MIN_PLOT_SPAN_KM,
    PLOT_OVERLAY_MARGIN,
    TARGET_SPRITE_PATH,
    PygameRPODashboard,
    _coast_prediction_model_key,
    _cr3bp_projection_mode_key,
    _cw_coast_states,
    _cylinder_projection_polygon_ric,
    _finite_projected_region_bounds,
    _game_asset_path_or_default,
    _linearized_cr3bp_moon_ric_coast_prediction,
    _new_history_ring,
    _nonlinear_cr3bp_moon_ric_coast_prediction,
    _plane_key_for_axes,
    _region_visible_on_plane,
    _satellite_pair_camera_center,
    _sphere_projection_polygon_ric,
)
from sim.game.training import (
    OPERATOR_RELAXED_REQUIRED_BURN_AXIS_SCENARIO_IDS,
    RPOTrainingConfig,
    nmt_curve_points_km,
    training_config_for_game_mode,
)

GAME_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LAUNCHER_MUSIC_PATH = Path(__file__).resolve().parent / "music" / "01_insert_coin_to_orbit.wav"
START_SCREEN_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "OEL_RPO_Trainer.png"
GAME_PROGRESS_PATH_ENV = "OEL_GAME_PROGRESS_PATH"
GAME_SETTINGS_PATH_ENV = "OEL_GAME_SETTINGS_PATH"
DIFFICULTY_OPTIONS: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
GAME_MODE_OPTIONS: tuple[str, ...] = ("pilot", "operator")
DOWNLOADABLE_GAME_EXCLUDED_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "rpo_arcade_pursuit",
    }
)
OPERATOR_MODE_EXCLUDED_SCENARIO_IDS: frozenset[str] = frozenset(
    {
        "rpo_10_defensive_target_demo",
    }
)
OPTION_X = 54
OPTION_Y = 136
OPTION_WIDTH = 398
OPTION_HEIGHT = 64
OPTION_ROW_HEIGHT = 78
PANEL_TOP = 124
MIN_PANEL_HEIGHT = 480
FOOTER_HEIGHT = 76
FOOTER_BOTTOM_MARGIN = 22
PREVIEW_PADDING = 20
PREVIEW_LINE_HEIGHT = 18
PREVIEW_SECTION_TITLE_GAP = 22
PREVIEW_SECTION_GAP = 10
PREVIEW_SCROLL_STEP_PX = PREVIEW_LINE_HEIGHT * 3
CLEAR_PROGRESS_RECT = (846, 36, 150, 30)
MUSIC_RECT = (518, 36, 144, 30)
RECORD_VIDEO_RECT = (682, 36, 144, 30)
MODE_TOGGLE_WIDTH = 148
MODE_TOGGLE_HEIGHT = 34
MODE_TOGGLE_RIGHT_MARGIN = 42
SETTINGS_BUTTON_SIZE = 34
MODE_SETTINGS_GAP = 10
OPERATOR_BURN_MAX_ROWS = 24
OPERATOR_BURN_HEADERS: tuple[str, ...] = ("T (s)", "R (m/s)", "I (m/s)", "C (m/s)")
OPERATOR_BURN_ROW_HEIGHT = 36
OPERATOR_BURN_TABLE_MIN_VISIBLE_ROWS = 2
OPERATOR_BURN_MARKER_COLOR = (255, 154, 64)
FRAME_DIALOG_WIDTH = 620
FRAME_DIALOG_HEIGHT = 500
FRAME_DIALOG_CHOICE_HEIGHT = 76
GRAPHICS_MODE_LABELS: dict[str, str] = {
    "compatibility": "Legacy",
    "standard": "Standard",
    "high_refresh": "High Refresh",
    "auto": "Auto",
}


def _launcher_panel_height(screen_height: int) -> int:
    return max(int(screen_height) - PANEL_TOP - FOOTER_HEIGHT, MIN_PANEL_HEIGHT)


def _visible_option_count(screen_height: int) -> int:
    list_bottom = PANEL_TOP + _launcher_panel_height(screen_height)
    return max(1, int((list_bottom - OPTION_Y) // OPTION_ROW_HEIGHT))


def _preview_bounds(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    panel_height = _launcher_panel_height(screen_height)
    return (490, PANEL_TOP, max(int(screen_width) - 532, 420), panel_height)


def _frame_convention_dialog_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    width = min(FRAME_DIALOG_WIDTH, max(int(screen_width) - 80, 360))
    height = min(FRAME_DIALOG_HEIGHT, max(int(screen_height) - 80, 360))
    return (
        max((int(screen_width) - width) // 2, 0),
        max((int(screen_height) - height) // 2, 0),
        width,
        height,
    )


def _frame_convention_dialog_choice_rects(
    screen_width: int,
    screen_height: int,
) -> dict[str, tuple[int, int, int, int]]:
    x, y, w, _h = _frame_convention_dialog_rect(screen_width, screen_height)
    button_w = max(w - 84, 260)
    button_h = FRAME_DIALOG_CHOICE_HEIGHT
    return {
        FRAME_CONVENTION_PRESET_OEL_DEFAULT: (x + 42, y + 122, button_w, button_h),
        FRAME_CONVENTION_PRESET_SPACE_FORCE: (x + 42, y + 212, button_w, button_h),
    }


def _frame_convention_dialog_checkbox_rect(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    x, y, _w, h = _frame_convention_dialog_rect(screen_width, screen_height)
    return (x + 42, y + h - 74, 22, 22)


def _frame_convention_dialog_graphics_rects(
    screen_width: int,
    screen_height: int,
) -> dict[str, tuple[int, int, int, int]]:
    x, y, w, _h = _frame_convention_dialog_rect(screen_width, screen_height)
    choices_x = x + 150
    choices_w = max(w - 192, 240)
    gap = 8
    button_w = max((choices_w - gap * (len(PRESENTATION_MODES) - 1)) // len(PRESENTATION_MODES), 48)
    return {
        mode: (choices_x + index * (button_w + gap), y + 314, button_w, 34)
        for index, mode in enumerate(PRESENTATION_MODES)
    }


def _frame_convention_dialog_continue_rect(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = _frame_convention_dialog_rect(screen_width, screen_height)
    return (x + w - 178, y + h - 80, 136, 34)


def _settings_button_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    x = max(int(screen_width) - MODE_TOGGLE_RIGHT_MARGIN - SETTINGS_BUTTON_SIZE, 0)
    y = max(int(screen_height) - FOOTER_BOTTOM_MARGIN - SETTINGS_BUTTON_SIZE, PANEL_TOP)
    return (x, y, SETTINGS_BUTTON_SIZE, SETTINGS_BUTTON_SIZE)


def _mode_toggle_rect(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    settings_x, settings_y, _settings_w, _settings_h = _settings_button_rect(screen_width, screen_height)
    x = max(settings_x - MODE_SETTINGS_GAP - MODE_TOGGLE_WIDTH, 0)
    return (x, settings_y, MODE_TOGGLE_WIDTH, MODE_TOGGLE_HEIGHT)


def _completed_difficulties_from_game(game: dict[str, Any]) -> tuple[str, ...]:
    progress = dict(game.get("progress", {}) or {})
    completed = progress.get("completed_difficulties", ())
    if isinstance(completed, str):
        completed = (completed,)
    values = {_normalize_difficulty(item) for item in completed}
    return tuple(item for item in DIFFICULTY_OPTIONS if item in values)


def _high_score_from_game(game: dict[str, Any]) -> int:
    progress = dict(game.get("progress", {}) or {})
    return max(int(progress.get("high_score", 0) or 0), 0)


def _normalize_difficulty(value: Any) -> str:
    key = str(value or "easy").strip().lower()
    if key == "normal":
        return "medium"
    if key == "expert":
        return "extreme"
    if key in DIFFICULTY_OPTIONS:
        return key
    return "easy"


def _normalize_game_mode(value: Any) -> str:
    key = str(value or "pilot").strip().lower()
    if key in {"operator", "op", "script", "scripted"}:
        return "operator"
    return "pilot"


def _toggle_game_mode(value: Any) -> str:
    return "operator" if _normalize_game_mode(value) == "pilot" else "pilot"


def _progress_stars(completed_difficulties: tuple[str, ...]) -> str:
    highest = -1
    for difficulty in completed_difficulties:
        if difficulty in DIFFICULTY_OPTIONS:
            highest = max(highest, DIFFICULTY_OPTIONS.index(difficulty))
    earned = highest + 1
    return "★" * earned + "☆" * (len(DIFFICULTY_OPTIONS) - earned)


def _format_high_score(score: int) -> str:
    value = int(max(score, 0))
    return f"{value:,}" if value > 0 else "--"

def _budget_line(option: GameScenarioOption) -> str:
    parts = []
    if option.time_budget_s is not None:
        parts.append(f"Time: {option.time_budget_s:.0f}s")
    if option.delta_v_budget_m_s is not None:
        parts.append(f"Chaser dV: {format_speed_m_s(option.delta_v_budget_m_s)}")
    if option.goal_speed_km_s is not None:
        parts.append(f"Speed Gate: {format_speed_km_s(option.goal_speed_km_s)}")
    if option.target_delta_v_budget_m_s is not None:
        parts.append(f"Target dV: {format_speed_m_s(option.target_delta_v_budget_m_s)}")
    return "   ".join(parts)


def _wrapped_budget_lines(option: GameScenarioOption, font: Any, width_px: int) -> list[str]:
    return _wrap_text_px(_budget_line(option), font, width_px)

def _wrap_text(value: str, max_chars: int) -> list[str]:
    words = str(value or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _wrap_text_px(value: str, font: Any, width_px: int) -> list[str]:
    words = str(value or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if _text_width(font, candidate) <= width_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = _fit_text_px(word, font, width_px) if _text_width(font, word) > width_px else word
    if current:
        lines.append(current)
    return lines or [""]


def _lines_that_fit(lines: list[str], font: Any, width_px: int, y: int, max_y: int) -> list[str]:
    if y >= max_y:
        return []
    available = max((max_y - y) // PREVIEW_LINE_HEIGHT, 0)
    if available <= 0:
        return []
    if len(lines) <= available:
        return lines
    kept = lines[:available]
    kept[-1] = _fit_text_px("...", font, width_px)
    return kept


def _fit_text_px(value: str, font: Any, width_px: int) -> str:
    text = " ".join(str(value or "").split())
    if _text_width(font, text) <= width_px:
        return text
    ellipsis = "..."
    if _text_width(font, ellipsis) > width_px:
        return ""
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if _text_width(font, candidate) <= width_px:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis


def _text_width(font: Any, text: str) -> int:
    if hasattr(font, "size"):
        return int(font.size(str(text))[0])
    surf = font.render(str(text), True, (255, 255, 255))
    if hasattr(surf, "get_width"):
        return int(surf.get_width())
    return len(str(text)) * 8


def _text_height(font: Any) -> int:
    if hasattr(font, "get_height"):
        return int(font.get_height())
    if hasattr(font, "size"):
        return int(font.size("Hg")[1])
    return PREVIEW_LINE_HEIGHT


def _text(screen: Any, font: Any, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, pos)


def _text_centered(screen: Any, font: Any, text: str, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not text:
        return
    surf = font.render(str(text), True, color)
    screen.blit(surf, (int(center[0] - surf.get_width() // 2), int(center[1] - surf.get_height() // 2)))


def _preview_content_height(option: GameScenarioOption, *, font: Any, small_font: Any, width_px: int) -> int:
    y = 0
    y += 34
    y += len(_wrapped_budget_lines(option, small_font, width_px)) * PREVIEW_LINE_HEIGHT
    y += PREVIEW_SECTION_GAP
    if _show_progress_text(option) and option.high_score > 0:
        y += PREVIEW_LINE_HEIGHT + PREVIEW_SECTION_GAP
    y = _section_height(option.learning_goal, small_font, y, width_px)
    y = _section_height(option.player_brief or option.description, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.pass_criteria, small_font, y + PREVIEW_SECTION_GAP, width_px)
    y = _bullets_height(option.instructor_notes, small_font, y + PREVIEW_SECTION_GAP, width_px)
    return max(y, _text_height(font))


def _section_height(body: str, font: Any, y: int, width_px: int) -> int:
    y += PREVIEW_SECTION_TITLE_GAP
    y += len(_wrap_text_px(body, font, width_px)) * PREVIEW_LINE_HEIGHT
    return y


def _show_progress_text(option: GameScenarioOption) -> bool:
    return str(option.scenario_id) != "rpo_00_tutorial"


def _bullets_height(items: tuple[str, ...], font: Any, y: int, width_px: int) -> int:
    if not items:
        return y
    y += PREVIEW_SECTION_TITLE_GAP
    bullet_width = max(width_px - _text_width(font, "- "), 1)
    for item in items:
        y += len(_wrap_text_px(item, font, bullet_width)) * PREVIEW_LINE_HEIGHT
    return y

def _pos_in_bounds(pos: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    px, py = pos
    x, y, w, h = bounds
    return x <= px <= x + w and y <= py <= y + h

def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)].rstrip() + "..."


def _launcher_dep(name, default):
    facade = sys.modules.get("sim.game.launcher")
    return getattr(facade, name, default)


__all__ = [name for name in globals() if not name.startswith("__")]
