from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PLOT_RECIPE_SCHEMA_VERSION = 1
PLOT_RECIPE_MATURITY_LEVELS = frozenset({"supported", "prototype", "experimental"})


@dataclass(frozen=True)
class ReviewPlotRecipe:
    """One discoverable, versioned plot recipe over review-store evidence."""

    recipe_id: str
    title: str
    description: str
    sql: str
    x_column: str
    y_columns: tuple[str, ...]
    plot_type: str = "line"
    group_column: str = ""
    x_label: str = ""
    y_label: str = ""
    artifact_id: str = ""
    maturity: str = "supported"
    semantic_metric_names: tuple[str, ...] = ()
    supported_tables: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    renderer_id: str = "generic"
    recipe_version: int = 1
    natural_language_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in PLOT_RECIPE_MATURITY_LEVELS:
            allowed = ", ".join(sorted(PLOT_RECIPE_MATURITY_LEVELS))
            raise ValueError(f"Unknown review plot recipe maturity {self.maturity!r}; expected one of: {allowed}")
        if not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Review plot recipe {self.recipe_id!r} must use read-only SELECT/WITH SQL.")
        if not self.supported_tables:
            raise ValueError(f"Review plot recipe {self.recipe_id!r} must declare supported_tables.")
        if self.recipe_version < 1:
            raise ValueError(f"Review plot recipe {self.recipe_id!r} must have a positive recipe_version.")

    @property
    def required_tables(self) -> tuple[str, ...]:
        """Compatibility alias used by the original review plotting API."""

        return self.supported_tables

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REVIEW_PLOT_RECIPES: dict[str, ReviewPlotRecipe] = {
    "object_eci_radius": ReviewPlotRecipe(
        recipe_id="object_eci_radius",
        title="Canonical ECI radius",
        description="Canonical ECI radius by object from the object_state review table.",
        sql=(
            "SELECT time_s, object_id, "
            "sqrt(pos_x_eci_km * pos_x_eci_km + pos_y_eci_km * pos_y_eci_km + "
            "pos_z_eci_km * pos_z_eci_km) AS radius_km "
            "FROM object_state ORDER BY object_id, time_s"
        ),
        x_column="time_s",
        y_columns=("radius_km",),
        group_column="object_id",
        x_label="Time (s)",
        y_label="ECI radius (km)",
        artifact_id="evidence_object_eci_radius",
        supported_tables=("object_state",),
        required_columns=("time_s", "object_id", "radius_km"),
        natural_language_triggers=("plot ECI radius", "show orbital radius over time"),
    ),
    "relative_range": ReviewPlotRecipe(
        recipe_id="relative_range",
        title="Relative range over time",
        description="Deputy-chief range from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_km "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_km",),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Range (km)",
        artifact_id="evidence_relative_range",
        semantic_metric_names=("closest_approach_km", "final_range_km"),
        supported_tables=("relative_state",),
        required_columns=("time_s", "pair_id", "range_km"),
        natural_language_triggers=("plot relative range", "show range over time"),
    ),
    "relative_range_rate": ReviewPlotRecipe(
        recipe_id="relative_range_rate",
        title="Relative range rate over time",
        description="Relative range rate from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_rate_km_s "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_rate_km_s",),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Range rate (km/s)",
        artifact_id="evidence_relative_range_rate",
        semantic_metric_names=("range_rate_km_s",),
        supported_tables=("relative_state",),
        required_columns=("time_s", "pair_id", "range_rate_km_s"),
        natural_language_triggers=("plot range rate", "show closing rate"),
    ),
    "relative_velocity_components": ReviewPlotRecipe(
        recipe_id="relative_velocity_components",
        title="Relative velocity components over time",
        description="RIC-frame relative velocity components from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, "
            "v_radial_km_s, v_intrack_km_s, v_crosstrack_km_s "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("v_radial_km_s", "v_intrack_km_s", "v_crosstrack_km_s"),
        group_column="pair_id",
        x_label="Time (s)",
        y_label="Relative velocity (km/s)",
        artifact_id="evidence_relative_velocity",
        supported_tables=("relative_state",),
        required_columns=(
            "time_s",
            "pair_id",
            "v_radial_km_s",
            "v_intrack_km_s",
            "v_crosstrack_km_s",
        ),
        natural_language_triggers=("plot RIC relative velocity", "show relative velocity components"),
    ),
    "relative_position_ric_2d": ReviewPlotRecipe(
        recipe_id="relative_position_ric_2d",
        title="Relative trajectory in rectangular RIC",
        description="Professional I-R, I-C, and C-R projections from recorded rectangular-RIC review evidence.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, "
            "r_radial_km, i_intrack_km, c_crosstrack_km "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="i_intrack_km",
        y_columns=("r_radial_km",),
        group_column="pair_id",
        x_label="In-track, I (km)",
        y_label="Radial, R (km)",
        artifact_id="evidence_relative_position_ric_2d",
        supported_tables=("relative_state",),
        required_columns=(
            "time_s",
            "pair_id",
            "r_radial_km",
            "i_intrack_km",
            "c_crosstrack_km",
        ),
        renderer_id="ric_rectangular_2d",
        natural_language_triggers=(
            "plot the 2D RIC trajectory",
            "show RIC projections",
            "plot radial in-track cross-track motion",
        ),
    ),
    "burn_activity": ReviewPlotRecipe(
        recipe_id="burn_activity",
        title="Burn activity by object",
        description="Active thrust samples by object.",
        sql="SELECT object_id, SUM(burn_active) AS active_samples FROM thrust GROUP BY object_id ORDER BY object_id",
        x_column="object_id",
        y_columns=("active_samples",),
        plot_type="bar",
        x_label="Object",
        y_label="Active thrust samples",
        artifact_id="evidence_burn_activity",
        semantic_metric_names=("burn_activity",),
        supported_tables=("thrust",),
        required_columns=("object_id", "active_samples"),
        natural_language_triggers=("plot burn activity", "show thrust activity by object"),
    ),
    "ground_access": ReviewPlotRecipe(
        recipe_id="ground_access",
        title="Ground access samples",
        description="Access sample counts by station and object from the ground_access review table.",
        sql=(
            "SELECT station_id || ':' || object_id AS station_object, SUM(access) AS access_samples "
            "FROM ground_access GROUP BY station_id, object_id ORDER BY station_id, object_id"
        ),
        x_column="station_object",
        y_columns=("access_samples",),
        plot_type="bar",
        x_label="Station:Object",
        y_label="Access samples",
        artifact_id="evidence_ground_access",
        supported_tables=("ground_access",),
        required_columns=("station_object", "access_samples"),
        natural_language_triggers=("plot ground access", "show access samples by station"),
    ),
    "attitude_error": ReviewPlotRecipe(
        recipe_id="attitude_error",
        title="Attitude tracking error",
        description="Shortest-arc desired-versus-actual quaternion error by object.",
        sql=(
            "SELECT time_s, object_id, quat_error_angle_deg FROM attitude_error "
            "ORDER BY object_id, time_s"
        ),
        x_column="time_s",
        y_columns=("quat_error_angle_deg",),
        group_column="object_id",
        x_label="Time (s)",
        y_label="Quaternion error angle (deg)",
        artifact_id="evidence_attitude_error",
        semantic_metric_names=("attitude_error_deg",),
        supported_tables=("attitude_error",),
        required_columns=("time_s", "object_id", "quat_error_angle_deg"),
        natural_language_triggers=("plot attitude error", "show pointing error over time"),
    ),
    "attitude_body_rates": ReviewPlotRecipe(
        recipe_id="attitude_body_rates",
        title="Body angular rates",
        description="Body angular-rate components from retained object-state evidence.",
        sql=(
            "SELECT time_s, object_id, omega_x_rad_s, omega_y_rad_s, omega_z_rad_s "
            "FROM object_state ORDER BY object_id, time_s"
        ),
        x_column="time_s",
        y_columns=("omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s"),
        group_column="object_id",
        x_label="Time (s)",
        y_label="Body rate (rad/s)",
        artifact_id="evidence_attitude_body_rates",
        semantic_metric_names=("attitude_body_rate_rad_s",),
        supported_tables=("object_state",),
        required_columns=("time_s", "object_id", "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s"),
        natural_language_triggers=("plot body rates", "show angular rates over time"),
    ),
    "coverage_fraction": ReviewPlotRecipe(
        recipe_id="coverage_fraction",
        title="Instantaneous whole-Earth coverage",
        description="Covered HEALPix cell-center fraction by analysis and sample time.",
        sql=(
            "SELECT analysis_id, time_s, "
            "100.0 * instantaneous_covered_fraction AS covered_percent "
            "FROM coverage_samples ORDER BY analysis_id, time_s"
        ),
        x_column="time_s",
        y_columns=("covered_percent",),
        group_column="analysis_id",
        x_label="Analysis time (s)",
        y_label="Covered cell centers (%)",
        artifact_id="evidence_coverage_fraction",
        semantic_metric_names=("coverage_fraction",),
        supported_tables=("coverage_samples",),
        required_columns=("analysis_id", "time_s", "covered_percent"),
        natural_language_triggers=(
            "plot coverage fraction",
            "show whole-Earth coverage over time",
        ),
    ),
    "directed_link_margin": ReviewPlotRecipe(
        recipe_id="directed_link_margin",
        title="Directed-link margin",
        description="Link margin, zero-dB closure threshold, and qualified samples by directed link.",
        sql=(
            "SELECT analysis_id, time_s, margin_db, available "
            "FROM link_samples ORDER BY analysis_id, time_s"
        ),
        x_column="time_s",
        y_columns=("margin_db",),
        group_column="analysis_id",
        x_label="Analysis time (s)",
        y_label="Margin (dB)",
        artifact_id="evidence_directed_link_margin",
        semantic_metric_names=("directed_link_margin_db",),
        supported_tables=("link_samples",),
        required_columns=("analysis_id", "time_s", "margin_db", "available"),
        renderer_id="directed_link_margin",
        natural_language_triggers=(
            "plot link margin",
            "show link closure",
            "plot directed-link availability",
        ),
    ),
    "campaign_closest_approach": ReviewPlotRecipe(
        recipe_id="campaign_closest_approach",
        title="Campaign closest approach by iteration",
        description="Monte Carlo closest-approach results by iteration.",
        sql="SELECT iteration, closest_approach_km FROM campaign_runs ORDER BY iteration",
        x_column="iteration",
        y_columns=("closest_approach_km",),
        plot_type="scatter",
        x_label="Iteration",
        y_label="Closest approach (km)",
        artifact_id="evidence_campaign_closest_approach",
        semantic_metric_names=("campaign_closest_approach_km",),
        supported_tables=("campaign_runs",),
        required_columns=("iteration", "closest_approach_km"),
        natural_language_triggers=("plot campaign closest approach", "show closest approach by iteration"),
    ),
    "sensitivity_effects": ReviewPlotRecipe(
        recipe_id="sensitivity_effects",
        title="Sensitivity effect sizes",
        description="Ranked sensitivity effect sizes by parameter.",
        sql="SELECT parameter_path, effect_size FROM sensitivity_rankings ORDER BY rank, parameter_path, metric_path",
        x_column="parameter_path",
        y_columns=("effect_size",),
        plot_type="bar",
        x_label="Parameter",
        y_label="Effect size",
        artifact_id="evidence_sensitivity_effects",
        semantic_metric_names=("sensitivity_effect_size",),
        supported_tables=("sensitivity_rankings",),
        required_columns=("parameter_path", "effect_size"),
        natural_language_triggers=("plot sensitivity effects", "show ranked sensitivity"),
    ),
}


def get_review_plot_recipe(recipe_id: str) -> ReviewPlotRecipe | None:
    return REVIEW_PLOT_RECIPES.get(str(recipe_id or "").strip())


def list_review_plot_recipes() -> list[ReviewPlotRecipe]:
    return [REVIEW_PLOT_RECIPES[key] for key in sorted(REVIEW_PLOT_RECIPES)]


__all__ = [
    "PLOT_RECIPE_MATURITY_LEVELS",
    "PLOT_RECIPE_SCHEMA_VERSION",
    "REVIEW_PLOT_RECIPES",
    "ReviewPlotRecipe",
    "get_review_plot_recipe",
    "list_review_plot_recipes",
]

# Body-neutral recipes consume only the planetary tables.
for _metric, _label in (("altitude_km", "Altitude (km)"),
                        ("density_kg_m3", "Density (kg/m³)"),
                        ("relative_speed_m_s", "Atmosphere-relative speed (m/s)")):
    _recipe_id = "planetary_" + _metric
    REVIEW_PLOT_RECIPES[_recipe_id] = ReviewPlotRecipe(
        recipe_id=_recipe_id, title=_label,
        description="Recorded planetary sample diagnostics; no integrator-stage history.",
        sql=f"SELECT s.time_s, e.central_body || ':' || s.object_id AS object_id, s.{_metric} FROM planetary_state s CROSS JOIN planetary_environment e ORDER BY s.object_id,s.time_s",
        x_column="time_s", y_columns=(_metric,), group_column="object_id",
        x_label="Elapsed TAI time (s)", y_label=_label,
        artifact_id="evidence_"+_recipe_id, supported_tables=("planetary_state","planetary_environment"),
        required_columns=("time_s","object_id",_metric))
REVIEW_PLOT_RECIPES["planetary_inertial_xy"] = ReviewPlotRecipe(
    recipe_id="planetary_inertial_xy", title="Body-centered J2000 trajectory",
    description="Projection in body-centered inertial axes; no Earth map or radius outline.",
    sql="SELECT x_inertial_km,y_inertial_km,e.central_body || ':' || s.object_id AS object_id FROM planetary_state s CROSS JOIN planetary_environment e ORDER BY s.object_id,s.time_s",
    x_column="x_inertial_km", y_columns=("y_inertial_km",), group_column="object_id",
    x_label="J2000 x (km)", y_label="J2000 y (km)", plot_type="line",
    artifact_id="evidence_planetary_inertial_xy", supported_tables=("planetary_state","planetary_environment"),
    required_columns=("x_inertial_km","y_inertial_km","object_id"))
REVIEW_PLOT_RECIPES["planetary_drag"] = ReviewPlotRecipe(
    recipe_id="planetary_drag", title="Planetary drag acceleration",
    description="Native drag vector magnitude at recorded samples.",
    sql="SELECT time_s,e.central_body || ':' || s.object_id AS object_id, sqrt(pow(json_extract(forces_json,'$.drag[0]'),2)+pow(json_extract(forces_json,'$.drag[1]'),2)+pow(json_extract(forces_json,'$.drag[2]'),2)) AS drag_km_s2 FROM planetary_state s CROSS JOIN planetary_environment e ORDER BY s.object_id,s.time_s",
    x_column="time_s", y_columns=("drag_km_s2",), group_column="object_id",
    x_label="Elapsed TAI time (s)", y_label="Drag acceleration (km/s²)",
    artifact_id="evidence_planetary_drag", supported_tables=("planetary_state","planetary_environment"),
    required_columns=("time_s","object_id","drag_km_s2"))
