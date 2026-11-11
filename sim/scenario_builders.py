# ruff: noqa: F401,F403,F405,I001
from sim.ingestion_normalization import *
from sim.ingestion_sources import *

def build_basic_propagation_scenario(
    packet: MissionInputPacket | Mapping[str, Any],
    *,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    initial_jd_utc: float | None = None,
    dynamics_model: str = "two_body",
    j2: bool = False,
    j3: bool = False,
    j4: bool = False,
    drag: bool = False,
    srp: bool = False,
    third_body_sun: bool = False,
    third_body_moon: bool = False,
    atmosphere_model: str | None = None,
    spherical_harmonics: Mapping[str, Any] | None = None,
    review_detail: str = "standard",
    trajectory_only: bool = False,
) -> ScenarioArtifact:
    packet_obj = packet if isinstance(packet, MissionInputPacket) else packet_from_dict(packet)
    objects = packet_obj.objects
    if not objects:
        raise ValueError("mission input packet contains no objects.")
    readiness_errors = _propagation_readiness_errors(objects)
    if readiness_errors:
        raise ValueError("; ".join(readiness_errors))
    root = _scenario_root(
        scenario_name=scenario_name,
        description="Agent-generated propagation scenario from normalized mission input packet.",
        output_dir=output_dir,
        duration_s=duration_s,
        dt_s=dt_s,
        initial_jd_utc=initial_jd_utc,
        dynamics_model=dynamics_model,
        j2=j2,
        j3=j3,
        j4=j4,
        drag=drag,
        srp=srp,
        third_body_sun=third_body_sun,
        third_body_moon=third_body_moon,
        atmosphere_model=atmosphere_model,
        spherical_harmonics=spherical_harmonics,
        review_detail=review_detail,
        packet=packet_obj,
    )
    for oid, obj in objects.items():
        root["objects"][str(oid)] = _scenario_object(obj, trajectory_only=trajectory_only)
    return ScenarioArtifact.from_dict(root)


def build_basic_rpo_scenario(
    packet: MissionInputPacket | Mapping[str, Any],
    *,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    target_id: str = "target",
    chaser_id: str = "chaser",
    initial_jd_utc: float | None = None,
    dynamics_model: str = "two_body",
    j2: bool = False,
    j3: bool = False,
    j4: bool = False,
    drag: bool = False,
    srp: bool = False,
    third_body_sun: bool = False,
    third_body_moon: bool = False,
    atmosphere_model: str | None = None,
    spherical_harmonics: Mapping[str, Any] | None = None,
    review_detail: str = "standard",
) -> ScenarioArtifact:
    packet_obj = packet if isinstance(packet, MissionInputPacket) else packet_from_dict(packet)
    objects = packet_obj.objects
    rpo_errors = _rpo_readiness_errors(objects, target_id=target_id, chaser_id=chaser_id)
    if rpo_errors:
        raise ValueError("; ".join(rpo_errors))
    root = _scenario_root(
        scenario_name=scenario_name,
        description="Agent-generated basic RPO scenario from normalized mission input packet.",
        output_dir=output_dir,
        duration_s=duration_s,
        dt_s=dt_s,
        initial_jd_utc=initial_jd_utc,
        dynamics_model=dynamics_model,
        j2=j2,
        j3=j3,
        j4=j4,
        drag=drag,
        srp=srp,
        third_body_sun=third_body_sun,
        third_body_moon=third_body_moon,
        atmosphere_model=atmosphere_model,
        spherical_harmonics=spherical_harmonics,
        review_detail=review_detail,
        packet=packet_obj,
    )
    root["objects"][target_id] = _scenario_object(objects[target_id], forced_role="target")
    root["objects"][chaser_id] = _scenario_object(objects[chaser_id], forced_role="chaser")
    return ScenarioArtifact.from_dict(root)
