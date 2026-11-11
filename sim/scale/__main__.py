from __future__ import annotations

from sim.pro_features import FEATURE_SCALE_ANALYTICS, require_pro_feature
from sim.scale.cli.common import (
    _first_tle_lines_from_file as _first_tle_lines_from_file,
)
from sim.scale.cli.common import (
    _initial_state_from_mission_input_packet as _initial_state_from_mission_input_packet,
)
from sim.scale.cli.common import (
    _json_mapping_from_path as _json_mapping_from_path,
)
from sim.scale.cli.common import (
    _spherical_harmonics_from_args as _spherical_harmonics_from_args,
)
from sim.scale.cli.dispatcher import dispatch
from sim.scale.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
