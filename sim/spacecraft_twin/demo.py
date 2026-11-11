"""Synthetic engineering fixture with known answers; not evidence of agent extraction accuracy."""

from pathlib import Path

import numpy as np

from .compiler import compile_model
from .contracts import SCHEMA, file_digest, write_json


def box_triangles(size=(2.0, 2.0, 2.0), offset=(0.0, 0.0, 0.0)):
    vertices = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * size + offset
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array(
        [[vertices[a], vertices[b], vertices[c]] for i, j, k, last in faces for a, b, c in [(i, j, k), (i, k, last)]]
    )


def write_stl(path, triangles):
    lines = ["solid synthetic"]
    for tri in triangles:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n /= np.linalg.norm(n)
        lines += ["facet normal " + " ".join(map(str, n)), "outer loop"]
        lines += ["vertex " + " ".join(map(str, v)) for v in tri]
        lines += ["endloop", "endfacet"]
    lines += ["endsolid synthetic"]
    Path(path).write_text("\n".join(lines))


def create_demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    sources = root / "sources"
    sources.mkdir()
    (sources / "engineering.md").write_text(
        "# Synthetic spacecraft revision B\nMass 42 kg. COM [0.1,0.2,0] m in body axes. Inertia about COM diag[12,10,8] kg m2.\nThruster max force 2 N; Isp 220 s.\nWheel torque 0.05 Nm; momentum 0.2 Nms; rotor inertia 0.001 kg m2; max speed 600 rad/s.\nFixed rigid geometry in metres; absorber coefficients Cd=2.2, Cr=1.0.\n"
    )
    (sources / "mounts.csv").write_text(
        "id,x_mm,y_mm,z_mm,force_x,force_y,force_z\nthruster_a,0,1000,0,1,0,0\nthruster_b,0,-1000,0,-1,0,0\n"
    )
    write_json(
        sources / "assembly.json",
        {
            "revision": "B",
            "frame": "body",
            "wheel_axes": [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
            "wheel_positions_m": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [-0.2, 0, 0]],
        },
    )
    triangles = np.concatenate([box_triangles(), box_triangles((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))])
    write_stl(sources / "assembly.stl", triangles)
    model = {
        "schema": SCHEMA,
        "id": "demo_sat",
        "version": "1.0.0",
        "mass_kg": 42.0,
        "center_of_mass_body_m": [0.1, 0.2, 0.0],
        "inertia_kg_m2": np.diag([12.0, 10.0, 8.0]).tolist(),
        "frames": {},
        "components": [],
        "sources": [],
        "evidence": {},
        "assumptions": [],
        "conflicts": [],
        "metadata": {"fixture": "synthetic; author-specified answers, not model capability evidence"},
    }
    for name, pos, axis in [("thruster_a", [0, 1000, 0], [1, 0, 0]), ("thruster_b", [0, -1000, 0], [-1, 0, 0])]:
        model["components"].append(
            {
                "id": name,
                "kind": "thruster",
                "frame": "body",
                "position": pos,
                "units": "mm",
                "axis": axis,
                "axis_convention": "force",
                "parameters": {"max_thrust_n": 2.0, "isp_s": 220.0},
            }
        )
    axes = [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]
    for i, axis in enumerate(axes):
        model["components"].append(
            {
                "id": f"wheel_{i}",
                "kind": "reaction_wheel",
                "frame": "body",
                "position": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [-0.2, 0, 0]][i],
                "units": "m",
                "axis": axis,
                "axis_convention": "positive_spin",
                "parameters": {
                    "max_torque_nm": 0.05,
                    "max_momentum_nms": 0.2,
                    "inertia_kg_m2": 0.001,
                    "max_speed_rad_s": 600.0,
                },
            }
        )
    for i, p in enumerate(sorted(sources.iterdir())):
        model["sources"].append(
            {"id": f"source_{i}", "path": p.relative_to(root).as_posix(), "sha256": file_digest(p), "revision": "B"}
        )
    by_name = {Path(s["path"]).name: s["id"] for s in model["sources"]}
    for field in compile_model(model, root)["required_evidence"]:
        if "/parameters/" in field or not field.startswith("components/"):
            name, locator = "engineering.md", "revision B properties"
        elif "/thruster_" in field:
            name, locator = "mounts.csv", field.split("/")[1] + " row; assembly frame body; force direction"
        else:
            name, locator = "assembly.json", field.split("/")[1] + " wheel_axes / wheel_positions_m"
        model["evidence"][field] = [{"source_id": by_name[name], "locator": locator, "basis": "extracted"}]
    write_json(root / "twin.json", model)
    return root / "twin.json"
