"""PDS SHA ingestion with explicit GM, radius, normalization and frame identity."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np

from .resources import DATA_ROOT, ResourceReceipt, read_json


@dataclass(frozen=True)
class PreparedGravity:
    model_id: str
    body_id: str
    mu_km3_s2: float
    radius_km: float
    frame_id: str | None
    compiled: object | None
    receipts: tuple[ResourceReceipt, ...]

    def metadata(self):
        from dataclasses import asdict

        return {
            "model": self.model_id,
            "body_id": self.body_id,
            "mu_km3_s2": self.mu_km3_s2,
            "reference_radius_km": self.radius_km,
            "fixed_frame": self.frame_id,
            "normalization": "fully_normalized_no_condon_shortley",
            "static": True,
            "degree": 0 if self.compiled is None else self.compiled.n_max,
            "order": 0 if self.compiled is None else self.compiled.m_max,
            "resources": [asdict(r) for r in self.receipts],
        }


def load_pds_sha(
    path, label_path, *, body_id, frame_id, degree, order, expected_sha256=None
):
    from sim.dynamics.orbit.spherical_harmonics import (
        SphericalHarmonicTerm,
        compile_spherical_harmonic_terms,
    )

    receipt, content = ResourceReceipt.read(path, expected_sha256)
    label_receipt, label = ResourceReceipt.read(label_path)
    label_text = label.decode("ascii")
    if not re.search(
        r'TARGET_NAME\s*=\s*"?' + re.escape(body_id.upper()) + r'"?(?=\s|$)', label_text
    ):
        raise ValueError("PDS gravity label target/body mismatch")
    rows = list(csv.reader(content.decode("ascii").splitlines()))
    header = [float(v) for v in rows[0]]
    if (
        len(header) != 8
        or not np.all(np.isfinite(header))
        or header[0] <= 0
        or header[1] <= 0
    ):
        raise ValueError("Invalid PDS SHA gravity header")
    radius, mu, _, nmax, mmax, normalization, lon0, lat0 = header
    if nmax != int(nmax) or mmax != int(mmax) or not 0 <= mmax <= nmax or header[2] < 0:
        raise ValueError("Invalid PDS SHA maximum degree/order or GM uncertainty")
    if normalization != 1 or lon0 != 0 or lat0 != 0:
        raise ValueError(
            "PDS SHA requires fully normalized coefficients and zero reference offsets"
        )
    if not 2 <= degree <= min(int(nmax), 20) or not 0 <= order <= min(
        degree, int(mmax)
    ):
        raise ValueError("Planetary gravity qualifies degree 2..20 and order 0..degree")
    seen, terms = set(), []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        if len(row) != 6:
            raise ValueError("PDS SHA coefficient record requires six columns")
        vals = [float(v) for v in row]
        if not np.all(np.isfinite(vals)):
            raise ValueError("PDS SHA coefficients must be finite")
        n, m = int(vals[0]), int(vals[1])
        if (
            not np.all(np.isfinite(vals))
            or vals[0] != n
            or vals[1] != m
            or not 0 <= m <= n <= nmax
            or (n, m) in seen
        ):
            raise ValueError("Invalid or duplicate PDS SHA coefficient")
        if vals[4] < 0 or vals[5] < 0 or (m == 0 and vals[3] != 0):
            raise ValueError("Invalid PDS SHA uncertainty or zonal sine coefficient")
        seen.add((n, m))
        if n == 0 and (m != 0 or vals[2] != 1 or vals[3] != 0):
            raise ValueError("PDS degree zero must be the unit monopole")
        if n == 1 and (vals[2] != 0 or vals[3] != 0):
            raise ValueError("Planetary gravity requires center-of-mass degree one")
        if 2 <= n <= degree and m <= order:
            terms.append(SphericalHarmonicTerm(n, m, vals[2], vals[3], normalized=True))
    required = {(n, m) for n in range(2, degree + 1) for m in range(min(n, order) + 1)}
    if not required.issubset(seen):
        raise ValueError("PDS SHA is missing requested coefficients")
    compiled = compile_spherical_harmonic_terms(terms)
    for value in vars(compiled).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return PreparedGravity(
        "pds_sha_static",
        body_id,
        mu,
        radius,
        frame_id,
        compiled,
        (receipt, label_receipt),
    )


def prepare_gravity(body, config):
    model = config.get("model", "point_mass")
    if model == "point_mass":
        return PreparedGravity(
            model,
            body.body_id,
            body.mu_km3_s2,
            body.reference_radius_km,
            None,
            None,
            (),
        )
    if model == "gmm3_static":
        raise ValueError("GMM-3 exact rotation model is not yet qualified")
    if model == "gmm2b_static":
        if body.body_id != "mars":
            raise ValueError("gmm2b_static requires Mars center")
        manifest_receipt, manifest = read_json(DATA_ROOT / "gmm2b_manifest.json")
        result = load_pds_sha(
            DATA_ROOT / manifest["coefficients"],
            DATA_ROOT / manifest["label"],
            body_id="mars",
            frame_id=manifest["frame_id"],
            degree=config.get("max_degree", 20),
            order=config.get("max_order", config.get("max_degree", 20)),
            expected_sha256=manifest["coefficients_sha256"],
        )
        result.receipts[1].verify()
        if result.receipts[1].sha256 != manifest["label_sha256"]:
            raise ValueError("Bundled gravity label SHA256 mismatch")
        from dataclasses import replace

        return replace(
            result, model_id=model, receipts=result.receipts + (manifest_receipt,)
        )
    if model == "pds_sha_static":
        return load_pds_sha(
            config["coeff_path"],
            config["label_path"],
            body_id=body.body_id,
            frame_id=config["frame_id"],
            degree=config.get("max_degree", 20),
            order=config.get("max_order", 20),
            expected_sha256=config.get("sha256"),
        )
    raise ValueError(f"Unsupported planetary gravity model {model!r}")
