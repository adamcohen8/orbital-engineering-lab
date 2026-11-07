"""Pinned redistributable gravity-source metadata shared by all download callers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GravityModelDownload:
    urls: tuple[str, ...]
    sha256: str
    min_size_bytes: int = 1
    size_bytes: int | None = None


# Preserve the established public import and pickle identity through the façade.
GravityModelDownload.__module__ = "sim.dynamics.orbit.spherical_harmonics"


REAL_MODEL_DOWNLOADS = {
    "EGM96": GravityModelDownload(
        urls=("https://icgem.gfz-potsdam.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc",),
        sha256="5247a9e9c316dd2c8f8fd491d53be0e163cb5cb3676b021754240cc9e44cb43b", min_size_bytes=5_000_000, size_bytes=5_620_000),
    "EGM2008": GravityModelDownload(
        urls=("https://icgem.gfz.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc",),
        sha256="ab5b524da073e63b5bdceb7ca47a0de07a26dd44a1c5798f39fc98dc80af70fd", min_size_bytes=252_140_870, size_bytes=252_140_870),
}
