from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.numeric_backend import normalize_numeric_backend
from sim.optimization.base import ObjectiveFn, OptimizationResult, ParameterBound


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 24
    iterations: int = 50
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        if self.particles <= 0:
            raise ValueError("particles must be positive.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if not np.isfinite(float(self.inertia)) or self.inertia < 0.0:
            raise ValueError("inertia must be non-negative.")
        if (
            not np.isfinite(float(self.cognitive))
            or not np.isfinite(float(self.social))
            or self.cognitive < 0.0
            or self.social < 0.0
        ):
            raise ValueError("cognitive and social terms must be non-negative.")
        backend = normalize_numeric_backend(self.numeric_backend)
        if backend not in {"python", "rust"}:
            raise ValueError("numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)


class ParticleSwarmOptimizer:
    """Simple bounded PSO for continuous parameter vectors."""

    def __init__(self, cfg: PSOConfig):
        self.cfg = cfg

    def optimize(self, objective: ObjectiveFn, bounds: list[ParameterBound], seed: int = 0) -> OptimizationResult:
        if not bounds:
            raise ValueError("At least one parameter bound is required.")

        rng = np.random.default_rng(seed)
        dim = len(bounds)
        lb = np.array([b.lower for b in bounds], dtype=float)
        ub = np.array([b.upper for b in bounds], dtype=float)
        span = ub - lb

        x = lb + rng.random((self.cfg.particles, dim)) * span
        v = rng.uniform(-0.2, 0.2, size=(self.cfg.particles, dim)) * span

        pbest_x = x.copy()
        pbest_f = np.array([objective(xi) for xi in x], dtype=float)
        g_idx = int(np.argmin(pbest_f))
        gbest_x = pbest_x[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])
        history = [gbest_f]

        for _ in range(self.cfg.iterations):
            r1 = rng.random((self.cfg.particles, dim))
            r2 = rng.random((self.cfg.particles, dim))
            if self.cfg.numeric_backend == "rust":
                from sim.rust_scale_backend import pso_propose

                x, v = pso_propose(
                    x,
                    v,
                    pbest_x,
                    gbest_x,
                    r1,
                    r2,
                    lb,
                    ub,
                    inertia=self.cfg.inertia,
                    cognitive=self.cfg.cognitive,
                    social=self.cfg.social,
                )
                f = np.array([objective(xi) for xi in x], dtype=float)
                improved = f < pbest_f
                if np.any(improved):
                    pbest_x[improved, :] = x[improved, :]
                    pbest_f[improved] = f[improved]
                    g_idx = int(np.argmin(pbest_f))
                    gbest_x = pbest_x[g_idx].copy()
                    gbest_f = float(pbest_f[g_idx])
            else:
                v = (
                    self.cfg.inertia * v
                    + self.cfg.cognitive * r1 * (pbest_x - x)
                    + self.cfg.social * r2 * (gbest_x[None, :] - x)
                )
                x = x + v
                x = np.clip(x, lb, ub)

                f = np.array([objective(xi) for xi in x], dtype=float)
                improved = f < pbest_f
                if np.any(improved):
                    pbest_x[improved, :] = x[improved, :]
                    pbest_f[improved] = f[improved]
                    g_idx = int(np.argmin(pbest_f))
                    gbest_x = pbest_x[g_idx].copy()
                    gbest_f = float(pbest_f[g_idx])
            history.append(gbest_f)

        metadata = {
            "optimizer": "pso",
            "particles": self.cfg.particles,
            "iterations": self.cfg.iterations,
            "inertia": self.cfg.inertia,
            "cognitive": self.cfg.cognitive,
            "social": self.cfg.social,
        }
        if self.cfg.numeric_backend == "rust":
            metadata["numeric_backend"] = "rust"
        return OptimizationResult(
            best_x=gbest_x,
            best_cost=gbest_f,
            history_best_cost=history,
            metadata=metadata,
        )
