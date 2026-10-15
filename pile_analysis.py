"""Axial pile finite-difference calculation adapted from Axial_Pile_FDA_2026.ipynb.

Linear-elastic, homogeneous shaft springs and a plugged pile-toe spring.
The N+3 ghost-node system, sign convention, and central difference force
calculation follow the supplied notebook and Prof. Elgamal's 2026 handout.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PileInputs:
    """Pile inputs in m, kN, and kPa (kN/m^2). Negative Q = compression."""

    N: int = 50
    L: float = 10.0  # m
    Q: float = -50.0  # kN
    d: float = 0.5  # outside diameter, m
    t: float = 0.05  # wall thickness, m
    E: float = 2e8  # kPa = kN/m^2
    kf: float = 1e5  # shaft soil modulus, kPa/m
    kbase: float = 1e12  # base soil modulus, kPa/m


@dataclass(frozen=True)
class PileResults:
    z: np.ndarray  # physical node depths, m
    u_num: np.ndarray  # numerical displacement, m
    F_num: np.ndarray  # numerical axial force, kN
    u_exact: np.ndarray  # analytical displacement, m
    F_exact: np.ndarray  # analytical axial force, kN
    A: float  # pipe wall area, m^2
    EA: float  # axial pile rigidity, kN
    k: float  # distributed shaft spring coefficient, kN/m^2
    kend: float  # total plugged toe stiffness, kN/m
    dz: float  # m

    @property
    def max_displacement_difference_mm(self) -> float:
        return float(1000 * np.max(np.abs(self.u_num - self.u_exact)))

    @property
    def max_force_difference_kN(self) -> float:
        return float(np.max(np.abs(self.F_num - self.F_exact)))


def _validate(p: PileInputs) -> None:
    if isinstance(p.N, bool) or not isinstance(p.N, (int, np.integer)) or not 2 <= p.N <= 500:
        raise ValueError("Number of segments must be an integer between 2 and 500.")
    for name in ("L", "Q", "d", "t", "E", "kf", "kbase"):
        if not math.isfinite(float(getattr(p, name))):
            raise ValueError(f"{name} must be a finite number.")
    if p.L <= 0 or p.d <= 0 or p.t <= 0 or p.E <= 0:
        raise ValueError("Pile length, diameter, wall thickness and Young's modulus must be positive.")
    if 2 * p.t > p.d:
        raise ValueError("The wall thickness cannot exceed half the outside diameter.")
    if p.kf < 0 or p.kbase < 0:
        raise ValueError("The shaft and tip stiffness moduli cannot be negative.")
    if p.kf == 0 and p.kbase == 0:
        raise ValueError("At least one soil spring stiffness must be positive; otherwise the pile is unrestrained.")


def _analytical_solution(
    z: np.ndarray, L: float, Q: float, EA: float, k: float, kend: float
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic closed form; stable for large lambda*L, including a zero-shaft limit.

    This is algebraically equivalent to the cosh/sinh solution in the notebook,
    with omega=kend/(EA*lambda) (NOT kbase/(EA*lambda)).
    """
    if k == 0:
        # With no shaft restraint, F=Q everywhere, and F(L)=kend*u(L).
        u = Q / kend + (L - z) * (Q / EA)
        return u, np.full_like(z, Q, dtype=float)

    lam = math.sqrt(k / EA)
    omega = kend / (EA * lam)
    s = lam * L
    remaining = lam * (L - z)
    exp2remaining = np.exp(-2.0 * remaining)
    remaining_part = -np.expm1(-2.0 * remaining)
    exp2full = math.exp(-2.0 * s)
    # 1 - exp(-2s) + omega * (1 + exp(-2s)), computed without cancellation.
    denom = -math.expm1(-2.0 * s) + omega * (1.0 + exp2full)
    exp_depth = np.exp(-lam * z)
    u_exact = (Q / (EA * lam)) * exp_depth * (
        1.0 + exp2remaining + omega * remaining_part
    ) / denom
    F_exact = Q * exp_depth * (
        remaining_part + omega * (1.0 + exp2remaining)
    ) / denom
    return u_exact, F_exact


def analyze_pile(p: PileInputs) -> PileResults:
    """Solve the original notebook's N+3 finite-difference matrix and compare to exact."""
    _validate(p)

    # From the original notebook: stiffness, geometry and plugged toe area.
    A = np.pi * (p.d**2 - (p.d - 2.0 * p.t)**2) / 4.0
    EA = p.E * A
    k = p.kf * np.pi * p.d
    kend = p.kbase * np.pi * p.d**2 / 4.0
    z = np.linspace(0.0, p.L, p.N + 1)
    dz = p.L / p.N

    # Same ghost-node system (N+3)x(N+3) as cells 8 and 10 of the notebook.
    K = np.zeros((p.N + 3, p.N + 3))
    f = np.zeros(p.N + 3)

    # Head BC: -u[-1] + u[1] = -2*dz*Q/EA.
    K[0, 0] = -1.0
    K[0, 2] = 1.0
    f[0] = -2.0 * dz * p.Q / EA

    B = -2.0 - k * dz**2 / EA
    for n in range(1, p.N + 2):
        K[n, n - 1] = 1.0
        K[n, n] = B
        K[n, n + 1] = 1.0

    # Toe BC: -u[N-1] + (2*dz*kend/EA)*u[N] + u[N+1] = 0.
    CN = 2.0 * dz * kend / EA
    K[p.N + 2, p.N] = -1.0
    K[p.N + 2, p.N + 1] = CN
    K[p.N + 2, p.N + 2] = 1.0

    try:
        u = np.linalg.solve(K, f)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The finite-difference system could not be solved for these settings. "
            "Try larger soil stiffnesses or fewer segments."
        ) from exc
    if not np.all(np.isfinite(u)):
        raise ValueError("The calculated displacements are non-finite. Change the input parameters.")

    # Central-difference force evaluated at the N+1 physical nodes.
    u_num = u[1:-1]
    F_num = -EA * (u[2:] - u[:-2]) / (2.0 * dz)

    u_exact, F_exact = _analytical_solution(z, p.L, p.Q, EA, k, kend)
    return PileResults(z, u_num, F_num, u_exact, F_exact, A, EA, k, kend, dz)
