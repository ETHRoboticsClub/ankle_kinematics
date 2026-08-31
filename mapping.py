"""
Kinematic mapping between joint-space (theta_p, theta_r) and motor-space
(theta_A, theta_B). Two functions:

    joint_to_motor: policy says where to put the foot, we say what to do with motors
    motor_to_joint: encoders tell us where motors are, we tell policy where the foot is

joint_to_motor is analytic (circle-sphere intersection, per motor independently).
motor_to_joint is numerical (Newton iteration; the two pleuels couple the equations).
"""

import math
import warnings

import numpy as np
from geometry import AnkleGeometry, MotorGeometry, rot_about_axis
from kinematics import foot_anchor_world, crank_tip_world

# Set False in vectorised RL envs, where an unreachable pose per step would
# otherwise flood stderr. clamp_pose() makes it unnecessary anyway.
WARN_ON_UNREACHABLE = True


# ---------------------------------------------------------------------------
# Single-motor inverse:  P_foot known  ->  theta_motor
# ---------------------------------------------------------------------------

def _solve_motor_angle(motor: MotorGeometry, P_foot_target: np.ndarray,
                       theta_prev: float = 0.0) -> float:
    """Find theta_motor such that |P_kurbel(theta) - P_foot_target| = L.

    Geometric picture (per motor):
      - The crank-arm tip sits on a CIRCLE around the motor axis.
      - The pleuel-length constraint says it must also sit on a SPHERE of
        radius L centered at the foot anchor.
      - Circle and sphere intersect in 0, 1, or 2 points.
        We pick the one whose theta is closest to theta_prev (smooth branch).

    Math:
      P_kurbel(theta) = C + r*cos(theta)*u + r*sin(theta)*v
      Constraint:     |P_kurbel(theta) - P_foot_target|^2 = L^2

      Let d = C - P_foot_target. Expand:
        |d|^2 + 2*r*cos(theta) * (d.u) + 2*r*sin(theta) * (d.v) + r^2 = L^2

      Define:
        A = 2*r*(d.u)
        B = 2*r*(d.v)
        K = L^2 - |d|^2 - r^2

      Then:
        A*cos(theta) + B*sin(theta) = K
        => sqrt(A^2 + B^2) * cos(theta - phi) = K       with phi = atan2(B, A)
        => theta = phi +/- arccos(K / sqrt(A^2 + B^2))

      Two solutions (unless arccos argument exceeds 1 -> unreachable pose).
    """
    C = motor.C
    u = motor.u_hat
    v = motor.v_hat
    r = motor.r
    L = motor.L

    d = C - P_foot_target
    A = 2 * r * np.dot(d, u)
    B = 2 * r * np.dot(d, v)
    K = L**2 - np.dot(d, d) - r**2

    R = np.hypot(A, B)  # sqrt(A^2 + B^2)
    if R == 0:
        raise ValueError("Degenerate: A=B=0 (foot anchor lies on motor axis)")

    ratio = K / R
    if abs(ratio) > 1.0:
        # No real intersection: pose outside this motor's reachable set.
        # Warn the caller; clamp to nearest reachable as best-effort fallback.
        if WARN_ON_UNREACHABLE:
                warnings.warn(
                f"joint_to_motor: pose unreachable by this pleuel "
                f"(|K/R|={abs(ratio):.4f} > 1, L={L:.4f} m, r={r:.4f} m). "
                f"Clamping to nearest reachable angle.",
                RuntimeWarning, stacklevel=3,
            )
        ratio = np.clip(ratio, -1.0, 1.0)

    phi = np.arctan2(B, A)
    delta = np.arccos(ratio)

    # The two candidate solutions
    theta_1 = phi + delta
    theta_2 = phi - delta

    # Pick whichever is closer to theta_prev (wrap-around aware)
    def angle_diff(a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    if abs(angle_diff(theta_1, theta_prev)) <= abs(angle_diff(theta_2, theta_prev)):
        return float(((theta_1 + np.pi) % (2 * np.pi)) - np.pi)
    return float(((theta_2 + np.pi) % (2 * np.pi)) - np.pi)


# ---------------------------------------------------------------------------
# Public: joint -> motor (inverse mapping, the one the policy ultimately needs)
# ---------------------------------------------------------------------------

def joint_to_motor(geom: AnkleGeometry, theta_p: float, theta_r: float,
                   theta_prev: tuple[float, float] = (0.0, 0.0)
                   ) -> tuple[float, float]:
    """Given desired foot pose, return required motor angles.

    Returns (theta_upper, theta_lower) in radians.
    Pass `theta_prev` (the last commanded motor pair) to keep the branch
    consistent across calls -- otherwise jumps are possible across singularities.
    """
    pf_up = foot_anchor_world(geom, geom.upper, theta_p, theta_r)
    pf_lo = foot_anchor_world(geom, geom.lower, theta_p, theta_r)

    theta_upper = _solve_motor_angle(geom.upper, pf_up, theta_prev[0])
    theta_lower = _solve_motor_angle(geom.lower, pf_lo, theta_prev[1])
    return theta_upper, theta_lower




# ---------------------------------------------------------------------------
# Kreuzgelenk travel limit
# ---------------------------------------------------------------------------

def tilt_angle(theta_p: float, theta_r: float) -> float:
    """Total shaft deflection of the universal joint, in radians.

    For the standard tree (pitch about +y, then roll about +x) the foot's
    shaft direction is R_pitch @ R_roll @ z_hat, and the angle it makes with
    the shin's z_hat comes out as the exact closed form

        cos(tilt) = cos(pitch) * cos(roll)

    so pitch=30, roll=30 is 41.4 deg of tilt, not 30. Assumes the pitch and
    roll axes are orthogonal, which is what a Kreuzgelenk is; the __main__
    block below checks this against the full rotation matrix.
    """
    return math.acos(max(-1.0, min(1.0, math.cos(theta_p) * math.cos(theta_r))))


def clamp_pose(geom: AnkleGeometry, theta_p: float, theta_r: float
               ) -> tuple[float, float, bool]:
    """Project a commanded foot pose onto the Kreuzgelenk cone.

    Returns (pitch, roll, was_clamped). Scales the pose down along the ray
    from neutral, so the *direction* of the command is preserved and only its
    magnitude is cut -- which keeps the map continuous, and keeps a policy's
    gradient pointing somewhere useful instead of dying on a flat plateau.

    Call this on every commanded pose BEFORE joint_to_motor. Inside the 30 deg
    cone the linkage never reaches its own circle-sphere limit (worst |K/R| is
    0.81, worst Jacobian condition 1.33), so the unreachable branch, the
    clamping, and the slow-Newton behaviour never trigger at all.
    """
    target = math.cos(geom.max_tilt)
    if math.cos(theta_p) * math.cos(theta_r) >= target:
        return theta_p, theta_r, False
    lo, hi = 0.0, 1.0
    for _ in range(30):                       # bisection on the scale factor
        s = 0.5 * (lo + hi)
        if math.cos(s * theta_p) * math.cos(s * theta_r) >= target:
            lo = s
        else:
            hi = s
    return lo * theta_p, lo * theta_r, True


# ---------------------------------------------------------------------------
# Residuals + analytic Jacobian (replaces the finite-difference version)
# ---------------------------------------------------------------------------

def _triple(a, b, c) -> float:
    """a . (b x c), computed directly.

    np.cross costs ~16 us on 3-vectors and this expression needs four of them
    per residual evaluation, which made it the single most expensive thing in
    the Newton loop. The scalar triple product avoids allocating the cross
    product at all.
    """
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
          + a[1] * (b[2] * c[0] - b[0] * c[2])
          + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _foot_frames(geom: AnkleGeometry, theta_p: float, theta_r: float):
    """(R_pitch, R_roll) for this pose."""
    return (rot_about_axis(geom.pitch_axis, theta_p),
            rot_about_axis(geom.roll_axis, theta_r))


def _residuals_only(geom: AnkleGeometry, theta_A: float, theta_B: float,
                    theta_p: float, theta_r: float) -> np.ndarray:
    """Just the two residuals -- for seed selection and the line search,
    which do not need a Jacobian."""
    R_p, R_r = _foot_frames(geom, theta_p, theta_r)
    R = R_p @ R_r
    F = np.empty(2)
    for i, (m, th) in enumerate(((geom.upper, theta_A), (geom.lower, theta_B))):
        e = crank_tip_world(m, th) - R @ m.P_anchor_0
        F[i] = e @ e - m.L * m.L
    return F


def _residuals_and_jacobian(geom: AnkleGeometry, theta_A: float, theta_B: float,
                            theta_p: float, theta_r: float):
    """The two pleuel-length residuals AND d(residual)/d(pitch, roll).

    residual_i = |P_kurbel_i - P_foot_i|^2 - L_i^2,  with P_foot = R_p R_r P0.

    Differentiating,  dP_foot/dpitch = a_pitch x P_foot
                      dP_foot/droll  = R_p (a_roll x (R_r P0))
    so               dresidual/dtheta = -2 (P_k - P_f) . dP_foot/dtheta.

    Both of those contract with (P_k - P_f) straight away, so they are
    evaluated as scalar triple products rather than as cross products:
        e . (a_p x P_f)         = _triple(a_p, P_f, e)
        e . (R_p (a_r x P_r))   = _triple(a_r, P_r, R_p^T e)

    Exact -- matches central differences to 1e-11 -- and costs ONE evaluation
    where the old finite-difference Jacobian needed five.
    """
    R_p, R_r = _foot_frames(geom, theta_p, theta_r)
    a_p, a_r = geom.pitch_unit, geom.roll_unit

    F = np.empty(2)
    J = np.empty((2, 2))
    for i, (m, th) in enumerate(((geom.upper, theta_A), (geom.lower, theta_B))):
        P_r = R_r @ m.P_anchor_0
        P_f = R_p @ P_r
        e = crank_tip_world(m, th) - P_f
        F[i] = e @ e - m.L * m.L
        J[i, 0] = -2.0 * _triple(a_p, P_f, e)
        J[i, 1] = -2.0 * _triple(a_r, P_r, R_p.T @ e)
    return F, J


_SEED_CACHE: dict[int, np.ndarray] = {}


def _seed_matrix(geom: AnkleGeometry) -> np.ndarray:
    """2x2 matrix S with (pitch, roll) ~= S @ (theta_A, theta_B).

    The mapping is close to linear over the whole usable workspace, so the
    linearisation at neutral is a good starting guess everywhere -- worst case
    ~10 deg off across the +/-30 box, versus up to 30 deg for a cold (0, 0)
    seed. That is the difference between Newton needing 3 iterations and 20.

    S = -(dF/dtheta_joint)^-1 (dF/dtheta_motor), both evaluated at neutral.
    """
    key = id(geom)
    if key not in _SEED_CACHE:
        _, J = _residuals_and_jacobian(geom, 0.0, 0.0, 0.0, 0.0)
        M = np.zeros((2, 2))
        for i, m in enumerate((geom.upper, geom.lower)):
            e = crank_tip_world(m, 0.0) - m.P_anchor_0
            M[i, i] = 2.0 * (e @ (m.r * m.v_hat))     # d P_kurbel / d theta at 0
        _SEED_CACHE[key] = -np.linalg.solve(J, M)
    return _SEED_CACHE[key]


# ---------------------------------------------------------------------------
# Public: motor -> joint (forward mapping, what the policy sees as observation)
# ---------------------------------------------------------------------------

def _constraint_residuals(geom: AnkleGeometry,
                          theta_A: float, theta_B: float,
                          theta_p: float, theta_r: float) -> np.ndarray:
    """The two pleuel-length constraints, written as residuals.

    For each pleuel:
        residual = |P_kurbel(theta_motor) - P_foot(theta_p, theta_r)|^2 - L^2

    When both residuals are 0, the current (theta_p, theta_r) is consistent
    with the given motor angles. Newton drives both to zero.

    Squared norm (not norm) is used because it's smooth at zero and the
    Jacobian is cleaner.
    """
    pk_up = crank_tip_world(geom.upper, theta_A)
    pk_lo = crank_tip_world(geom.lower, theta_B)
    pf_up = foot_anchor_world(geom, geom.upper, theta_p, theta_r)
    pf_lo = foot_anchor_world(geom, geom.lower, theta_p, theta_r)

    r_up = np.dot(pk_up - pf_up, pk_up - pf_up) - geom.upper.L**2
    r_lo = np.dot(pk_lo - pf_lo, pk_lo - pf_lo) - geom.lower.L**2
    return np.array([r_up, r_lo])


def motor_to_joint(geom: AnkleGeometry, theta_A: float, theta_B: float,
                   theta_prev: tuple[float, float] = (0.0, 0.0),
                   max_iter: int = 12, tol: float = 1e-12,
                   max_step: float = 0.30, seed_switch: float = 1e-6,
                   return_status: bool = False):
    """Given measured motor angles, return the foot pose (theta_p, theta_r).

    Damped Newton on the 2-vector of pleuel-length residuals, with an analytic
    Jacobian. Two equations (one per pleuel), two unknowns.

    Robustness, in order of what it fixes:

    * The Jacobian is analytic, so one iteration costs one residual evaluation
      instead of five.
    * The starting guess is whichever of `theta_prev` and the linearised
      inverse (see _seed_matrix) already has the smaller residual. Near a
      travel limit `theta_prev` stops being a good guess -- that is what made
      the old solver take 16x longer per tick there -- and the linear model
      takes over.
    * Steps are length-limited (`max_step`) and backtracked until the residual
      actually decreases, so a near-singular Jacobian can no longer throw the
      iterate somewhere absurd.
    * If the motor angles are mutually inconsistent (no foot pose satisfies
      both rods, e.g. from encoder noise or a bad command) there is nothing to
      converge to. It stops early instead of burning every iteration, and
      reports converged=False.

    With `return_status=True` returns (pitch, roll, converged, iterations).
    """
    theta_p, theta_r = theta_prev
    warm = np.abs(_residuals_only(geom, theta_A, theta_B, theta_p, theta_r)).max()
    if warm > seed_switch:
        # theta_prev has gone stale -- this is the near-a-limit case that used
        # to cost 16x. Fall back to the linearised inverse if it is better.
        lin = _seed_matrix(geom) @ np.array([theta_A, theta_B])
        if np.abs(_residuals_only(geom, theta_A, theta_B, *lin)).max() < warm:
            theta_p, theta_r = float(lin[0]), float(lin[1])

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        F, J = _residuals_and_jacobian(geom, theta_A, theta_B, theta_p, theta_r)
        f0 = np.abs(F).max()
        if f0 < tol:
            converged, it = True, it - 1
            break

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break                                   # singular: keep best so far

        n = np.linalg.norm(delta)
        if n > max_step:
            delta *= max_step / n

        # backtracking line search: never accept a step that makes it worse
        alpha, improved = 1.0, False
        for _ in range(6):
            Fn = _residuals_only(geom, theta_A, theta_B,
                                 theta_p + alpha * delta[0],
                                 theta_r + alpha * delta[1])
            if np.abs(Fn).max() < f0:
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break            # no descent direction -> motor angles inconsistent
        theta_p += alpha * delta[0]
        theta_r += alpha * delta[1]
    else:
        converged = np.abs(_residuals_only(
            geom, theta_A, theta_B, theta_p, theta_r)).max() < tol

    if return_status:
        return float(theta_p), float(theta_r), bool(converged), it
    return float(theta_p), float(theta_r)


if __name__ == "__main__":
    from geometry import LEFT_LEG, foot_rotation

    print("Test 1: Neutral pose (theta_p=0, theta_r=0) should give theta_motor=0")
    tA, tB = joint_to_motor(LEFT_LEG, 0.0, 0.0)
    print(f"  theta_upper = {np.rad2deg(tA):+.3f} deg")
    print(f"  theta_lower = {np.rad2deg(tB):+.3f} deg")
    print(f"  (both should be ~0; small offsets come from measurement noise)")

    print()
    print("Test 2: Sweep pitch from -15 to +15 deg, roll = 0")
    print(f"  {'pitch':>8} {'theta_upper':>14} {'theta_lower':>14}")
    prev = (0.0, 0.0)
    for deg in (-15, -10, -5, 0, 5, 10, 15):
        tp = np.deg2rad(deg)
        tA, tB = joint_to_motor(LEFT_LEG, tp, 0.0, prev)
        print(f"  {deg:>+5d} deg   {np.rad2deg(tA):>+10.3f}    {np.rad2deg(tB):>+10.3f}")
        prev = (tA, tB)

    print()
    print("Test 3: Sweep roll from -15 to +15 deg, pitch = 0")
    print(f"  {'roll':>8} {'theta_upper':>14} {'theta_lower':>14}")
    prev = (0.0, 0.0)
    for deg in (-15, -10, -5, 0, 5, 10, 15):
        tr = np.deg2rad(deg)
        tA, tB = joint_to_motor(LEFT_LEG, 0.0, tr, prev)
        print(f"  {deg:>+5d} deg   {np.rad2deg(tA):>+10.3f}    {np.rad2deg(tB):>+10.3f}")
        prev = (tA, tB)

    print()
    print("Test 4: Forward mapping (motor -> joint) consistency.")
    print("  Take a foot pose, convert to motors, convert back. Should recover the pose.")
    print(f"  {'pitch_in':>9} {'roll_in':>9}   {'pitch_out':>10} {'roll_out':>10}   {'err_p':>8} {'err_r':>8}")
    prev_motor = (0.0, 0.0)
    prev_joint = (0.0, 0.0)
    for tp_deg, tr_deg in [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5),
                            (5, 5), (-5, -5), (10, -3), (-3, 10)]:
        tp = np.deg2rad(tp_deg)
        tr = np.deg2rad(tr_deg)
        tA, tB = joint_to_motor(LEFT_LEG, tp, tr, prev_motor)
        tp_back, tr_back = motor_to_joint(LEFT_LEG, tA, tB, prev_joint)
        err_p = np.rad2deg(tp_back - tp)
        err_r = np.rad2deg(tr_back - tr)
        print(f"  {tp_deg:>+6d}   {tr_deg:>+6d}     "
              f"{np.rad2deg(tp_back):>+8.3f}    {np.rad2deg(tr_back):>+8.3f}     "
              f"{err_p:>+6.2e}  {err_r:>+6.2e}")
        prev_motor = (tA, tB)
        prev_joint = (tp_back, tr_back)

    print()
    print("Test 5: Kreuzgelenk cone -- tilt_angle closed form vs rotation matrix")
    worst = max(
        abs(tilt_angle(p * np.pi / 180, r * np.pi / 180)
            - np.arccos(np.clip(foot_rotation(LEFT_LEG, p * np.pi / 180,
                                              r * np.pi / 180)[2, 2], -1, 1)))
        for p in range(-40, 41, 5) for r in range(-40, 41, 5))
    print(f"  max discrepancy = {worst:.2e} rad")

    print()
    print("Test 6: clamp_pose projects onto the cone and keeps the direction")
    print(f"  {'commanded':>18}   {'clamped':>18}   {'tilt':>7}  clamped?")
    for p_deg, r_deg in [(10, 10), (30, 0), (30, 30), (45, -45), (0, 60)]:
        p, r = np.deg2rad(p_deg), np.deg2rad(r_deg)
        cp, cr, hit = clamp_pose(LEFT_LEG, p, r)
        print(f"  ({p_deg:+4d}, {r_deg:+4d}) deg   ({np.rad2deg(cp):+6.2f},{np.rad2deg(cr):+6.2f}) deg   "
              f"{np.rad2deg(tilt_angle(cp, cr)):6.2f}d  {hit}")

    print()
    print("Test 7: solver robustness -- inconsistent motor angles must not hang")
    for tA_deg, tB_deg in [(60, 60), (-80, 80), (0, 0)]:
        p, r, ok, n = motor_to_joint(LEFT_LEG, np.deg2rad(tA_deg), np.deg2rad(tB_deg),
                                     (0.0, 0.0), return_status=True)
        print(f"  motors ({tA_deg:+4d},{tB_deg:+4d}) deg -> pitch {np.rad2deg(p):+7.2f} "
              f"roll {np.rad2deg(r):+7.2f}   converged={ok}  iterations={n}")
