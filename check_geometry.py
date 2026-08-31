"""
One-shot validation of the geometry in geometry.py. Run after any CAD change:

    python3.11 check_geometry.py

Checks, in order:
  1. Pleuel-length residual   -- did all points come from the same neutral pose?
  2. Neutral offset           -- does (pitch=0, roll=0) really give theta_motor=0?
  3. Reachable envelope       -- how far can pitch/roll go before a pleuel runs
                                 out of circle-sphere intersection?
  4. Transmission ratio       -- deg of motor travel per deg of foot travel, and
                                 the conditioning of the 2x2 motor->joint Jacobian.
  5. Left/right sign check    -- do both legs move the same way for the same pose?
"""

import numpy as np
from geometry import LEFT_LEG, RIGHT_LEG, AnkleGeometry, MotorGeometry
from kinematics import foot_anchor_world, crank_tip_world
from mapping import joint_to_motor, motor_to_joint, clamp_pose, tilt_angle

DEG = np.pi / 180.0


def reach_ratio(motor: MotorGeometry, P_foot: np.ndarray) -> float:
    """|K/R| from the circle-sphere intersection. <1 reachable, >1 not.
    1.0 is the exact edge of this pleuel's workspace."""
    d = motor.C - P_foot
    A = 2 * motor.r * np.dot(d, motor.u_hat)
    B = 2 * motor.r * np.dot(d, motor.v_hat)
    K = motor.L**2 - np.dot(d, d) - motor.r**2
    return abs(K / np.hypot(A, B))


def reachable(geom: AnkleGeometry, tp: float, tr: float) -> bool:
    return all(
        reach_ratio(m, foot_anchor_world(geom, m, tp, tr)) <= 1.0
        for m in (geom.upper, geom.lower)
    )


def jacobian(geom: AnkleGeometry, tp: float, tr: float, h: float = 1e-5):
    """d(theta_upper, theta_lower) / d(pitch, roll), evaluated at (tp, tr)."""
    prev = joint_to_motor(geom, tp, tr)
    cols = []
    for i in range(2):
        d = [0.0, 0.0]
        d[i] = h
        p = np.array(joint_to_motor(geom, tp + d[0], tr + d[1], prev))
        m = np.array(joint_to_motor(geom, tp - d[0], tr - d[1], prev))
        cols.append((p - m) / (2 * h))
    return np.column_stack(cols)


def main():
    geom = RIGHT_LEG
    print(f"=== geometry check: {geom.side} leg ===\n")

    # --- 1. length residual -------------------------------------------------
    print("1. Pleuel length residual (must be ~0; large = points from different poses)")
    for name, m in (("upper", geom.upper), ("lower", geom.lower)):
        err = (m.sanity_check - m.L) * 1000
        flag = "OK" if abs(err) < 0.1 else "*** FAIL ***"
        print(f"   {name}: r = {m.r*1000:7.2f} mm   L = {m.L*1000:8.3f} mm   "
              f"residual = {err:+7.4f} mm   {flag}")

    # --- 2. neutral offset --------------------------------------------------
    print("\n2. Neutral pose -> motor angles (should be 0.000)")
    tA, tB = joint_to_motor(geom, 0.0, 0.0)
    print(f"   theta_upper = {tA/DEG:+8.4f} deg")
    print(f"   theta_lower = {tB/DEG:+8.4f} deg")

    # --- 3. reachable envelope ----------------------------------------------
    print("\n3. Reachable envelope (pure pitch / pure roll, 0.5 deg steps)")
    for label, axis in (("pitch", 0), ("roll", 1)):
        lo = hi = 0.0
        for sign in (+1, -1):
            a = 0.0
            while a < 90.0:
                a += 0.5
                args = (a * sign * DEG, 0.0) if axis == 0 else (0.0, a * sign * DEG)
                if not reachable(geom, *args):
                    break
            edge = (a - 0.5) * sign
            if sign > 0:
                hi = edge
            else:
                lo = edge
        print(f"   {label:>5}: {lo:+6.1f} deg  ..  {hi:+6.1f} deg")

    print("\n   combined envelope (pitch x roll grid, 5 deg steps):")
    hdr = "        roll:" + "".join(f"{r:>6}" for r in range(-30, 31, 5))
    print(hdr)
    for p in range(-30, 31, 5):
        row = f"   pitch {p:+4d}:"
        for r in range(-30, 31, 5):
            row += "     ." if reachable(geom, p * DEG, r * DEG) else "     X"
        print(row)
    print("   ( . = reachable,  X = a pleuel runs out of travel )")

    # The linkage envelope above is usually NOT the binding limit -- the
    # Kreuzgelenk is. Report how much margin is left inside its cone.
    lim = geom.max_tilt
    worst_reach = worst_cond = 0.0
    at_r = at_c = (0.0, 0.0)
    for p in np.arange(-90, 90.1, 1.0):
        for r in np.arange(-90, 90.1, 1.0):
            if tilt_angle(p * DEG, r * DEG) > lim:
                continue
            rr = max(reach_ratio(m, foot_anchor_world(geom, m, p * DEG, r * DEG))
                     for m in (geom.upper, geom.lower))
            cc = np.linalg.cond(jacobian(geom, p * DEG, r * DEG))
            if rr > worst_reach:
                worst_reach, at_r = rr, (p, r)
            if cc > worst_cond:
                worst_cond, at_c = cc, (p, r)
    print(f"\n   Kreuzgelenk cone limit: {lim/DEG:.0f} deg total tilt "
          f"(geometry.max_tilt). Inside it:")
    print(f"     worst |K/R| = {worst_reach:.3f}  at pitch={at_r[0]:+.0f} roll={at_r[1]:+.0f}"
          f"   (1.0 = the linkage's own limit)")
    print(f"     worst cond  = {worst_cond:.3f}  at pitch={at_c[0]:+.0f} roll={at_c[1]:+.0f}")
    box = tilt_angle(lim, lim) / DEG
    print(f"   NOTE: pitch and roll BOTH at {lim/DEG:.0f} deg is {box:.1f} deg of total tilt, "
          f"outside the cone.")
    print(f"         Use mapping.clamp_pose() -- a naive per-axis box clamp does not hold.")

    # --- 4. transmission ----------------------------------------------------
    print("\n4. Transmission: deg motor per deg foot  (d theta_motor / d theta_joint)")
    print(f"   {'pose':>14}  {'dU/dp':>7} {'dU/dr':>7} {'dL/dp':>7} {'dL/dr':>7}   {'cond':>6}")
    for p, r in [(0, 0), (10, 0), (-10, 0), (0, 10), (0, -10),
                 (10, 10), (-10, -10), (20, 0), (0, 20)]:
        if not reachable(geom, p * DEG, r * DEG):
            print(f"   p={p:+3d} r={r:+3d}    -- unreachable --")
            continue
        J = jacobian(geom, p * DEG, r * DEG)
        c = np.linalg.cond(J)
        flag = "" if c < 5 else "  <-- poorly conditioned"
        print(f"   p={p:+3d} r={r:+3d}      {J[0,0]:+7.3f} {J[0,1]:+7.3f} "
              f"{J[1,0]:+7.3f} {J[1,1]:+7.3f}   {c:6.2f}{flag}")
    print("   cond ~1 = balanced; large cond = near-singular, foot gets sloppy in one direction.")

    # --- 5. left vs right ---------------------------------------------------
    print("\n5. Left vs right leg, same commanded foot pose")
    print(f"   {'pose':>12}   {'R upper':>9} {'R lower':>9}   {'L upper':>9} {'L lower':>9}")
    for p, r in [(10, 0), (-10, 0), (0, 10), (0, -10)]:
        rA, rB = joint_to_motor(RIGHT_LEG, p * DEG, r * DEG)
        lA, lB = joint_to_motor(LEFT_LEG,  p * DEG, r * DEG)
        print(f"   p={p:+3d} r={r:+3d}    {rA/DEG:+9.3f} {rB/DEG:+9.3f}   "
              f"{lA/DEG:+9.3f} {lB/DEG:+9.3f}")
    print("   Both shafts point along -x, which the sagittal mirror leaves unchanged, so")
    print("   +theta is the SAME physical rotation sense on both legs (identical hardware,")
    print("   identical driver sign). Consequence: a mirror-symmetric pose needs OPPOSITE")
    print("   motor signs on the two legs for pitch, and matching signs for roll.")


if __name__ == "__main__":
    main()
