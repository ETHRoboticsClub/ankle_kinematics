"""
Forward kinematics of the two moving pleuel-attachment points:

  - foot_anchor_world(theta_p, theta_r):
        Where the foot-side pleuel attachment sits when the foot is
        tilted to (pitch, roll).

  - crank_tip_world(theta_motor):
        Where the crank-side pleuel attachment sits when the motor
        has rotated by theta_motor (from its neutral pose).

Both are pure geometry. No physics, no time, no mass. They answer
"where is this physical attachment point in the pivot frame, given
these angles?"

Output is always a 3D point in the pivot frame (meters).
"""

import numpy as np
try:                       # imported as a package
    from .geometry import AnkleGeometry, MotorGeometry, foot_rotation
except ImportError:        # run directly as a script
    from geometry import AnkleGeometry, MotorGeometry, foot_rotation


def foot_anchor_world(geom: AnkleGeometry, motor: MotorGeometry,
                      theta_p: float, theta_r: float) -> np.ndarray:
    """Pivot-frame position of the FOOT-SIDE pleuel attachment after the
    foot tilts to (pitch, roll).

    The foot is a rigid body rotating around the pivot, so its attachment
    point just rides along: apply the (pitch, roll) rotation to the
    neutral attachment position.

    By hand:
      1. Build R_pitch(theta_p) and R_roll(theta_r) as 3x3 matrices.
      2. Compose: R = R_pitch @ R_roll  (URDF tree shin -> pitch -> roll -> foot,
         so roll is applied to the point first, then pitch).
      3. Apply to the neutral anchor: result = R @ P_anchor_0.
    """
    R = foot_rotation(geom, theta_p, theta_r)
    return R @ motor.P_anchor_0


def crank_tip_world(motor: MotorGeometry, theta_motor: float) -> np.ndarray:
    """Pivot-frame position of the CRANK-SIDE pleuel attachment after the
    motor rotates by theta_motor.

    As the motor spins, the crank-arm tip traces a circle:
      - centered on the projection of P_crank_0 onto the motor axis (= C),
      - radius r (= the physical length of the crank arm),
      - lying in the plane perpendicular to n_shaft,
      - parameterized by theta_motor (= 0 at the neutral pose).

    Two orthogonal unit vectors u_hat and v_hat span that plane:
      u_hat = direction from C toward P_crank_0     (defines theta_motor = 0)
      v_hat = n_shaft x u_hat                       (right-hand rule)

    Any point on the circle:
      crank_tip(theta) = C + r * (cos(theta) * u_hat + sin(theta) * v_hat)

    By hand:
      1. Compute cos(theta) and sin(theta).
      2. Form the in-plane displacement r*cos*u_hat + r*sin*v_hat.
      3. Add the circle center C.
    """
    c, s = np.cos(theta_motor), np.sin(theta_motor)
    return motor.C + motor.r * (c * motor.u_hat + s * motor.v_hat)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (if anyone imports the old names)
# ---------------------------------------------------------------------------
P_foot = foot_anchor_world
P_kurbel = crank_tip_world


if __name__ == "__main__":
    from geometry import LEFT_LEG

    print("Sanity: at neutral, computed positions match the stored neutrals.")
    for name, m in (("upper", LEFT_LEG.upper), ("lower", LEFT_LEG.lower)):
        pk = crank_tip_world(m, 0.0)
        pf = foot_anchor_world(LEFT_LEG, m, 0.0, 0.0)
        err_k = np.linalg.norm(pk - m.P_crank_0)
        err_f = np.linalg.norm(pf - m.P_anchor_0)
        print(f"  {name}: crank_tip(0) error = {err_k*1000:.4f} mm, "
              f"foot_anchor(0,0) error = {err_f*1000:.4f} mm")

    print()
    print("At neutral, |crank_tip - foot_anchor| should equal L (modulo measurement noise).")
    for name, m in (("upper", LEFT_LEG.upper), ("lower", LEFT_LEG.lower)):
        pk = crank_tip_world(m, 0.0)
        pf = foot_anchor_world(LEFT_LEG, m, 0.0, 0.0)
        d = np.linalg.norm(pk - pf)
        print(f"  {name}: dist = {d*100:.2f} cm vs L = {m.L*100:.2f} cm")

    print()
    print("Upper motor sweep: crank tip relative to its circle center C, in cm.")
    m = LEFT_LEG.upper
    for deg in (-90, -45, 0, 45, 90):
        theta = np.deg2rad(deg)
        offset = (crank_tip_world(m, theta) - m.C) * 100
        print(f"  theta = {deg:+4d} deg -> offset = "
              f"({offset[0]:+.2f}, {offset[1]:+.2f}, {offset[2]:+.2f}) cm, "
              f"|offset| = {np.linalg.norm(offset):.2f} cm (should be r = {m.r*100:.2f})")
