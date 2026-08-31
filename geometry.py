"""
Geometry of the ankle parallel mechanism.

Coordinate convention (Pivot frame, fixed to shin at the universal-joint pivot):
  +x : forward
  +y : right
  +z : up
  Origin: center of universal bearing (ankle pivot)

Units: meters. Measurements are taken in cm and converted via the CM constant.
The foot is in neutral pose (pitch = 0, roll = 0) when measurements were taken.

Joint convention (URDF tree):
  shin -> pitch_joint (around y) -> dummy_link -> roll_joint (around x) -> foot
  So: pitch is applied first, roll second (relative to the rotated dummy frame).

There are two motors per leg ("upper" and "lower" -- the upper one sits higher
on the shin). Each motor drives ONE foot-side pleuel anchor via a crank arm
and a rigid pleuel rod.

Naming:
  P_shaft     : a point on the MOTOR ROTATION AXIS (centerline of the motor
                shaft). NOT the crank attachment, NOT the motor body. Just any
                point on the imaginary line the shaft rotates around.
  n_shaft     : direction of the motor rotation axis (unit vector)
  P_crank_0   : the PHYSICAL ATTACHMENT POINT where the pleuel mounts to the
                crank arm, in neutral pose. This is OFFSET from the shaft axis
                by the crank radius r.
  P_anchor_0  : the PHYSICAL ATTACHMENT POINT where the pleuel mounts to the
                foot, in neutral pose.
  L           : pleuel length (sphere-to-sphere distance between the two
                pleuel-end ball-joint centers).
"""

from dataclasses import dataclass
import numpy as np


CM = 0.01    # convert cm -> m
MM = 0.001   # convert mm -> m


@dataclass(frozen=True)
class MotorGeometry:
    """One motor + its crank arm + its pleuel + its foot anchor.

    All points are 3D vectors in the pivot frame, in meters.

    P_shaft may be ANY point on the motor's rotation axis -- the code
    auto-projects onto the perpendicular plane through P_crank_0 to find
    the true crank-circle center. This makes the geometry input forgiving
    of axial measurement choice.

    Important: P_shaft is on the motor SHAFT CENTERLINE. It is NOT the
    same as P_crank_0 (which is the crank-arm endpoint where the pleuel
    attaches). P_crank_0 is offset from the shaft axis by the crank
    radius r -- that's the whole point of the crank arm.
    """
    P_shaft:    np.ndarray   # any point on the motor's rotation axis
    n_shaft:    np.ndarray   # direction of the motor's rotation axis
    P_crank_0:  np.ndarray   # pleuel-attachment on crank-arm, NEUTRAL pose
    P_anchor_0: np.ndarray   # pleuel-attachment on foot,     NEUTRAL pose
    L: float                 # pleuel length (ball-center to ball-center)

    def __post_init__(self):
        """Derive n_unit / C / r / u_hat / v_hat ONCE, here.

        These used to be plain @property, which meant they were recomputed on
        every single access -- and they cascade (v_hat -> n_unit + u_hat ->
        C -> n_unit), so one crank_tip_world() call re-derived the whole chain
        four times over. That cost ~53 us of the 58 us that call took, and it
        dominated the Newton solver in mapping.py. Now it is ~0.1 us.
        """
        set_ = lambda k, v: object.__setattr__(self, k, v)
        for k in ("P_shaft", "n_shaft", "P_crank_0", "P_anchor_0"):
            a = np.asarray(getattr(self, k), dtype=float)
            a.flags.writeable = False
            set_(k, a)
        set_("L", float(self.L))

        n = self.n_shaft / np.linalg.norm(self.n_shaft)
        C = self.P_shaft + float(np.dot(self.P_crank_0 - self.P_shaft, n)) * n
        arm = self.P_crank_0 - C
        r = float(np.linalg.norm(arm))
        if r < 1e-9:
            raise ValueError("crank radius is zero: P_crank_0 sits on the motor axis")
        u = arm / r
        v = np.cross(n, u)
        for k, val in (("_n_unit", n), ("_C", C), ("_u_hat", u), ("_v_hat", v)):
            val.flags.writeable = False
            set_(k, val)
        set_("_r", r)

    @property
    def n_unit(self) -> np.ndarray:
        """Motor axis as a true unit vector."""
        return self._n_unit

    @property
    def C(self) -> np.ndarray:
        """Center of the crank circle: the projection of P_crank_0 onto the
        motor rotation axis. Geometrically correct regardless of where
        P_shaft was sampled along the axis."""
        return self._C

    @property
    def r(self) -> float:
        """Crank radius: perpendicular distance from motor axis to crank tip.
        Equivalently: the physical length of the crank arm."""
        return self._r

    @property
    def u_hat(self) -> np.ndarray:
        """Unit vector from crank-circle center toward crank tip in neutral.
        Defines theta_motor = 0. Perpendicular to n_shaft by construction."""
        return self._u_hat

    @property
    def v_hat(self) -> np.ndarray:
        """Orthogonal companion to u_hat in the crank swing-plane.
        v_hat = n x u_hat  (right-hand rule: theta increases CCW when
        looking along -n_shaft)."""
        return self._v_hat

    @property
    def sanity_check(self) -> float:
        """In neutral pose, |P_crank_0 - P_anchor_0| must equal L."""
        return float(np.linalg.norm(self.P_crank_0 - self.P_anchor_0))


@dataclass(frozen=True)
class AnkleGeometry:
    """Full ankle of one leg: two motors + universal-joint axes."""
    upper: MotorGeometry     # the upper motor (higher on the shin)
    lower: MotorGeometry     # the lower motor
    pitch_axis: np.ndarray   # rotation axis of pitch joint (in pivot frame)
    roll_axis:  np.ndarray   # rotation axis of roll joint  (in pivot frame)
    side: str                # "left" or "right" leg
    max_tilt: float = np.deg2rad(30.0)
    """Kreuzgelenk travel limit, as a CONE on the total shaft deflection.

    A universal joint limits the angle between the two shafts, not pitch and
    roll separately. pitch=30 and roll=30 at the same time is 41.4 deg of
    total tilt (cos_tilt = cos_pitch * cos_roll), which a +/-30 deg
    Kreuzgelenk cannot do. Use mapping.clamp_pose() to project a commanded
    pose onto this cone before calling joint_to_motor."""


    def __post_init__(self):
        """Cache the unit joint axes; mapping._residuals_and_jacobian reads
        them on every Newton evaluation."""
        for name in ("pitch", "roll"):
            a = np.asarray(getattr(self, f"{name}_axis"), dtype=float)
            a = a / np.linalg.norm(a)
            a.flags.writeable = False
            object.__setattr__(self, f"{name}_unit", a)


# ---------------------------------------------------------------------------
# Concrete geometry: RIGHT leg, CAD measurements (2026-08-30)
# ---------------------------------------------------------------------------
# NEW LEG DESIGN: both motor output shafts now point BACKWARD (-x) instead of
# sideways (+/-y). Both motor axes lie on the shin centerline (y = 0) at
# x = -28.694, stacked 75 mm apart in z. Each crank arm reaches out sideways
# (r = 56.8 mm) and drives a near-vertical pleuel down to the foot.
#
#   upper motor: crank points -y at neutral -> drives the LEFT  (-y) foot anchor
#   lower motor: crank points +y at neutral -> drives the RIGHT (+y) foot anchor
#
# Measured in Fusion, World XYZ Delta from the Kreuzgelenk pitch/roll axis
# intersection. Both pleuel lengths reproduce |P_crank_0 - P_anchor_0| to
# 0.000 mm, so all points come from one consistent neutral pose.
#
# Previous design (shafts along +/-y, r ~ 22-24 mm) is kept in
# geometry_OLD_2026-06-02.py.bak for reference.

RIGHT_LEG = AnkleGeometry(
    upper=MotorGeometry(
        P_shaft    = np.array([ -28.694,   0.000, 253.959]) * MM,
        n_shaft    = np.array([  -1.0,     0.0,     0.0]),
        P_crank_0  = np.array([ -43.194, -56.800, 253.959]) * MM,
        P_anchor_0 = np.array([ -55.000, -60.550, -15.000]) * MM,
        L          = 269.244 * MM,
    ),
    lower=MotorGeometry(
        P_shaft    = np.array([ -28.694,   0.000, 178.959]) * MM,
        n_shaft    = np.array([  -1.0,     0.0,     0.0]),
        P_crank_0  = np.array([ -43.194,  56.800, 178.959]) * MM,
        P_anchor_0 = np.array([ -55.000,  60.550, -15.000]) * MM,
        L          = 194.354 * MM,
    ),
    pitch_axis = np.array([0.0, 1.0, 0.0]),
    roll_axis  = np.array([1.0, 0.0, 0.0]),
    side       = "right",
)


def mirror_leg(source: AnkleGeometry, new_side: str = None) -> AnkleGeometry:
    """Build the opposite leg by mirroring across the sagittal (xz) plane.

    Mirrored:     motor positions, motor axes, crank attachments, foot anchors.
    NOT mirrored: pitch_axis and roll_axis.

    Why: in the standard URDF convention, both legs' shin frames have the
    same local-axis orientation, so the joint axes stay (0,1,0) and (1,0,0)
    for both. The policy then drives both legs symmetrically with one
    sign convention. If your URDF actually flips joint axes for the other
    leg, override pitch_axis/roll_axis explicitly when constructing it.
    """

    def mirror_point(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], -p[1], p[2]])

    def mirror_motor(m: MotorGeometry) -> MotorGeometry:
        return MotorGeometry(
            P_shaft    = mirror_point(m.P_shaft),
            n_shaft    = mirror_point(m.n_shaft),
            P_crank_0  = mirror_point(m.P_crank_0),
            P_anchor_0 = mirror_point(m.P_anchor_0),
            L          = m.L,
        )

    if new_side is None:
        new_side = "right" if source.side == "left" else "left"

    return AnkleGeometry(
        upper      = mirror_motor(source.upper),
        lower      = mirror_motor(source.lower),
        pitch_axis = source.pitch_axis.copy(),
        roll_axis  = source.roll_axis.copy(),
        side       = new_side,
        max_tilt   = source.max_tilt,
    )


LEFT_LEG = mirror_leg(RIGHT_LEG, new_side="left")


# ---------------------------------------------------------------------------
# Rotation helpers (active rotations of points about an axis through origin)
# ---------------------------------------------------------------------------

def rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation matrix about a unit axis by `angle` radians (Rodrigues)."""
    ax = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [   0.0, -ax[2],  ax[1]],
        [ ax[2],    0.0, -ax[0]],
        [-ax[1],  ax[0],    0.0],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def foot_rotation(geom: AnkleGeometry, theta_p: float, theta_r: float) -> np.ndarray:
    """Rotation matrix taking a point at neutral to its rotated position.

    URDF tree: pitch first (around pitch_axis), then roll (around the rotated
    roll axis). Composed as R_total = R_pitch @ R_roll when expressed in the
    parent (pivot) frame: the roll rotation is applied first to the *point*,
    then the pitch.

    p_world = R_pitch @ R_roll @ p_neutral
    """
    R_p = rot_about_axis(geom.pitch_axis, theta_p)
    R_r = rot_about_axis(geom.roll_axis,  theta_r)
    return R_p @ R_r


if __name__ == "__main__":
    for leg in (LEFT_LEG, RIGHT_LEG):
        print(f"=== {leg.side} leg ===")
        for name, m in (("upper", leg.upper), ("lower", leg.lower)):
            print(f"  {name}: r = {m.r*100:.2f} cm, "
                  f"L = {m.L*100:.2f} cm, "
                  f"|P_crank_0 - P_anchor_0| = {m.sanity_check*100:.2f} cm "
                  f"(should be ~L)")
