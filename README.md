# Ankle Parallel-Mechanism — Kinematic Mapping

Adapter layer between the policy (which thinks in foot pose `pitch`/`roll`)
and the hardware (which speaks motor angles `theta_A`/`theta_B`). Stateless,
deterministic, no ML — pure geometry derived from a parallel RSU-style
ankle: two motors per leg, each connected via a rigid pleuel to one anchor
on the foot, with a universal joint at the ankle pivot.

## TL;DR for the controls team

```python
from ankle_kinematics.geometry import LEFT_LEG, RIGHT_LEG
from ankle_kinematics.mapping import joint_to_motor, motor_to_joint, clamp_pose

# Action pipeline (every control tick):
#   ALWAYS clamp_pose first. The Kreuzgelenk only allows 30 deg of total tilt,
#   and an unclamped command is what makes the solver slow and the mapping
#   non-smooth for RL. See "Travel limits" below.
pitch_target, roll_target, hit_limit = clamp_pose(LEFT_LEG, pitch_cmd, roll_cmd)
theta_A_cmd, theta_B_cmd = joint_to_motor(LEFT_LEG, pitch_target, roll_target,
                                          theta_prev=last_motor_commands)

# Observation pipeline (every control tick):
pitch_meas, roll_meas = motor_to_joint(LEFT_LEG, theta_A_encoder, theta_B_encoder,
                                       theta_prev=last_foot_pose)
```

Pass `theta_prev` to keep the solver on a consistent branch across calls.
A reasonable initial value is `(0.0, 0.0)` at startup.

## Conventions (read this before plugging in)

**Pivot frame** (fixed to the shin, origin at the universal-joint center):

| Axis | Direction |
|------|-----------|
| `+x` | forward |
| `+y` | right |
| `+z` | up |

**Units everywhere: meters, radians.** Measurements in `geometry.py` are
written in cm and converted once via a `CM` constant — internal API is SI.

**Joint sign convention** (right-hand rule):

- `pitch` rotates the foot around `+y`. With the default `pitch_axis = (0, 1, 0)`,
  positive pitch makes the toe go **down** (plantarflexion). If you want
  the opposite ("toe up = positive"), set `pitch_axis = (0, -1, 0)` in
  `geometry.py` — the rest of the code Just Works.
- `roll` rotates the foot around `+x`. With the default `roll_axis = (1, 0, 0)`,
  positive roll makes the **right** side of the foot go up.
- **Confirm with the controls team which convention they want before
  deployment** — flipping is one line in `geometry.py`.

**Motor sign convention:** depends on each motor's `n_motor` direction (in
`geometry.py`). The mapping handles the bookkeeping; for the team, the
direction `theta_motor = 0` corresponds to the crank arm pointing in the
direction stored in `u_hat`. In the current geometry both motors' `theta = 0`
is at foot-neutral (modulo measurement noise — see "Calibration" below).

**URDF joint tree assumed:**
```
shin → pitch (around y) → dummy → roll (around x) → foot
```
If the URDF tree is reversed, the rotation composition in `kinematics.foot_rotation`
needs to flip (R_pitch and R_roll swapped). Ping me.

**Left vs. right leg:** `RIGHT_LEG = mirror_leg(LEFT_LEG)` mirrors motor
positions and motor axes across the sagittal (xz) plane. It does **not**
mirror `pitch_axis`/`roll_axis` — both legs use the same joint-axis
convention so the policy drives them symmetrically with one set of signs.
If your URDF actually flips joint axes for the right leg, override
explicitly when constructing `RIGHT_LEG`.

## Files

| File | What's in it |
|------|--------------|
| `geometry.py` | All measured dimensions (shaft axis points, axes, crank attachments, foot anchors, pleuel lengths). Single source of truth — only thing to change when CAD values land. |
| `kinematics.py` | `foot_anchor_world(theta_p, theta_r)` and `crank_tip_world(theta_motor)` — forward kinematics of the two pleuel endpoints. |
| `mapping.py` | `joint_to_motor` (analytic, circle-sphere intersection, per motor independently) and `motor_to_joint` (Newton iteration on the two coupled constraints). |
| `visualize.py` | matplotlib slider-based 3D viewer (for debugging / sanity checks). Run `python3.11 visualize.py`. |

## How it works (one line each)

- Each pleuel imposes one constraint: `|P_kurbel(theta_motor) - P_foot(theta_p, theta_r)| = L`.
- **Inverse mapping** decouples per motor: for each pleuel, given foot pose,
  intersect the crank circle with a sphere of radius `L` around the foot
  anchor → at most 2 solutions, picked by closeness to `theta_prev`.
- **Forward mapping** couples both: 2 equations in 2 unknowns, solved with
  Newton's method (numerical 2x2 Jacobian, converges in 3-5 steps from
  any reasonable `theta_prev`).

## Travel limits (read this if you are writing the controller)

The **Kreuzgelenk allows 30 deg of total shaft tilt**, and that is a **cone,
not a box**. Pitch and roll do not add independently:

    cos(tilt) = cos(pitch) * cos(roll)

so pitch=30 *and* roll=30 is **41.4 deg** of tilt -- past what the joint can
do. A per-axis clamp to +/-30 each does not keep you legal. Use
`clamp_pose(geom, pitch, roll)`, which scales the command down along its own
direction until it sits on the cone, so the direction survives and only the
magnitude is cut. It returns `(pitch, roll, was_clamped)`.

Why it matters beyond not breaking the joint: at the box corner
(pitch -30, roll -30) the *linkage* sits at |K/R| = 0.968, i.e. 97 % of the
way to its own circle-sphere singularity, with the Jacobian condition number
doubling to 1.97. That is where `joint_to_motor` starts clamping to a
tangency point -- the commanded pose stops moving the motors, which reads to
a policy as a dead flat plateau -- and where `motor_to_joint` needs far more
Newton iterations. Inside the 30 deg cone the worst |K/R| is **0.812** and
the worst condition number **1.33**, and the unreachable branch never fires
at all. Verified by sweeping commands from -90 to +90 deg in both axes: 0 of
8100 clamped poses hit it.

`geometry.max_tilt` holds the limit (default 30 deg) and is mirrored to both
legs. **Still open: the motor-side swing limits** -- how far each crank can
turn before it fouls the shin. Those are a separate constraint and are not
modelled yet.

## Performance

Per call, measured on the 2026-08-30 geometry:

| | before | now |
|---|---|---|
| `joint_to_motor`, in bounds | 168 us | 58 us |
| `joint_to_motor`, out of bounds | 188 us | 71 us |
| `motor_to_joint`, warm seed | 176 us | 86 us |
| `motor_to_joint`, cold seed | 2790 us | 413 us |
| `motor_to_joint`, inconsistent motor angles | 3855 us | 506 us |

Full control tick for both legs (clamp + inverse + forward): **291 us, ~3.4 kHz**.
Before, a tick near a travel limit cost ~6 ms, i.e. ~168 Hz.

Three things were wrong:

1. `MotorGeometry.C / r / u_hat / v_hat` were plain `@property`, recomputed on
   every access, and they cascade (`v_hat` -> `n_unit` + `u_hat` -> `C` ->
   `n_unit`). One `crank_tip_world` call re-derived the whole chain and spent
   53 of its 58 us doing it. They are now computed once in `__post_init__`.
2. `motor_to_joint` built its Jacobian by central differences: five residual
   evaluations per iteration. It is now analytic (`_residuals_and_jacobian`,
   exact to 1e-11 against central differences), one evaluation per iteration,
   with the cross products contracted into scalar triple products because
   `np.cross` on 3-vectors costs 16 us a call.
3. The Newton loop seeded only from `theta_prev`. Near a travel limit
   `theta_prev` goes stale and every tick paid cold-seed cost. It now falls
   back to a linearised inverse (`_seed_matrix`) whenever the warm seed's
   residual is large, limits step length, backtracks so a step can never make
   the residual worse, and stops early with `converged=False` when the motor
   angles admit no solution at all instead of burning all 20 iterations.
   Convergence over the whole cone is 2.7 iterations mean, 3 max, from a cold
   seed every time.

`motor_to_joint(..., return_status=True)` returns
`(pitch, roll, converged, iterations)` -- check `converged` on hardware, it is
your encoder-inconsistency detector. Set `mapping.WARN_ON_UNREACHABLE = False`
in vectorised RL envs.

## Calibration status

Geometry in `geometry.py` is **CAD-exact** as of 2026-08-30 (new leg design,
motor shafts pointing backward). The pleuel-length residual is 0.0002 mm on
both rods, and `joint_to_motor(0, 0)` returns 0.000 deg on both motors --
`theta_motor = 0` genuinely *is* foot-neutral now. The ~14 deg / ~9 deg
neutral offset of the old hand-measured geometry is gone.

Run `python3.11 check_geometry.py` after any CAD change: it re-checks the
length residual, the neutral offset, the reachable pitch/roll envelope, the
transmission ratio and Jacobian conditioning, and the left/right sign
relationship, all in one go.

## What's not in here yet (open items)

- Motor angular limits (`theta_min`, `theta_max`) → joint-space limits
  (the *joint*-side Kreuzgelenk cone is done, see "Travel limits")
- Velocity-limit translation through the mapping Jacobian
- A `from_cad(json)` loader so geometry can be hot-swapped without touching code
- Round-trip CI test that runs `forward(inverse(...))` over a grid and fails
  if any case drifts more than 1e-6 rad

## Round-trip sanity check

`mapping.py` contains a built-in test:
```bash
python3.11 mapping.py
```
shows joint→motor sweeps and a forward/inverse round-trip. The round-trip
errors should be ~1e-10 rad (machine precision) for poses near the previous
solution; large pose jumps can hit a different (also valid) branch.

## Dependencies

- `numpy` (mapping + kinematics)
- `matplotlib` (only for `visualize.py`)
- Python 3.11+ recommended

## Contact

Elia Huber
