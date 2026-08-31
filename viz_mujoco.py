"""
MuJoCo visualization of the parallel ankle mechanism.

Loads the leg (shin + universal cross + foot) from STL meshes, models the
ankle as the two ideal pitch + roll revolute joints (just like the policy
sees it), and overlays the parallel mechanism (motors, crank arms, pleuels)
as DECORATIVE line geoms updated every frame from the analytic mapping.

Use the joint sliders in MuJoCo's GUI (right panel "Joint") to drive pitch
and roll. The mechanism updates live, the motor angles are printed.

Run:
    python3.11 viz_mujoco.py
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from geometry import LEFT_LEG
from kinematics import crank_tip_world, foot_anchor_world
from mapping import joint_to_motor


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_XML = os.path.join(HERE, "ankle_viz.xml")


# RGBA colors (match the MJCF materials for visual coherence)
COLOR_MOTOR  = np.array([0.30, 0.62, 1.00, 1.0])
COLOR_CRANK  = np.array([0.62, 0.78, 1.00, 1.0])
COLOR_PLEUEL = np.array([0.48, 1.00, 0.70, 1.0])
COLOR_ANCHOR = np.array([1.00, 0.42, 0.42, 1.0])


def add_capsule(scn, from_pt, to_pt, radius, rgba):
    """Append a capsule geom between two world-frame points to user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).flatten(),
        rgba=rgba.astype(np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(from_pt, dtype=np.float64),
                         np.asarray(to_pt,   dtype=np.float64))
    scn.ngeom += 1


def add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=rgba.astype(np.float32),
    )
    scn.ngeom += 1


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    pitch_id = model.joint("pitch").id
    roll_id  = model.joint("roll").id
    pitch_qpos_adr = model.jnt_qposadr[pitch_id]
    roll_qpos_adr  = model.jnt_qposadr[roll_id]

    state = {"prev_motors": (0.0, 0.0), "last_print": 0.0}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera setup for a nice angle
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        viewer.cam.distance = 0.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.05]

        # Hide MuJoCo's built-in joint indicators (the huge red/blue arrows)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

        print("MuJoCo viewer running.")
        print("  - Open the 'Joint' panel on the right and drag pitch/roll.")
        print("  - The colored linkage updates live based on joint_to_motor().")
        print("  - Close the window to exit.\n")

        while viewer.is_running():
            step_start = time.time()

            # Read current joint state (driven by user via GUI sliders)
            theta_p = float(data.qpos[pitch_qpos_adr])
            theta_r = float(data.qpos[roll_qpos_adr])

            # Solve the inverse mapping for the current foot pose
            tA, tB = joint_to_motor(LEFT_LEG, theta_p, theta_r,
                                    state["prev_motors"])
            state["prev_motors"] = (tA, tB)

            # Compute all linkage points (in pivot/shin frame == world frame here)
            pk_up = crank_tip_world(LEFT_LEG.upper, tA)
            pk_lo = crank_tip_world(LEFT_LEG.lower, tB)
            pf_up = foot_anchor_world(LEFT_LEG, LEFT_LEG.upper, theta_p, theta_r)
            pf_lo = foot_anchor_world(LEFT_LEG, LEFT_LEG.lower, theta_p, theta_r)
            pm_up = LEFT_LEG.upper.C
            pm_lo = LEFT_LEG.lower.C

            # Reset and rebuild user scene
            viewer.user_scn.ngeom = 0

            # Motor cylinders along the rotation axis (short stub)
            for pm, n in ((pm_up, LEFT_LEG.upper.n_unit),
                          (pm_lo, LEFT_LEG.lower.n_unit)):
                axis_end_a = pm + 0.015 * n
                axis_end_b = pm - 0.015 * n
                add_capsule(viewer.user_scn, axis_end_a, axis_end_b,
                            0.012, COLOR_MOTOR)

            # Crank arms (motor center -> kurbel endpoint)
            add_capsule(viewer.user_scn, pm_up, pk_up, 0.004, COLOR_CRANK)
            add_capsule(viewer.user_scn, pm_lo, pk_lo, 0.004, COLOR_CRANK)
            add_sphere(viewer.user_scn, pk_up, 0.006, COLOR_CRANK)
            add_sphere(viewer.user_scn, pk_lo, 0.006, COLOR_CRANK)

            # Pleuel rods (kurbel endpoint -> foot anchor)
            add_capsule(viewer.user_scn, pk_up, pf_up, 0.003, COLOR_PLEUEL)
            add_capsule(viewer.user_scn, pk_lo, pf_lo, 0.003, COLOR_PLEUEL)

            # Foot anchor markers
            add_sphere(viewer.user_scn, pf_up, 0.006, COLOR_ANCHOR)
            add_sphere(viewer.user_scn, pf_lo, 0.006, COLOR_ANCHOR)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Periodic status print
            now = time.time()
            if now - state["last_print"] > 0.5:
                print(f"\rpitch={np.rad2deg(theta_p):+6.1f}°  "
                      f"roll={np.rad2deg(theta_r):+6.1f}°    "
                      f"-> motor_upper={np.rad2deg(tA):+7.2f}°  "
                      f"motor_lower={np.rad2deg(tB):+7.2f}°    "
                      f"|L_up actual={np.linalg.norm(pk_up - pf_up)*100:.2f}cm "
                      f"L_lo actual={np.linalg.norm(pk_lo - pf_lo)*100:.2f}cm    ",
                      end="", flush=True)
                state["last_print"] = now

            # Cap to ~60 FPS so we don't spin the CPU
            sleep = 1/60 - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
