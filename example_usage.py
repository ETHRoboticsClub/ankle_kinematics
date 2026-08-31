"""
Minimal end-to-end usage example for the controls team.

Run:
    python3.11 example_usage.py
"""

import numpy as np
from geometry import LEFT_LEG
from mapping import joint_to_motor, motor_to_joint


def main():
    # Simulated control state
    prev_motor = (0.0, 0.0)   # last commanded motor angles (rad)
    prev_joint = (0.0, 0.0)   # last observed foot pose       (rad)

    # 1) Action pipeline: the policy outputs a target foot pose
    pitch_target = np.deg2rad(8.0)    # rad
    roll_target  = np.deg2rad(-3.0)   # rad

    theta_A_cmd, theta_B_cmd = joint_to_motor(
        LEFT_LEG, pitch_target, roll_target, theta_prev=prev_motor,
    )
    print(f"Target foot pose : pitch={np.rad2deg(pitch_target):+6.2f}°, "
          f"roll={np.rad2deg(roll_target):+6.2f}°")
    print(f"Motor command    : A={np.rad2deg(theta_A_cmd):+7.2f}°, "
          f"B={np.rad2deg(theta_B_cmd):+7.2f}°")
    prev_motor = (theta_A_cmd, theta_B_cmd)

    # 2) Observation pipeline: encoders report current motor positions,
    #    we convert back to foot pose for the policy.
    theta_A_meas = theta_A_cmd        # pretend the motors got there exactly
    theta_B_meas = theta_B_cmd

    pitch_meas, roll_meas = motor_to_joint(
        LEFT_LEG, theta_A_meas, theta_B_meas, theta_prev=prev_joint,
    )
    print(f"Encoder readback : A={np.rad2deg(theta_A_meas):+7.2f}°, "
          f"B={np.rad2deg(theta_B_meas):+7.2f}°")
    print(f"Reconstructed    : pitch={np.rad2deg(pitch_meas):+6.2f}°, "
          f"roll={np.rad2deg(roll_meas):+6.2f}°")
    prev_joint = (pitch_meas, roll_meas)

    err_p = np.rad2deg(pitch_meas - pitch_target)
    err_r = np.rad2deg(roll_meas - roll_target)
    print(f"Round-trip error : Δpitch={err_p:+.2e}°, Δroll={err_r:+.2e}°")


if __name__ == "__main__":
    main()
