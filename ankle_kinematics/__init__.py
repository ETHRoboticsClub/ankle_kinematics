"""Ankle parallel-mechanism kinematics.

Maps between foot pose (pitch, roll) and motor angles for a two-motor
parallel RSU ankle. See README.md.

    from ankle_kinematics import LEFT_LEG, RIGHT_LEG, clamp_pose, joint_to_motor

    pitch, roll, hit_limit = clamp_pose(LEFT_LEG, pitch_cmd, roll_cmd)
    theta_A, theta_B = joint_to_motor(LEFT_LEG, pitch, roll, theta_prev=last)
"""

from .geometry import (LEFT_LEG, RIGHT_LEG, AnkleGeometry, MotorGeometry,
                       mirror_leg, foot_rotation, rot_about_axis)
from .kinematics import foot_anchor_world, crank_tip_world
from .mapping import (joint_to_motor, motor_to_joint, clamp_pose, tilt_angle,
                      WARN_ON_UNREACHABLE)

__all__ = [
    "LEFT_LEG", "RIGHT_LEG", "AnkleGeometry", "MotorGeometry", "mirror_leg",
    "foot_rotation", "rot_about_axis", "foot_anchor_world", "crank_tip_world",
    "joint_to_motor", "motor_to_joint", "clamp_pose", "tilt_angle",
    "WARN_ON_UNREACHABLE",
]
