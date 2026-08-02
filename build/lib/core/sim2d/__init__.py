"""Lightweight 2D kinematic simulator + social-force pedestrians (plan D1, D7).

Runs 1000x+ real-time, vectorizes across envs -> all RL training happens here.
Gazebo is used only for validation/video. Shares observation + MPC + CBF code with
Gazebo, so the transfer gap is purely dynamics/behavior fidelity (a measured result).
"""
