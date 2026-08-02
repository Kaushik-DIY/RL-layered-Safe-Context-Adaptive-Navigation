"""Demo scene definitions (ROS-free, like the rest of `core`).

Only DATA and pure helpers live here: the Gazebo world generator, the offline
verifier, the ROS scene-director node and the launch file all import the SAME
scene definition, so the geometry can never drift between them.
"""
