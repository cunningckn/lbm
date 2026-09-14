"""RoboCOIN multi-embodiment LeRobot v2.1 reader."""
from ._lerobot_bind import bind_lerobot
from lbm.action_space import unimanual_joint
# Common camera names across RoboCOIN; missing streams are handled by scanner.
NAME,SPEC,scan,read_vectors,read_frames=bind_lerobot('robocoin','franka',('observation.images.cam_head_rgb','observation.images.cam_left_wrist_rgb','observation.images.cam_right_wrist_rgb'),8,8,30.0,kind='lerobot',action_space=unimanual_joint())
