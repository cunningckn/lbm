"""MolmoAct LeRobot v2.1 reader (Franka, 10 Hz)."""
from ._lerobot_bind import bind_lerobot
from lbm.action_space import unimanual_joint
NAME,SPEC,scan,read_vectors,read_frames=bind_lerobot('molmoact','franka',('first_view','second_view','wrist_image'),7,7,10.0,kind='lerobot',state_columns=('state',),action_columns=('actions',),language_column='task',action_space=unimanual_joint(rep="abs"))
