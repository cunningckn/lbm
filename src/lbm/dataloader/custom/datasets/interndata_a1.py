"""InternData-A1 reader. Native downloads contain LeRobot v3 archives; the\nreader follows the standard packed parquet/video layout after extraction."""
from ._lerobot_bind import bind_lerobot
from lbm.action_space import unimanual_joint
NAME,SPEC,scan,read_vectors,read_frames=bind_lerobot('interndata_a1','agibot_genie1',('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'),8,8,30.0,kind='lerobot',action_space=unimanual_joint(rep="abs"))
