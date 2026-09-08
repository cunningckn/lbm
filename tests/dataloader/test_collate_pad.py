import numpy as np

from lbm.dataloader.pad import collate_fn, pad_loader_batch


def _sample(*, n_cams, t_act, d_act, t_hist=1, tag="aloha", cams=None, d_state=None):
    cams = cams or tuple(f"cam{i}" for i in range(n_cams))
    image = [np.zeros((t_hist, 8, 8, 3), dtype=np.uint8) for _ in cams]
    image[0][:] = 7
    return {
        "image": image,
        "action": np.ones((t_act, d_act), dtype=np.float32),
        "state": np.zeros((1, d_state or d_act), dtype=np.float32),
        "lang": "pick",
        "robot_tag": tag,
        "camera_keys": cams,
    }


def test_collate_pads_action_and_cameras():
    a = _sample(n_cams=3, t_act=50, d_act=14, tag="aloha", cams=("cam_high", "cam_left_wrist", "cam_right_wrist"))
    b = _sample(n_cams=2, t_act=10, d_act=7, tag="franka", cams=("primary_image", "wrist_image"))
    out = collate_fn([a, b])
    assert out["action"].shape == (2, 50, 14)
    assert out["action_mask"].shape == (2, 50, 14)
    assert out["action_mask"][0].all()
    assert out["action_mask"][1, :10, :7].all()
    assert not out["action_mask"][1, 10:, :].any()
    assert not out["action_mask"][1, :10, 7:].any()
    assert out["image"].shape[1] == 5
    assert out["camera_mask"].shape == (2, 5)
    assert int(out["camera_mask"][0].sum()) == 3
    assert int(out["camera_mask"][1].sum()) == 2
    assert out["embodiment_id"].tolist() == [7, 25]


def test_pad_loader_batch_state():
    a = _sample(n_cams=1, t_act=4, d_act=14, d_state=14)
    b = _sample(n_cams=1, t_act=4, d_act=7, d_state=8)
    out = pad_loader_batch([a, b])
    assert out["state"].shape[-1] == 14
    assert out["state_mask"][1, 0, :8].all()
    assert not out["state_mask"][1, 0, 8:].any()
