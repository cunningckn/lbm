"""Frozen reference parity plus explicit coordinate and sparse camera contracts."""

import unittest

import numpy as np
from molmo_motion_cache.geometry224 import content_mask, resize224, select_camera, transform_points
from PIL import Image, ImageOps


def reference(image, image_size=224, pad_value=0):
    # Frozen PI0 function body, source SHA recorded in geometry224.REFERENCE.
    if image_size is None or image.shape[:2] == (image_size, image_size):
        return image
    pil_image = Image.fromarray(image)
    fitted = ImageOps.contain(pil_image, (image_size, image_size), method=Image.Resampling.BILINEAR)
    canvas = Image.new(pil_image.mode, (image_size, image_size), color=pad_value)
    canvas.paste(fitted, ((image_size - fitted.width) // 2, (image_size - fitted.height) // 2))
    return np.array(canvas)


class GeometryTest(unittest.TestCase):
    def test_frozen_reference_all_shapes_patterns(self):
        for w, h in [(624, 352), (854, 480), (640, 480), (480, 640), (512, 512), (641, 479), (224, 224)]:
            yy, xx = np.indices((h, w))
            for rgb in [
                np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype("uint8"),
                np.repeat((((xx // 3 + yy // 3) % 2) * 255)[..., None], 3, -1).astype("uint8"),
            ]:
                out, g = resize224(rgb, source_size=[w, h])
                np.testing.assert_array_equal(out, reference(rgb))
                self.assertEqual(out.shape, (224, 224, 3))
                self.assertTrue((out[~content_mask(g)] == 0).all())
                if (w, h) == (641, 479):
                    self.assertEqual(g["content_size"], [224, 167])
                    self.assertEqual(g["padding"], [0, 28, 0, 29])

    def test_center_matrix_projection_and_nan(self):
        _, g = resize224(np.zeros((479, 641, 3), dtype="uint8"))
        p = np.array([[0.0, 0.0], [640.0, 478.0], [np.nan, 3.0]])
        q = transform_points(p, g)
        sx, sy = 224 / 641, 167 / 479
        np.testing.assert_allclose(q[:2], (p[:2] + 0.5) * [sx, sy] - 0.5 + [0, 28])
        self.assertTrue(np.isnan(q[2, 0]))
        k = np.array([[400, 0, 320], [0, 400, 239], [0, 0, 1.0]])
        xyz = np.array([[0.1, 0.2, 1.0], [-0.3, 0.1, 2.0]])
        uvw = xyz @ k.T
        out = xyz @ (np.array(g["A"]) @ k).T
        np.testing.assert_allclose(transform_points(uvw[:, :2] / uvw[:, 2:], g), out[:, :2] / out[:, 2:])

    def test_sparse_camera_and_rejection(self):
        full = {
            "camera_pose_indices": np.array([9, 2]),
            "camera_poses": np.stack([np.eye(4) * 9, np.eye(4) * 2]),
            "camera_intrinsics_static": np.eye(3)[None],
            "camera_intrinsics_dynamic": np.empty((0, 4)),
        }
        poses, k = select_camera(full, [2, 9, 2])
        np.testing.assert_array_equal(poses[:, 0, 0], [2, 9, 2])
        self.assertEqual(k.shape, (3, 3, 3))
        with self.assertRaises(ValueError):
            select_camera(full, [0])
        with self.assertRaises(ValueError):
            resize224(np.zeros((4, 5, 3), dtype="uint8"), source_size=[4, 5])
        with self.assertRaises(ValueError):
            resize224(np.zeros((4, 5, 3)))


if __name__ == "__main__":
    unittest.main()
