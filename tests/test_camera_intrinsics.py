import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.graphics_utils import focal2fov, getProjectionMatrix, getProjectionMatrixFromIntrinsics


def _pixels_from_projection(matrix, points, width, height):
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=1)
    hom = points_h @ matrix
    ndc = hom[:, :3] / hom[:, 3:4]
    u = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
    return torch.stack([u, v], dim=-1)


def test_intrinsic_projection_matches_pinhole_pixels():
    width, height = 1280, 720
    fx, fy = 910.0, 880.0
    cx, cy = 570.25, 390.75
    points = torch.tensor(
        [
            [0.10, -0.05, 1.20],
            [-0.20, 0.15, 2.00],
            [0.35, 0.22, 3.50],
        ],
        dtype=torch.float32,
    )

    projection = getProjectionMatrixFromIntrinsics(
        0.01, 100.0, fx, fy, cx, cy, width, height
    ).transpose(0, 1)
    actual = _pixels_from_projection(projection, points, width, height)
    expected = torch.stack(
        [
            fx * points[:, 0] / points[:, 2] + cx,
            fy * points[:, 1] / points[:, 2] + cy,
        ],
        dim=-1,
    )

    assert torch.allclose(actual, expected, atol=1e-4)


def test_center_intrinsics_match_legacy_fov_projection_pixels():
    width, height = 1024, 768
    fx, fy = 700.0, 710.0
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.3, -0.1, 1.7],
            [-0.2, 0.4, 2.4],
        ],
        dtype=torch.float32,
    )

    legacy = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1)
    intrinsic = getProjectionMatrixFromIntrinsics(
        0.01, 100.0, fx, fy, cx, cy, width, height
    ).transpose(0, 1)

    assert torch.allclose(
        _pixels_from_projection(intrinsic, points, width, height),
        _pixels_from_projection(legacy, points, width, height),
        atol=1e-4,
    )


def _load_camera_utils_with_stubbed_scene(monkeypatch):
    scene_module = types.ModuleType("scene")
    camera_module = types.ModuleType("scene.cameras")
    camera_module.Camera = object
    monkeypatch.setitem(sys.modules, "scene", scene_module)
    monkeypatch.setitem(sys.modules, "scene.cameras", camera_module)

    spec = importlib.util.spec_from_file_location(
        "camera_utils_under_test", ROOT / "utils" / "camera_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resize_intrinsics_uses_pixel_center_convention(monkeypatch):
    module = _load_camera_utils_with_stubbed_scene(monkeypatch)
    cam_info = SimpleNamespace(width=1000, height=750, fx=800.0, fy=810.0, cx=420.0, cy=360.0)
    resolution = (625, 333)
    sx = resolution[0] / cam_info.width
    sy = resolution[1] / cam_info.height

    fx, fy, cx, cy = module._scaled_camera_intrinsics(cam_info, resolution)

    assert fx == pytest.approx(cam_info.fx * sx)
    assert fy == pytest.approx(cam_info.fy * sy)
    assert cx == pytest.approx((cam_info.cx + 0.5) * sx - 0.5)
    assert cy == pytest.approx((cam_info.cy + 0.5) * sy - 0.5)


def _load_dataset_readers_with_stubs(monkeypatch):
    scene_module = types.ModuleType("scene")
    scene_module.__path__ = []

    colmap_loader = types.ModuleType("scene.colmap_loader")
    for name in (
        "read_extrinsics_text",
        "read_intrinsics_text",
        "read_extrinsics_binary",
        "read_intrinsics_binary",
        "read_points3D_binary",
        "read_points3D_text",
    ):
        setattr(colmap_loader, name, lambda *args, **kwargs: None)
    colmap_loader.qvec2rotmat = lambda qvec: np.eye(3)

    gaussian_model = types.ModuleType("scene.gaussian_model")

    class BasicPointCloud(NamedTuple):
        points: np.array
        colors: np.array
        normals: np.array

    gaussian_model.BasicPointCloud = BasicPointCloud

    plyfile = types.ModuleType("plyfile")
    plyfile.PlyData = object
    plyfile.PlyElement = object

    monkeypatch.setitem(sys.modules, "scene", scene_module)
    monkeypatch.setitem(sys.modules, "scene.colmap_loader", colmap_loader)
    monkeypatch.setitem(sys.modules, "scene.gaussian_model", gaussian_model)
    monkeypatch.setitem(sys.modules, "plyfile", plyfile)

    spec = importlib.util.spec_from_file_location(
        "dataset_readers_under_test", ROOT / "scene" / "dataset_readers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colmap_pinhole_intrinsic_parsing(monkeypatch):
    module = _load_dataset_readers_with_stubs(monkeypatch)

    assert module._parse_colmap_pinhole_intrinsics(
        SimpleNamespace(model="SIMPLE_PINHOLE", params=[500.0, 320.0, 240.0])
    ) == (500.0, 500.0, 320.0, 240.0)
    assert module._parse_colmap_pinhole_intrinsics(
        SimpleNamespace(model="PINHOLE", params=[510.0, 520.0, 321.0, 241.0])
    ) == (510.0, 520.0, 321.0, 241.0)
    with pytest.raises(ValueError, match="undistorted pinhole"):
        module._parse_colmap_pinhole_intrinsics(
            SimpleNamespace(model="SIMPLE_RADIAL", params=[500.0, 320.0, 240.0, 0.1])
        )
