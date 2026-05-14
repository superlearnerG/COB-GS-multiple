#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.read_write_model import (
    read_points3D_binary as read_points3D_binary_with_ids,
    read_points3D_text as read_points3D_text_with_ids,
)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    fx: float
    fy: float
    cx: float
    cy: float
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    depth_scale: float
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def _parse_colmap_pinhole_intrinsics(intr):
    model = str(intr.model).upper()
    params = intr.params
    if model == "SIMPLE_PINHOLE":
        focal = float(params[0])
        return focal, focal, float(params[1]), float(params[2])
    if model == "PINHOLE":
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(
        f"COLMAP camera model '{intr.model}' is not supported for direct loading. "
        "Use an undistorted pinhole COLMAP model (PINHOLE or SIMPLE_PINHOLE)."
    )


def _center_intrinsics_from_fov(FovX, FovY, width, height):
    return (
        float(fov2focal(FovX, width)),
        float(fov2focal(FovY, height)),
        float(width - 1) / 2.0,
        float(height - 1) / 2.0,
    )


def _split_name_keys(name):
    stripped = name.strip()
    if not stripped:
        return set()
    basename = os.path.basename(stripped)
    stem = Path(basename).stem
    return {stripped, basename, stem}


def _read_split_list(list_path):
    split_names = set()
    with open(list_path, "r") as file:
        for line in file:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            split_names.update(_split_name_keys(item))
    return split_names


def _name_in_split(image_name, split_names):
    return bool(_split_name_keys(image_name) & split_names)


def _is_numbered_test_image(image_name):
    stem = Path(os.path.basename(image_name)).stem
    try:
        return int(stem) % 8 == 0
    except ValueError:
        return False


def _resolve_raw_depth_folder(path, depths, use_depth_loss):
    if not use_depth_loss:
        return ""
    depth_dir = depths if depths else "depth"
    depth_folder = depth_dir if os.path.isabs(depth_dir) else os.path.join(path, depth_dir)
    if not os.path.isdir(depth_folder):
        raise FileNotFoundError(f"--use_depth_loss expects raw .npy depth maps under '{depth_folder}'.")
    print(f"[Depth Loss] Loading raw depth maps from {depth_folder}")
    return depth_folder


def _raw_depth_path(depths_folder, image_name):
    if depths_folder == "":
        return ""
    return os.path.join(depths_folder, f"{Path(image_name).stem}.npy")


def _read_points3d_with_ids(path):
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    try:
        return read_points3D_binary_with_ids(bin_path)
    except Exception:
        return read_points3D_text_with_ids(txt_path)


def _select_evenly_spaced(items, max_count):
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count, dtype=int)
    return [items[int(idx)] for idx in indices]


def _estimate_colmap_raw_depth_scale(path, cam_extrinsics, depths_folder, max_views=32, max_points_per_view=12000):
    points3d = _read_points3d_with_ids(path)
    xyz_by_id = {int(point_id): point.xyz for point_id, point in points3d.items()}
    ratios = []
    used_views = 0

    extrinsics = sorted(cam_extrinsics.values(), key=lambda extr: extr.name)
    for extr in _select_evenly_spaced(extrinsics, max_views):
        depth_path = _raw_depth_path(depths_folder, extr.name)
        if not os.path.exists(depth_path):
            continue

        point_ids = np.asarray(extr.point3D_ids)
        xys = np.asarray(extr.xys)
        valid_indices = np.flatnonzero(point_ids != -1)
        if valid_indices.size == 0:
            continue
        if valid_indices.size > max_points_per_view:
            valid_indices = _select_evenly_spaced(valid_indices.tolist(), max_points_per_view)

        matched_xys = []
        matched_xyz = []
        for idx in valid_indices:
            point_id = int(point_ids[idx])
            xyz = xyz_by_id.get(point_id)
            if xyz is None:
                continue
            matched_xys.append(xys[idx])
            matched_xyz.append(xyz)
        if not matched_xyz:
            continue

        raw_depth = np.load(depth_path, mmap_mode="r")
        matched_xys = np.asarray(matched_xys, dtype=np.float64)
        matched_xyz = np.asarray(matched_xyz, dtype=np.float64)
        u = np.rint(matched_xys[:, 0]).astype(np.int64)
        v = np.rint(matched_xys[:, 1]).astype(np.int64)
        in_image = (u >= 0) & (v >= 0) & (u < raw_depth.shape[1]) & (v < raw_depth.shape[0])
        if not np.any(in_image):
            continue

        R = qvec2rotmat(extr.qvec)
        t = np.asarray(extr.tvec, dtype=np.float64)
        z_colmap = (R @ matched_xyz[in_image].T).T[:, 2] + t[2]
        raw_z = np.asarray(raw_depth[v[in_image], u[in_image]], dtype=np.float64)
        valid = np.isfinite(raw_z) & (raw_z > 0.0) & np.isfinite(z_colmap) & (z_colmap > 0.0)
        view_ratios = z_colmap[valid] / raw_z[valid]
        view_ratios = view_ratios[np.isfinite(view_ratios) & (view_ratios > 0.0) & (view_ratios < 100.0)]
        if view_ratios.size == 0:
            continue
        ratios.append(view_ratios)
        used_views += 1

    if not ratios:
        raise RuntimeError(
            "Unable to estimate --depth_scale from COLMAP tracks and raw depth maps. "
            "Pass a positive --depth_scale manually."
        )

    ratios = np.concatenate(ratios)
    if ratios.size < 100:
        raise RuntimeError(
            f"Only {ratios.size} valid COLMAP/raw-depth correspondences were found; "
            "pass a positive --depth_scale manually."
        )

    scale = float(np.median(ratios))
    print(
        "[Depth Loss] Estimated raw-depth scale from COLMAP tracks: "
        f"{scale:.6f} ({ratios.size} samples from {used_views} views; "
        f"p05={np.percentile(ratios, 5):.6f}, p95={np.percentile(ratios, 95):.6f})"
    )
    return scale


def _resolve_depth_scale_for_colmap(path, cam_extrinsics, depths_folder, requested_depth_scale, use_depth_loss):
    if not use_depth_loss:
        return 1.0
    requested_depth_scale = float(requested_depth_scale)
    if requested_depth_scale > 0.0:
        print(f"[Depth Loss] Using manual raw-depth scale: {requested_depth_scale:.6f}")
        return requested_depth_scale
    return _estimate_colmap_raw_depth_scale(path, cam_extrinsics, depths_folder)


def _resolve_depth_scale_for_synthetic(requested_depth_scale, use_depth_loss):
    if not use_depth_loss:
        return 1.0
    requested_depth_scale = float(requested_depth_scale)
    if requested_depth_scale > 0.0:
        print(f"[Depth Loss] Using manual raw-depth scale: {requested_depth_scale:.6f}")
        return requested_depth_scale
    print("[Depth Loss] Using raw-depth scale 1.000000 for non-COLMAP depth maps")
    return 1.0

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, depth_scale, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        fx, fy, cx, cy = _parse_colmap_pinhole_intrinsics(intr)
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = _raw_depth_path(depths_folder, extr.name)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                              fx=fx, fy=fy, cx=cx, cy=cy, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              depth_scale=depth_scale,
                              width=width, height=height, is_test=_name_in_split(image_name, test_cam_names_list))
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, use_depth_loss=False, depth_scale=0.0, llffhold=0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depths_params = None

    train_split_names = None
    test_split_names = set()
    if eval:
        train_list_path = os.path.join(path, "train_list.txt")
        test_list_path = os.path.join(path, "test_list.txt")
        if os.path.exists(train_list_path) and os.path.exists(test_list_path):
            print("------------train_list/test_list-------------")
            train_split_names = _read_split_list(train_list_path)
            test_split_names = _read_split_list(test_list_path)
        else:
            print("------------numbered holdout mod 8-------------")
            for cam_id in cam_extrinsics:
                image_name = cam_extrinsics[cam_id].name
                if _is_numbered_test_image(image_name):
                    test_split_names.update(_split_name_keys(image_name))

    reading_dir = "images" if images == None else images
    depths_folder = _resolve_raw_depth_folder(path, depths, use_depth_loss)
    resolved_depth_scale = _resolve_depth_scale_for_colmap(path, cam_extrinsics, depths_folder, depth_scale, use_depth_loss)
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=depths_folder, depth_scale=resolved_depth_scale, test_cam_names_list=test_split_names)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval and train_split_names is not None:
        train_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, train_split_names)]
        test_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, test_split_names)]
        if not train_cam_infos:
            raise ValueError(f"No COLMAP cameras matched {os.path.join(path, 'train_list.txt')}.")
        if not test_cam_infos:
            raise ValueError(f"No COLMAP cameras matched {os.path.join(path, 'test_list.txt')}.")
    else:
        train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
        test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, depth_scale, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            fx, fy, cx, cy = _center_intrinsics_from_fov(FovX, FovY, image.size[0], image.size[1])

            depth_path = _raw_depth_path(depths_folder, image_name)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            fx=fx, fy=fy, cx=cx, cy=cy,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path,
                            depth_scale=depth_scale, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, use_depth_loss=False, depth_scale=0.0, extension=".png"):

    depths_folder = _resolve_raw_depth_folder(path, depths, use_depth_loss)
    resolved_depth_scale = _resolve_depth_scale_for_synthetic(depth_scale, use_depth_loss)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, resolved_depth_scale, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, resolved_depth_scale, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
