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

import numpy as np
from PIL import Image
from utils.projection_utils import camera_intrinsics

WARNED = False


def _load_raw_depth_as_invdepth(depth_path, depth_scale=1.0):
    depth_scale = float(depth_scale)
    if depth_scale <= 0.0:
        raise ValueError(f"Depth scale must be positive when loading raw depth, got {depth_scale}.")

    raw_depth = np.load(depth_path)
    raw_depth = np.asarray(raw_depth)
    if raw_depth.ndim == 3:
        if raw_depth.shape[-1] == 1:
            raw_depth = raw_depth[..., 0]
        elif raw_depth.shape[0] == 1:
            raw_depth = raw_depth[0]
        else:
            raise ValueError(f"Expected a single-channel raw depth map at '{depth_path}', got shape {raw_depth.shape}.")
    if raw_depth.ndim != 2:
        raise ValueError(f"Expected a 2D raw depth map at '{depth_path}', got shape {raw_depth.shape}.")

    raw_depth = raw_depth.astype(np.float32, copy=False)
    valid_mask = np.isfinite(raw_depth) & (raw_depth > 0.0)
    if not np.any(valid_mask):
        raise ValueError(f"Raw depth map '{depth_path}' has no finite positive depth values.")

    invdepthmap = np.zeros_like(raw_depth, dtype=np.float32)
    scaled_depth = raw_depth[valid_mask] * depth_scale
    valid_scaled = np.isfinite(scaled_depth) & (scaled_depth > 0.0)
    if not np.any(valid_scaled):
        raise ValueError(f"Scaled raw depth map '{depth_path}' has no finite positive depth values.")
    valid_indices = np.nonzero(valid_mask)
    invdepthmap[valid_indices[0][valid_scaled], valid_indices[1][valid_scaled]] = 1.0 / scaled_depth[valid_scaled]
    return invdepthmap

def _scaled_camera_intrinsics(cam_info, resolution):
    fx = getattr(cam_info, "fx", None)
    fy = getattr(cam_info, "fy", None)
    cx = getattr(cam_info, "cx", None)
    cy = getattr(cam_info, "cy", None)
    if fx is None or fy is None or cx is None or cy is None:
        return None, None, None, None

    orig_w = float(cam_info.width)
    orig_h = float(cam_info.height)
    scale_x = float(resolution[0]) / orig_w
    scale_y = float(resolution[1]) / orig_h
    return (
        float(fx) * scale_x,
        float(fy) * scale_y,
        (float(cx) + 0.5) * scale_x - 0.5,
        (float(cy) + 0.5) * scale_y - 0.5,
    )

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    from scene.cameras import Camera

    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            invdepthmap = _load_raw_depth_as_invdepth(cam_info.depth_path, getattr(cam_info, "depth_scale", 1.0))
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    fx, fy, cx, cy = _scaled_camera_intrinsics(cam_info, resolution)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  fx=fx, fy=fy, cx=cx, cy=cy,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera: "Camera"):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    fx, fy, cx, cy = camera_intrinsics(camera)
    width = getattr(camera, "image_width", getattr(camera, "width", None))
    height = getattr(camera, "image_height", getattr(camera, "height", None))
    if width is None or height is None:
        raise AttributeError("Camera JSON export requires image_width/image_height or width/height.")
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : int(width),
        'height' : int(height),
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fx' : float(fx),
        'fy' : float(fy),
        'cx' : float(cx),
        'cy' : float(cy),
    }
    return camera_entry
