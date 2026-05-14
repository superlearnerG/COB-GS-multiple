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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def _camera_tanfov(viewpoint_camera):
    fx = getattr(viewpoint_camera, "fx", None)
    fy = getattr(viewpoint_camera, "fy", None)
    if fx is not None and fy is not None:
        return (
            float(viewpoint_camera.image_width) / (2.0 * float(fx)),
            float(viewpoint_camera.image_height) / (2.0 * float(fy)),
        )
    return (
        math.tan(viewpoint_camera.FoVx * 0.5),
        math.tan(viewpoint_camera.FoVy * 0.5),
    )

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, gaussian_filter=None, mask_override=None, return_alpha=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    means3D = pc.get_xyz if gaussian_filter is None else pc.get_xyz[gaussian_filter]
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx, tanfovy = _camera_tanfov(viewpoint_camera)

    include_mask = bool(getattr(opt, "include_mask", False)) or bool(return_alpha)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        include_mask=include_mask
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity if gaussian_filter is None else pc.get_opacity[gaussian_filter]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        if gaussian_filter is not None:
            cov3D_precomp = cov3D_precomp[gaussian_filter]
    else:
        scales = pc.get_scaling if gaussian_filter is None else pc.get_scaling[gaussian_filter]
        rotations = pc.get_rotation if gaussian_filter is None else pc.get_rotation[gaussian_filter]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            if gaussian_filter is not None:
                shs_view = shs_view[gaussian_filter]
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc = pc.get_features_dc if gaussian_filter is None else pc.get_features_dc[gaussian_filter]
                shs = pc.get_features_rest if gaussian_filter is None else pc.get_features_rest[gaussian_filter]
            else:
                shs = pc.get_features if gaussian_filter is None else pc.get_features[gaussian_filter]
    else:
        colors_precomp = override_color

    if include_mask:
        if return_alpha:
            mask_precomp = torch.ones((means3D.shape[0],), dtype=opacity.dtype, device=opacity.device)
        elif mask_override is None:
            mask_precomp = pc.get_mask if gaussian_filter is None else pc.get_mask[gaussian_filter]
        else:
            mask_precomp = mask_override
            if gaussian_filter is not None and mask_override.shape[0] == pc.get_xyz.shape[0]:
                mask_precomp = mask_override[gaussian_filter]
        mask_signals = torch.zeros((means3D.shape[0], 2), requires_grad=True, device="cuda") + 0
        try:
            mask_signals.retain_grad()
        except:
            pass
    else:
        mask_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)

        mask_signals = torch.zeros((1,), device=opacity.device)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, mask_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            mask_precomp = mask_precomp,
            mask_signals = mask_signals,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, mask_image,radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            mask_precomp = mask_precomp,
            mask_signals = mask_signals,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)

    if include_mask:
        out = {
            "render": rendered_image,
            "mask": mask_image,
            "mask_signals": mask_signals,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth_image
        }
        if return_alpha:
            out["alpha"] = mask_image
    else:
        out = {
            "render": rendered_image,
            "mask": mask_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "depth" : depth_image
            }

    return out
