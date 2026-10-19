"""
NeuroScope 3D - 3D Polygonal Surface Mesh Extraction Service
============================================================
Converts voxel masks and brain parenchyma contours into smooth 3D polygonal surface meshes
calibrated in physical continuous world millimetre (mm) space using Marching Cubes.
Exports GLTF binary (.glb) and Wavefront (.obj) models for interactive 3D WebGL viewing.
"""
from typing import Tuple, Dict, Any, Optional
import io
import numpy as np
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
import scipy.ndimage
from skimage.measure import marching_cubes
import trimesh
import nibabel as nib


def get_region_binary_mask(label_data: np.ndarray, region: str) -> np.ndarray:
    """Extracts binary mask for target anatomical / tumor region."""
    reg = region.upper()
    if reg == "WT":
        return np.isin(label_data, [1, 2, 4])
    elif reg == "TC":
        return np.isin(label_data, [1, 4])
    elif reg == "ET":
        return (label_data == 4)
    elif reg == "BRAIN_SURFACE":
        # Non-zero brain parenchyma threshold
        return (label_data > 0)
    else:
        raise ValueError(f"Unknown region '{region}'. Must be 'WT', 'TC', 'ET', or 'BRAIN_SURFACE'.")


def generate_region_mesh(
    volume_data: np.ndarray,
    affine: np.ndarray,
    region: str = "WT",
    smoothing_sigma: float = 0.75,
    max_faces: int = 35000
) -> trimesh.Trimesh:
    """
    Extracts a smooth 3D polygonal surface mesh in continuous physical millimetre space.
    Transforms voxel coordinates using the 4x4 spatial affine matrix.
    """
    binary_vol = get_region_binary_mask(volume_data, region)
    active_count = np.count_nonzero(binary_vol)
    if active_count < 8:
        # Return empty minimal mesh if region is empty
        return trimesh.Trimesh()

    # Apply 3D Gaussian smoothing to suppress stepped voxel artifacts
    smoothed = scipy.ndimage.gaussian_filter(binary_vol.astype(np.float32), sigma=smoothing_sigma)

    # Run Marching Cubes
    verts, faces, normals, _ = marching_cubes(smoothed, level=0.5)

    # Transform vertices from voxel index space to physical world coordinates (mm)
    # verts is (N, 3). Add homogeneous coordinate 1: (N, 4)
    homo_verts = np.c_[verts, np.ones(len(verts), dtype=np.float32)]
    world_verts = (affine @ homo_verts.T).T[:, :3]

    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces, vertex_normals=normals)

    # Simplify mesh if face count exceeds target budget for smooth 60 FPS WebGL
    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadratic_decimation(max_faces)
        except Exception:
            pass  # Fall back to unsimplified mesh if decimation fails

    # Assign clinical color theme metadata: WT=Violet, TC=Coral, ET=Amber
    color_map = {
        "WT": [139, 92, 246, 180],       # Violet
        "TC": [249, 115, 22, 220],        # Coral
        "ET": [245, 158, 11, 255],        # Amber
        "BRAIN_SURFACE": [148, 163, 184, 40], # Glass Slate/Blue
    }
    mesh.visual.vertex_colors = color_map.get(region.upper(), [200, 200, 200, 200])

    return mesh


def export_mesh_bytes(mesh: trimesh.Trimesh, export_format: str = "glb") -> bytes:
    """Exports a Trimesh model to binary GLTF (.glb) or Wavefront (.obj) bytes."""
    fmt = export_format.lower()
    if fmt == "glb" or fmt == "gltf":
        return mesh.export(file_type="glb")
    elif fmt == "obj":
        return mesh.export(file_type="obj").encode("utf-8")
    else:
        raise ValueError(f"Unsupported mesh format '{export_format}'. Supported: 'glb', 'obj'.")
