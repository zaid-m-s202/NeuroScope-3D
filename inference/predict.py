"""
NeuroScope 3D - Standalone Inference Pipeline CLI
Executes preprocessing, model forward pass, nesting enforcement, BraTS 2021 label conversion,
grid restoration, and NIfTI output writing.
Independent of web application.
"""
import argparse
import os
import sys
import time
import numpy as np
import nibabel as nib
from typing import Dict, Any, Tuple, Optional

from inference.preprocessing import preprocess_study, PreprocessingError
from inference.postprocessing import (
    enforce_nesting_and_convert_labels,
    restore_original_grid,
    validate_label_map,
    derive_region_masks,
    PostprocessingError,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="NeuroScope 3D Standalone Inference Pipeline. Research/teaching only."
    )
    parser.add_argument("--t1", required=True, help="Path to T1-weighted NIfTI file")
    parser.add_argument("--t1gd", required=True, help="Path to T1Gd (contrast) NIfTI file")
    parser.add_argument("--t2", required=True, help="Path to T2-weighted NIfTI file")
    parser.add_argument("--flair", required=True, help="Path to FLAIR NIfTI file")
    parser.add_argument("--output", required=True, help="Output path for predicted BraTS 2021 label map (.nii or .nii.gz)")
    parser.add_argument("--weights", default=os.getenv("MODEL_WEIGHTS_PATH", "models/brats_mri_segmentation/models/model.pt"),
                        help="Path to pretrained model weights (checkpoint .pt)")
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cpu"), choices=["cuda", "cpu"],
                        help="Inference device (cuda or cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for subregions (default: 0.5)")
    return parser.parse_args(args)


# Global model cache to ensure checkpoint is loaded once per process lifetime
_MODEL_CACHE = {}


def clear_model_cache():
    """Clears cached model instances from memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def get_or_load_model(weights_path: str, device_name: str = "cpu"):
    """
    Retrieves a cached SegResNet model or loads checkpoint once from disk.
    Configured for evaluation mode with no gradients.
    """
    try:
        import torch
        from monai.networks.nets import SegResNet
    except ImportError as e:
        raise RuntimeError(
            f"Missing deep learning dependency for model execution: {e}. "
            "Please ensure torch and monai are installed."
        )

    resolved_weights = os.path.abspath(weights_path)
    if not os.path.exists(resolved_weights):
        alt = resolved_weights.replace(".pt", ".ts") if resolved_weights.endswith(".pt") else resolved_weights.replace(".ts", ".pt")
        if os.path.exists(alt):
            resolved_weights = alt
        else:
            raise FileNotFoundError(
                f"Pretrained model weights not found at '{resolved_weights}'. "
                "Download or specify the MONAI brats_mri_segmentation bundle weights via --weights."
            )

    actual_device_name = "cuda" if (torch.cuda.is_available() and device_name == "cuda") else "cpu"
    cache_key = (resolved_weights, actual_device_name)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], torch.device(actual_device_name)

    device = torch.device(actual_device_name)

    # If weights are TorchScript bundle (.ts), load via torch.jit.load
    if resolved_weights.endswith(".ts"):
        try:
            model = torch.jit.load(resolved_weights, map_location=device)
            model.eval()
            _MODEL_CACHE[cache_key] = model
            return model, device
        except Exception:
            pass

    # Instantiate SegResNet adhering strictly to bundle config
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)

    # Load weights
    checkpoint = torch.load(resolved_weights, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model, device


def get_adaptive_roi_size(spatial_shape, max_roi=(240, 240, 160), factor=16):
    """
    Computes an optimal sliding window ROI size appropriate for the input volume size,
    constrained by max_roi and aligned to factor (16) for SegResNet downsampling.
    """
    roi = []
    for d, max_d in zip(spatial_shape, max_roi):
        if d <= max_d:
            rem = d % factor
            aligned = d if rem == 0 else d + (factor - rem)
            roi.append(min(aligned, max_d))
        else:
            roi.append(max_d)
    return roi


def run_model_inference(
    preprocessed_tensor: np.ndarray,
    weights_path: str,
    device_name: str = "cpu"
) -> np.ndarray:
    """
    Executes SegResNet model forward pass via PyTorch / MONAI.
    Expects input shape (4, H, W, D).
    Returns raw 3-channel logits or probabilities with shape (3, H, W, D).
    Loads checkpoint once from cache, uses eval mode, adapts ROI size to image, and safely handles OOM.
    """
    try:
        import torch
        from monai.inferers import SlidingWindowInferer
    except ImportError as e:
        raise RuntimeError(
            f"Missing deep learning dependency for model execution: {e}. "
            "Please ensure torch and monai are installed."
        )

    model, device = get_or_load_model(weights_path=weights_path, device_name=device_name)

    # Configure sliding window inferer appropriate for input volume size
    spatial_shape = preprocessed_tensor.shape[1:]
    roi_size = get_adaptive_roi_size(spatial_shape, max_roi=(240, 240, 160), factor=16)

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
    )

    try:
        input_tensor = torch.from_numpy(preprocessed_tensor).unsqueeze(0).to(device)  # shape (1, 4, H, W, D)

        with torch.no_grad():
            raw_output = inferer(input_tensor, model)  # shape (1, 3, H, W, D)

        output_np = raw_output.squeeze(0).cpu().numpy().astype(np.float32)  # shape (3, H, W, D)
        return output_np
    except Exception as err:
        if "cuda" in str(device) and ("out of memory" in str(err).lower() or type(err).__name__ == "OutOfMemoryError"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"GPU Out of Memory (CUDA OOM) during model inference: {err}. "
                "Ensure sufficient VRAM is available or switch inference device to CPU (--device cpu)."
            )
        raise RuntimeError(f"Error during SegResNet forward pass: {err}")


def run_pipeline(
    t1: str = None,
    t1gd: str = None,
    t2: str = None,
    flair: str = None,
    output: str = None,
    weights: str = "models/brats_mri_segmentation/models/model.pt",
    device: str = "cpu",
    threshold: float = 0.5,
    **kwargs
) -> str:
    """
    Main orchestrator for standalone inference.
    Executes full pipeline and writes BraTS 2021 NIfTI file on original grid.
    Supports both standard and _path keyword arguments.
    """
    t1 = t1 or kwargs.get("t1_path")
    t1gd = t1gd or kwargs.get("t1gd_path") or kwargs.get("t1ce_path")
    t2 = t2 or kwargs.get("t2_path")
    flair = flair or kwargs.get("flair_path")
    output = output or kwargs.get("output_path")
    weights = weights or kwargs.get("weights_path", "models/brats_mri_segmentation/models/model.pt")
    start_time = time.time()
    sequence_paths = {"t1": t1, "t1gd": t1gd, "t2": t2, "flair": flair}

    print("[NeuroScope 3D] Starting standalone inference pipeline...")
    print(f"[NeuroScope 3D] Device requested: {device}")

    # 1. Preprocessing
    print("[NeuroScope 3D] Loading and validating sequences [T1Gd, T1, T2, FLAIR]...")
    tensor, meta = preprocess_study(sequence_paths)
    print(f"[NeuroScope 3D] Preprocessed tensor shape: {tensor.shape}")

    # 2. Forward Pass
    print(f"[NeuroScope 3D] Running SegResNet inference (weights: {weights})...")
    raw_channels = run_model_inference(tensor, weights_path=weights, device_name=device)
    print(f"[NeuroScope 3D] Raw model output shape: {raw_channels.shape} (Channels: 0=TC, 1=WT, 2=ET)")

    # 3. Postprocessing: Nesting & BraTS Label Conversion
    print(f"[NeuroScope 3D] Applying threshold {threshold}, enforcing ET <= TC <= WT, and mapping BraTS labels...")
    label_map = enforce_nesting_and_convert_labels(raw_channels, threshold=threshold, is_logits=True)

    # 4. Restore Original Grid
    original_shape = tuple(meta["original_shape"])
    print(f"[NeuroScope 3D] Restoring original spatial grid {original_shape} with nearest-neighbour interpolation...")
    restored_labels = restore_original_grid(label_map, target_shape=original_shape)

    # 5. Validation of Output
    validate_label_map(restored_labels, expected_shape=original_shape)
    unique_labels = sorted(np.unique(restored_labels).tolist())
    print(f"[NeuroScope 3D] Output label validation passed. Unique labels present: {unique_labels}")

    # 6. Region Breakdown Summary
    region_masks = derive_region_masks(restored_labels)
    voxel_spacing = meta["voxel_spacing_mm"]
    voxel_vol_ml = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0

    print("[NeuroScope 3D] Derived region summary:")
    for region_name, mask in region_masks.items():
        vcount = int(np.sum(mask))
        vml = vcount * voxel_vol_ml
        print(f"  - {region_name}: {vcount} voxels ({vml:.2f} mL)")

    # 7. Write NIfTI file to disk
    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ref_affine = meta["ref_affine_np"]
    ref_header = meta["ref_header"]
    out_nii = nib.Nifti1Image(restored_labels, affine=ref_affine, header=ref_header)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, output)

    # 8. Compute and Save Voxel-wise Probability Maps & Boundary Contours
    save_probability_and_boundary_maps(
        raw_channels=raw_channels,
        label_map=restored_labels,
        meta=meta,
        output_dir=output_dir,
    )

    elapsed = time.time() - start_time
    print(f"[NeuroScope 3D] Output successfully saved to: {output}")
    print(f"[NeuroScope 3D] Pipeline finished in {elapsed:.2f} seconds.")
    return output


def generate_boundary_map(label_map: np.ndarray) -> np.ndarray:
    """
    Computes boundary contours for BraTS regions from a discrete label map:
    - Label 2 (ED / WT contour) on Whole Tumour boundary
    - Label 1 (NCR / TC contour) on Tumour Core boundary
    - Label 4 (ET contour) on Enhancing Tumour boundary
    Non-boundary and background voxels remain 0 (transparent).
    """
    import scipy.ndimage
    wt_mask = (label_map > 0)
    tc_mask = (label_map == 1) | (label_map == 4)
    et_mask = (label_map == 4)

    wt_bound = wt_mask ^ scipy.ndimage.binary_erosion(wt_mask)
    tc_bound = tc_mask ^ scipy.ndimage.binary_erosion(tc_mask)
    et_bound = et_mask ^ scipy.ndimage.binary_erosion(et_mask)

    boundary = np.zeros_like(label_map, dtype=np.uint8)
    boundary[wt_bound] = 2
    boundary[tc_bound] = 1
    boundary[et_bound] = 4
    return boundary


def save_probability_and_boundary_maps(
    raw_channels: np.ndarray,
    label_map: np.ndarray,
    meta: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """
    Computes genuine voxel-wise sigmoid probability maps from MONAI SegResNet
    raw channels:
      Channel 0: TC (Tumor Core)
      Channel 1: WT (Whole Tumor)
      Channel 2: ET (Enhancing Tumor)
    Also computes composite maximum probability and boundary contours.
    Saves NIfTI files (.nii.gz) in output_dir:
      - prob_wt.nii.gz: Whole Tumour sigmoid probability map
      - prob_tc.nii.gz: Tumour Core sigmoid probability map
      - prob_et.nii.gz: Enhancing Tumour sigmoid probability map
      - prob_max.nii.gz: Composite maximum probability map
      - boundary_seg.nii.gz: Crisp boundary contours
    """
    if not output_dir:
        return {}
    os.makedirs(output_dir, exist_ok=True)

    ref_affine = meta.get("ref_affine_np")
    if ref_affine is None:
        ref_affine = np.eye(4)
    ref_header = meta.get("ref_header")

    # Sigmoid activation on raw model output channels
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_channels, -30.0, 30.0)))
    original_shape = tuple(meta.get("original_shape", raw_channels.shape[1:]))

    # Resample continuous probabilities to original reference grid if needed
    if probs.shape[1:] != original_shape:
        from scipy.ndimage import zoom
        zoom_factors = [original_shape[i] / float(probs.shape[1 + i]) for i in range(3)]
        prob_tc = zoom(probs[0], zoom=zoom_factors, order=1, mode="nearest", prefilter=False).astype(np.float32)
        prob_wt = zoom(probs[1], zoom=zoom_factors, order=1, mode="nearest", prefilter=False).astype(np.float32)
        prob_et = zoom(probs[2], zoom=zoom_factors, order=1, mode="nearest", prefilter=False).astype(np.float32)
    else:
        prob_tc = probs[0].astype(np.float32)
        prob_wt = probs[1].astype(np.float32)
        prob_et = probs[2].astype(np.float32)

    prob_max = np.maximum.reduce([prob_wt, prob_tc, prob_et]).astype(np.float32)

    prob_maps = {
        "wt": prob_wt,
        "tc": prob_tc,
        "et": prob_et,
        "max": prob_max,
    }

    saved = {}
    for ch_name, p_arr in prob_maps.items():
        out_path = os.path.join(output_dir, f"prob_{ch_name}.nii.gz")
        p_nii = nib.Nifti1Image(p_arr, affine=ref_affine, header=ref_header)
        p_nii.set_data_dtype(np.float32)
        nib.save(p_nii, out_path)
        saved[ch_name] = out_path

    # Save boundary mask
    try:
        boundary_arr = generate_boundary_map(label_map)
        b_path = os.path.join(output_dir, "boundary_seg.nii.gz")
        b_nii = nib.Nifti1Image(boundary_arr, affine=ref_affine, header=ref_header)
        b_nii.set_data_dtype(np.uint8)
        nib.save(b_nii, b_path)
        saved["boundary"] = b_path
    except Exception as e:
        print(f"[NeuroScope 3D Warning] Failed to save boundary mask: {e}")

    return saved


def main():
    args = parse_args()
    try:
        run_pipeline(
            t1=args.t1,
            t1gd=args.t1gd,
            t2=args.t2,
            flair=args.flair,
            output=args.output,
            weights=args.weights,
            device=args.device,
            threshold=args.threshold,
        )
    except (PreprocessingError, PostprocessingError, FileNotFoundError, RuntimeError) as e:
        print(f"\n[NeuroScope 3D ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[NeuroScope 3D UNEXPECTED ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
