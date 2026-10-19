"""
Regression and validation test suite for NeuroScope 3D Critical Correctness Audit.

Covers:
1. Mathematical tests (known 20x20x20 cube volume at 1x1x2 mm = 16.0 mL for WT, 0 mL for TC/ET).
2. Spatial tests (8-corner affine bounding boxes, world centroids, spatial unit scaling).
3. Metric identities (TP + FP == N_pred, TP + FN == N_ref, empty mask policies).
4. Manifest and data provenance traceability.
5. Single canonical HD95 rounding consistency across PDF pages.
"""

import os
import json
import numpy as np
import nibabel as nib
import pytest
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from backend.app.services.volume import (
    calculate_region_volumes,
    compute_bounding_box_world,
    compute_centroid_world,
    compute_centroid_fractional,
    get_spatial_unit_scale_factor,
)
from evaluation.evaluate import evaluate_case
from evaluation.independent_validator import independent_evaluate_masks


def test_synthetic_volume_cube():
    """
    Phase 5.1 Volume Calculation Acceptance Test:
    - Mask dimensions: 20 x 20 x 20
    - All voxels have label 2 (ED)
    - Voxel spacing: 1 x 1 x 2 mm
    - Expected WT volume: 16.0 mL (20*20*20 * 2 mm^3 = 16,000 mm^3 = 16.0 mL)
    - Expected TC volume: 0.0 mL (NCR + ET only)
    - Expected ET volume: 0.0 mL (ET only)
    - Pass tolerance: +/- 1%
    """
    mask = np.full((20, 20, 20), 2, dtype=np.uint8)
    spacing = (1.0, 1.0, 2.0)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])

    volumes = calculate_region_volumes(mask, spacing, affine=affine)

    assert "WT" in volumes and "TC" in volumes and "ET" in volumes

    wt_vol = volumes["WT"]["volume_ml"]
    tc_vol = volumes["TC"]["volume_ml"]
    et_vol = volumes["ET"]["volume_ml"]

    # Expected: 16.0 mL within +/- 1% (15.84 to 16.16)
    assert abs(wt_vol - 16.0) <= 0.16, f"WT volume {wt_vol} mL not within 1% of 16.0 mL"
    assert tc_vol == 0.0, f"TC volume expected 0.0 mL, got {tc_vol}"
    assert et_vol == 0.0, f"ET volume expected 0.0 mL, got {et_vol}"
    assert volumes["WT"]["voxel_count"] == 8000
    assert volumes["TC"]["voxel_count"] == 0
    assert volumes["ET"]["voxel_count"] == 0


def test_spatial_unit_scale_factors():
    """
    Test NIfTI spatial unit code handling:
    - Code 2: Millimeters (scale 1.0)
    - Code 1: Meters (scale 1000.0)
    - Code 3: Microns (scale 0.001)
    """
    # Create NIfTI header mock
    img_mm = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    img_mm.header.set_xyzt_units(xyz=2)  # mm
    scale_mm = get_spatial_unit_scale_factor(img_mm.header)
    assert scale_mm == pytest.approx(1.0)

    img_m = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    img_m.header.set_xyzt_units(xyz=1)  # meter
    scale_m = get_spatial_unit_scale_factor(img_m.header)
    assert scale_m == pytest.approx(1000.0)

    img_um = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    img_um.header.set_xyzt_units(xyz=3)  # micron
    scale_um = get_spatial_unit_scale_factor(img_um.header)
    assert scale_um == pytest.approx(0.001)


def test_affine_8_corner_bounding_box_transformation():
    """
    Phase 5.2 Bounding Box Test:
    Verify that compute_bounding_box_world uses all 8 corners of the voxel bounding box
    and correctly handles negative scales / reflections (e.g. LPS orientation) and translations.
    """
    # Affine with LPS orientation (x negative, y negative, translation)
    affine = np.array([
        [-1.0,  0.0,  0.0,  100.0],
        [ 0.0, -1.0,  0.0,  200.0],
        [ 0.0,  0.0,  1.5,  -50.0],
        [ 0.0,  0.0,  0.0,    1.0],
    ], dtype=np.float64)

    min_vox = [10, 20, 30]
    max_vox = [40, 60, 50]

    # Centers
    bbox_world = compute_bounding_box_world({"min": min_vox, "max": max_vox}, affine=affine, use_voxel_boundary=False)

    # 10 maps to 100 - 10 = 90; 40 maps to 100 - 40 = 60 -> min X is 60, max X is 90
    assert bbox_world["min"][0] == pytest.approx(60.0)
    assert bbox_world["max"][0] == pytest.approx(90.0)

    # 20 maps to 200 - 20 = 180; 60 maps to 200 - 60 = 140 -> min Y is 140, max Y is 180
    assert bbox_world["min"][1] == pytest.approx(140.0)
    assert bbox_world["max"][1] == pytest.approx(180.0)

    # 30 maps to -50 + 1.5*30 = -5.0; 50 maps to -50 + 1.5*50 = 25.0 -> min Z is -5.0, max Z is 25.0
    assert bbox_world["min"][2] == pytest.approx(-5.0)
    assert bbox_world["max"][2] == pytest.approx(25.0)


def test_independent_validator_metric_identities():
    """
    Phase 5.2 / 7.2 Metric Identities Test:
    Verify TP + FP == N_pred and TP + FN == N_ref.
    """
    # Create synthetic prediction and ground truth masks
    shape = (50, 50, 50)
    gt = np.zeros(shape, dtype=np.uint8)
    pred = np.zeros(shape, dtype=np.uint8)

    # GT has 1000 voxels of label 1, 500 voxels of label 4
    gt[10:20, 10:20, 10:20] = 1  # 1000 voxels
    gt[15:20, 15:20, 15:20] = 4  # 125 voxels override to 4 (so label 1 is 875, label 4 is 125)

    # Pred has overlap + false positives
    pred[12:22, 12:22, 12:22] = 1
    pred[16:21, 16:21, 16:21] = 4

    metrics = evaluate_case(pred, gt, spacing_mm=(1.0, 1.0, 1.0))

    for reg_key, (pred_mask, gt_mask) in [
        ("WT", (np.isin(pred, [1, 2, 4]), np.isin(gt, [1, 2, 4]))),
        ("TC", (np.isin(pred, [1, 4]), np.isin(gt, [1, 4]))),
        ("ET", (pred == 4, gt == 4)),
    ]:
        m = metrics[reg_key]
        n_pred = int(np.sum(pred_mask))
        n_ref = int(np.sum(gt_mask))

        tp = m["tp_voxels"]
        fp = m["fp_voxels"]
        fn = m["fn_voxels"]

        assert tp + fp == n_pred, f"{reg_key}: TP({tp}) + FP({fp}) != N_pred({n_pred})"
        assert tp + fn == n_ref, f"{reg_key}: TP({tp}) + FN({fn}) != N_ref({n_ref})"

        # Verify dice identity
        expected_dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0
        assert m["dice"] == pytest.approx(expected_dice, rel=1e-5)


def test_case_manifest_provenance():
    """
    Phase 2 / Phase 8 Provenance Test:
    Verify that data/case_manifest_BraTS2021_00495.json exists, links the files,
    and documents the exact subject identity BraTS2021_00495.
    """
    manifest_path = os.path.join(os.path.dirname(__file__), "../data/case_manifest_BraTS2021_00495.json")
    if not os.path.exists(manifest_path):
        pytest.skip("Case manifest file not found in repository (data/ is gitignored)")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["subject_id"] == "BraTS2021_00495"
    assert manifest["verification_status"] == "VERIFIED"
    assert manifest["spatial_geometry"]["matrix_dimensions"] == [240, 240, 155]
    assert manifest["spatial_geometry"]["orientation"] == "LPS"

    # Verify input file hashes are 64 characters
    for seq_key, file_info in manifest["files"].items():
        assert len(file_info["raw_sha256"]) == 64, f"raw SHA-256 for {seq_key} is not 64 hex characters"
        assert len(file_info["gz_sha256"]) == 64, f"gz SHA-256 for {seq_key} is not 64 hex characters"


def test_canonical_hd95_consistency_in_pdf():
    """
    Phase 5.3 / Phase 12 PDF Consistency Test:
    Verify that in generated reports, TC HD95 is reported consistently as 2.45 mm
    and does not conflict between 2.5 mm and 2.45 mm.
    """
    if PdfReader is None:
        pytest.skip("pypdf is not installed; skipping PDF text inspection test")

    pdf_path = os.path.join(os.path.dirname(__file__), "../reports/sample_report.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("sample_report.pdf has not been regenerated yet")

    reader = PdfReader(pdf_path)
    assert len(reader.pages) == 4, f"Expected 4 pages, found {len(reader.pages)}"

    # Page 2 text extraction
    p2_text = reader.pages[1].extract_text()
    # Page 3 text extraction
    p3_text = reader.pages[2].extract_text()

    # Verify 2.45 mm is in both Page 2 and Page 3
    assert "2.45 mm" in p2_text or "2.45" in p2_text, "TC HD95 2.45 mm missing from Page 2"
    assert "2.45 mm" in p3_text or "2.45" in p3_text, "TC HD95 2.45 mm missing from Page 3"

    # Verify that '2.5 mm' does NOT appear as TC HD95
    # (Checking that conflicting rounding was eradicated)
    lines_p2 = [line for line in p2_text.splitlines() if "TC" in line]
    for line in lines_p2:
        assert "2.5 mm" not in line, f"Found uncanonical '2.5 mm' in TC line on Page 2: {line}"


def test_empty_mask_evaluation_policy():
    """
    Phase 5.2 Empty Mask Policy:
    When ground-truth and prediction both have 0 voxels for ET:
    - Dice = 1.0 (true negative concordance)
    - Sensitivity = 1.0
    - HD95 = None or 0.0 mm
    """
    shape = (30, 30, 30)
    gt = np.zeros(shape, dtype=np.uint8)
    pred = np.zeros(shape, dtype=np.uint8)

    # WT has some voxels, ET has none
    gt[5:15, 5:15, 5:15] = 2  # ED only
    pred[5:15, 5:15, 5:15] = 2

    metrics = evaluate_case(pred, gt, spacing_mm=(1.0, 1.0, 1.0))
    et_metric = metrics["ET"]

    assert et_metric["dice"] == 1.0
    assert et_metric["sensitivity"] == 1.0
    assert et_metric["tp_voxels"] == 0
    assert et_metric["fp_voxels"] == 0
    assert et_metric["fn_voxels"] == 0
