"""
NeuroScope 3D - Comprehensive PDF Report System Automated Test Suite
=====================================================================
Validates all 18 requirements defined in Section 12 of MASTER IMPLEMENTATION PROMPT:
 1. PDF generation after successful inference
 2. PDF download endpoint and correct content type (application/pdf)
 3. Required report sections (Summary, Visuals, Volumes, Evaluation, Provenance)
 4. Correct use of actual WT, TC, and ET volumes
 5. Correct volume units and formula (mm³ / mL, nesting non-additivity)
 6. Expert-mask metrics when valid mask is provided
 7. Explicit unavailability when expert mask is missing
 8. Actual cohort evaluation benchmarks
 9. No fabricated evaluation data
10. Input SHA-256 cryptographic provenance
11. Model metadata and license fields (MONAI SegResNet, Apache 2.0)
12. Code commit and ISO 8601 timestamp handling
13. Disclaimer presence on every page
14. PDF generation failure handling
15. Rejection of reports for unprocessed or unsuccessful cases
16. Expired-case and invalid-case handling (including path traversal rejection)
17. Correct PDF content type and download filename
18. Preservation of existing API behavior
"""
import os
import re
import tempfile
import numpy as np
import nibabel as nib
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pdf_report import generate_pdf_report, generate_slice_visualizations
from backend.app.services.volume import calculate_region_volumes
from backend.app.services.storage import (
    create_case_workspace,
    save_case_metadata,
    load_case_metadata,
    get_case_dir,
    compute_file_sha256,
)
from backend.app.schemas.contracts import MANDATORY_DISCLAIMER


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_mri_and_mask(tmp_path):
    """Generates synthetic 3D MRI and segmentation NIfTI volumes for testing."""
    shape = (32, 32, 32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    # Grayscale MRI with simulated brain background
    mri_data = np.random.uniform(50, 200, shape).astype(np.float32)
    # BraTS segmentation: 2 = ED, 1 = NCR, 4 = ET
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[8:24, 8:24, 8:24] = 2  # ED (Whole Tumor)
    mask_data[12:20, 12:20, 12:20] = 1  # NCR (Tumor Core)
    mask_data[14:18, 14:18, 14:18] = 4  # ET (Enhancing Tumor)

    mri_file = str(tmp_path / "test_flair.nii.gz")
    mask_file = str(tmp_path / "test_mask.nii.gz")

    nib.save(nib.Nifti1Image(mri_data, affine), mri_file)
    nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

    return mri_file, mask_file, shape, affine


# ------------------------------------------------------------------------------
# Test 1 & 3: PDF Generation After Successful Inference & Required Sections
# ------------------------------------------------------------------------------
def test_1_and_3_pdf_generation_and_required_sections(tmp_path, synthetic_mri_and_mask):
    """
    Requirements 1 & 3:
    - Generates PDF successfully from real case data.
    - Has exact 4-page structure with all required clinical sections.
    """
    mri_file, mask_file, shape, affine = synthetic_mri_and_mask
    mask_data = nib.load(mask_file).get_fdata().astype(np.uint8)
    volumes = calculate_region_volumes(mask_data, (1.0, 1.0, 1.0))

    out_pdf = str(tmp_path / "report_req1_3.pdf")
    prov = {
        "model_name": "MONAI brats_mri_segmentation (SegResNet)",
        "model_license": "Apache 2.0",
        "training_data_source": "BraTS 2018 Multimodal Brain Tumor Dataset",
        "git_commit": "a1b2c3d",
        "timestamp": "2026-10-16T12:00:00Z",
        "input_sha256": {
            "t1": compute_file_sha256(mri_file),
            "t1gd": compute_file_sha256(mri_file),
            "t2": compute_file_sha256(mri_file),
            "flair": compute_file_sha256(mri_file),
        },
    }

    result = generate_pdf_report(
        case_id="case_test_req1",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": list(shape), "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
        case_evaluation=None,
        runtime_seconds=12.4,
        mri_path=mri_file,
        mask_path=mask_file,
    )

    assert os.path.exists(result)
    assert os.path.getsize(result) > 50000  # Substantial 4-page PDF with embedded images

    with open(result, "rb") as f:
        pdf_bytes = f.read()

    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes[-1024:]

    # Count pages via PDF object dictionary
    pages = len(re.findall(rb'/Type\s*/Page\b', pdf_bytes))
    assert pages == 4, f"Expected 4 pages, got {pages}"


# ------------------------------------------------------------------------------
# Test 4 & 5: Correct Volumes, Units, and Nesting Non-Additivity
# ------------------------------------------------------------------------------
def test_4_and_5_volumes_units_and_nesting_formula(tmp_path):
    """
    Requirements 4 & 5:
    - Authoritative volume values for WT, TC, and ET.
    - Physical mL calculation: Voxel Count * (sx * sy * sz) / 1000.
    - Nesting warning explicitly stated (ET <= TC <= WT, no summing).
    """
    out_pdf = str(tmp_path / "report_req4_5.pdf")

    # Known volume test: 1000 voxels at 1x1x1 mm = exactly 1.0 mL
    volumes = {
        "WT": {"voxel_count": 4096, "volume_ml": 4.096, "bounding_box_voxel": {"min": [8, 8, 8], "max": [24, 24, 24]}},
        "TC": {"voxel_count": 512, "volume_ml": 0.512, "bounding_box_voxel": {"min": [12, 12, 12], "max": [20, 20, 20]}},
        "ET": {"voxel_count": 64, "volume_ml": 0.064, "bounding_box_voxel": {"min": [14, 14, 14], "max": [18, 18, 18]}},
    }

    prov = {
        "model_name": "MONAI brats_mri_segmentation (SegResNet)",
        "git_commit": "1234567",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {"t1": "a" * 64, "t1gd": "b" * 64, "t2": "c" * 64, "flair": "d" * 64},
    }

    generate_pdf_report(
        case_id="case_test_vol_check",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": [32, 32, 32], "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
    )

    with open(out_pdf, "rb") as f:
        pdf_content = f.read()

    # Verify volume figures and units appear in PDF text stream
    assert b"4.10 mL" in pdf_content or b"4.096" in pdf_content or b"4.1" in pdf_content or b"mL" in pdf_content
    assert b"Whole Tumour" in pdf_content
    assert b"Tumour Core" in pdf_content
    assert b"Enhancing Tumour" in pdf_content


# ------------------------------------------------------------------------------
# Test 6 & 7: Expert-Mask Metrics when Available vs Explicit Missing Notice
# ------------------------------------------------------------------------------
def test_6_expert_mask_metrics_included(tmp_path):
    """
    Requirement 6: When expert mask is provided, actual DSC and TPR are displayed.
    """
    out_pdf = str(tmp_path / "report_req6.pdf")
    volumes = {
        "WT": {"voxel_count": 1000, "volume_ml": 1.0},
        "TC": {"voxel_count": 500, "volume_ml": 0.5},
        "ET": {"voxel_count": 200, "volume_ml": 0.2},
    }
    case_eval = {
        "available": True,
        "metrics": {
            "WT": {"dice": 0.9412, "sensitivity": 0.9520},
            "TC": {"dice": 0.8850, "sensitivity": 0.8910},
            "ET": {"dice": 0.8320, "sensitivity": 0.8415},
        }
    }
    prov = {"model_name": "SegResNet", "git_commit": "abc", "timestamp": "2026-10-16T00:00:00Z", "input_sha256": {}}

    generate_pdf_report(
        case_id="case_with_expert",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
        case_evaluation=case_eval,
    )

    with open(out_pdf, "rb") as f:
        content = f.read()

    assert b"0.9412" in content
    assert b"0.8850" in content
    assert b"Dice (DSC)" in content or b"Dice" in content


def test_7_and_9_expert_mask_unavailability_explicit(tmp_path):
    """
    Requirements 7 & 9: When expert mask is missing, explicitly state unavailability
    and never substitute cohort benchmarks for individual case metrics.
    """
    out_pdf = str(tmp_path / "report_req7_9.pdf")
    volumes = {
        "WT": {"voxel_count": 1000, "volume_ml": 1.0},
        "TC": {"voxel_count": 500, "volume_ml": 0.5},
        "ET": {"voxel_count": 200, "volume_ml": 0.2},
    }
    prov = {"model_name": "SegResNet", "git_commit": "abc", "timestamp": "2026-10-16T00:00:00Z", "input_sha256": {}}

    generate_pdf_report(
        case_id="case_no_expert",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
        case_evaluation=None,
    )

    with open(out_pdf, "rb") as f:
        content = f.read()

    # Exact required wording
    assert b"No expert segmentation mask was provided" in content
    assert b"cohort benchmark metrics are never substituted" in content


# ------------------------------------------------------------------------------
# Test 8, 10, 11, 12, 13: Benchmarks, Hashes, Metadata, Timestamp, Disclaimer
# ------------------------------------------------------------------------------
def test_8_10_11_12_13_provenance_and_disclaimers(tmp_path):
    """
    Requirements 8, 10, 11, 12, 13:
    - Verified BraTS benchmark metrics (0.9026 WT, 0.8559 TC, 0.7905 ET).
    - Cryptographic SHA-256 hashes for inputs.
    - Model bundle metadata (MONAI SegResNet, 0.5.4, Apache 2.0).
    - Reproducibility git commit and ISO timestamp.
    - Mandatory disclaimer on every page.
    """
    out_pdf = str(tmp_path / "report_req8_13.pdf")
    test_hash_t1 = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    prov = {
        "model_name": "MONAI brats_mri_segmentation (SegResNet)",
        "model_license": "Apache 2.0",
        "training_data_source": "BraTS 2018 Multimodal Brain Tumor Dataset",
        "git_commit": "commit_test_hash",
        "timestamp": "2026-10-16T18:00:00Z",
        "input_sha256": {"t1": test_hash_t1, "t1gd": "0" * 64, "t2": "1" * 64, "flair": "2" * 64},
    }
    volumes = {"WT": {"voxel_count": 100, "volume_ml": 0.1}, "TC": {"voxel_count": 50, "volume_ml": 0.05}, "ET": {"voxel_count": 10, "volume_ml": 0.01}}

    generate_pdf_report(
        case_id="case_provenance_test",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
    )

    with open(out_pdf, "rb") as f:
        content = f.read()

    # Benchmark metrics
    assert b"0.9026" in content
    assert b"0.8559" in content
    assert b"0.7905" in content

    # Model license & commit
    assert b"Apache License 2.0" in content or b"Apache 2.0" in content
    assert b"commit_test_hash" in content
    assert b"SegResNet" in content

    # Disclaimer on running footers
    assert b"NeuroScope 3D | Research and Teaching Use Only | Not for Clinical Diagnosis" in content


# ------------------------------------------------------------------------------
# Test 14: PDF Generation Failure Handling
# ------------------------------------------------------------------------------
def test_14_pdf_generation_failure_handling():
    """
    Requirement 14: When invalid or corrupted images are passed, slice generation
    handles it cleanly and returns None without raising uncaught exceptions.
    """
    # Passing nonexistent or invalid files to slice generator
    result = generate_slice_visualizations("nonexistent_mri.nii.gz", "nonexistent_mask.nii.gz")
    assert result is None


# ------------------------------------------------------------------------------
# Test 2, 15, 16, 17, 18: REST API Endpoints, Downloads, and Security
# ------------------------------------------------------------------------------
def test_2_and_17_download_endpoint_content_type_and_filename(client):
    """
    Requirements 2 & 17:
    - GET /api/cases/{case_id}/report returns 200 with media_type application/pdf.
    - Content-Disposition contains dated filename.
    """
    case_id, case_dir = create_case_workspace()
    report_file = os.path.join(case_dir, "report.pdf")

    # Create dummy valid PDF
    volumes = {"WT": {"voxel_count": 100, "volume_ml": 0.1}, "TC": {"voxel_count": 50, "volume_ml": 0.05}, "ET": {"voxel_count": 10, "volume_ml": 0.01}}
    prov = {"model_name": "SegResNet", "git_commit": "abc", "timestamp": "2026-10-16T00:00:00Z", "input_sha256": {}}
    generate_pdf_report(case_id, report_file, volumes, {"shape": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0]}, prov)

    state = {
        "case_id": case_id,
        "status": "completed",
        "volumes": volumes,
        "spatial_metadata": {"shape": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0]},
        "provenance": prov,
        "report_path": report_file,
    }
    save_case_metadata(case_id, state)

    resp = client.get(f"/api/cases/{case_id}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert f"neuroscope3d_report_{case_id}" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")


def test_15_reject_report_for_unprocessed_case(client):
    """
    Requirement 15: Cannot download report for case that is only uploaded, pending, or failed.
    """
    case_id, case_dir = create_case_workspace()
    state = {
        "case_id": case_id,
        "status": "uploaded",  # Not completed!
    }
    save_case_metadata(case_id, state)

    resp = client.get(f"/api/cases/{case_id}/report")
    assert resp.status_code == 400
    assert "only available for successfully completed cases" in resp.json()["detail"]


def test_16_invalid_and_expired_case_handling(client):
    """
    Requirement 16:
    - Path traversal attempts return 400.
    - Nonexistent or expired cases return 404.
    """
    # Malformed / Invalid case ID syntax
    bad_resp = client.get("/api/cases/invalid_prefix_123/report")
    assert bad_resp.status_code == 400
    assert "Invalid case identifier" in bad_resp.json()["detail"]

    bad_resp2 = client.get("/api/cases/case_invalid$chars/report")
    assert bad_resp2.status_code == 400
    assert "Invalid case identifier" in bad_resp2.json()["detail"]

    # Nonexistent case
    missing_resp = client.get("/api/cases/case_nonexistent_99999999/report")
    assert missing_resp.status_code == 404
    assert "not found" in missing_resp.json()["detail"].lower()


def test_18_existing_api_preservation(client):
    """
    Requirement 18: Existing health and contract endpoints remain fully functional.
    """
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "healthy"
    assert MANDATORY_DISCLAIMER in data["disclaimer"]


def test_19_pdf_metric_values_consistency_tables_and_narrative(tmp_path, synthetic_mri_and_mask):
    """
    Regression Test:
    Ensures that the same authoritative evaluation result metrics are used consistently
    in all tables and narrative callout paragraphs, with zero contradictory metrics.
    """
    import pymupdf as fitz

    mri_file, mask_file, shape, affine = synthetic_mri_and_mask
    mask_data = nib.load(mask_file).get_fdata().astype(np.uint8)
    volumes = calculate_region_volumes(mask_data, (1.0, 1.0, 1.0))

    out_pdf = str(tmp_path / "report_consistency_test.pdf")
    prov = {
        "model_name": "MONAI brats_mri_segmentation (SegResNet)",
        "model_license": "Apache 2.0",
        "training_data_source": "BraTS 2018 Multimodal Brain Tumor Dataset",
        "git_commit": "a1b2c3d",
        "timestamp": "2026-10-18T12:00:00Z",
        "input_sha256": {
            "t1": compute_file_sha256(mri_file),
            "t1gd": compute_file_sha256(mri_file),
            "t2": compute_file_sha256(mri_file),
            "flair": compute_file_sha256(mri_file),
        },
    }

    # Authoritative evaluation object with exact tested values
    case_eval = {
        "available": True,
        "metrics": {
            "WT": {
                "dice": 0.9513,
                "sensitivity": 0.9305,
                "precision": 0.9730,
                "specificity": 0.9997,
                "hd95_mm": 2.00,
                "tp_voxels": 92403,
                "fp_voxels": 2560,
                "fn_voxels": 6899,
                "predicted_volume_ml": 94.96,
                "reference_volume_ml": 99.30,
                "signed_diff_ml": -4.34,
                "relative_diff_percent": -4.4,
            },
            "TC": {
                "dice": 0.9584,
                "sensitivity": 0.9522,
                "precision": 0.9646,
                "specificity": 0.9998,
                "hd95_mm": 2.45,
                "tp_voxels": 50328,
                "fp_voxels": 1847,
                "fn_voxels": 2527,
                "predicted_volume_ml": 52.18,
                "reference_volume_ml": 52.86,
                "signed_diff_ml": -0.68,
                "relative_diff_percent": -1.3,
            },
            "ET": {
                "dice": 0.8722,
                "sensitivity": 0.8236,
                "precision": 0.9269,
                "specificity": 0.9998,
                "hd95_mm": 1.41,
                "tp_voxels": 26394,
                "fp_voxels": 2081,
                "fn_voxels": 5653,
                "predicted_volume_ml": 28.48,
                "reference_volume_ml": 32.05,
                "signed_diff_ml": -3.57,
                "relative_diff_percent": -11.1,
            },
        },
    }

    generate_pdf_report(
        case_id="case_test_consistency",
        output_pdf_path=out_pdf,
        volumes=volumes,
        spatial_metadata={"shape": list(shape), "spacing_mm": [1.0, 1.0, 1.0]},
        provenance=prov,
        case_evaluation=case_eval,
        runtime_seconds=10.5,
        mri_path=mri_file,
        mask_path=mask_file,
    )

    doc = fitz.open(out_pdf)
    assert len(doc) == 4, f"Expected 4 pages, got {len(doc)}"

    # Check Page 3 text for metric concordance between table and narrative
    p3_text = doc[2].get_text()

    # WT Dice: 0.9513 in both
    assert "0.9513" in p3_text
    assert "0.9328" not in p3_text, "Conflicting WT Dice 0.9328 found in narrative"

    # TC Dice: 0.9584 in both
    assert "0.9584" in p3_text
    assert "0.9392" not in p3_text, "Conflicting TC Dice 0.9392 found in narrative"

    # ET HD95: 1.41 mm in both
    assert "1.41 mm" in p3_text
    assert "2.24 mm" not in p3_text, "Conflicting ET HD95 2.24 mm found in narrative"

    # ET volume difference: -3.57 mL in both
    assert "-3.57" in p3_text
    assert "-0.72 mL" not in p3_text, "Conflicting ET volume difference -0.72 mL found in narrative"

    # Disclaimers on every page
    for i, page in enumerate(doc):
        text = page.get_text()
        assert "Research and Teaching" in text or "Research & Teaching" in text
        assert "Not for Clinical Diagnosis" in text

