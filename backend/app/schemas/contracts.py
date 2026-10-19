"""
NeuroScope 3D - Pydantic Contracts and Schemas
Adheres strictly to docs/API_CONTRACT.md and PRD_BASE.md.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

MANDATORY_DISCLAIMER = "Research and teaching prototype only. Not for clinical diagnosis."


class ModelReadiness(BaseModel):
    loaded: bool = False
    model_name: str = "brats_mri_segmentation"
    device: str = "cpu"


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    commit_id: Optional[str] = None
    timestamp: str
    model_readiness: ModelReadiness = Field(default_factory=ModelReadiness)
    disclaimer: str = MANDATORY_DISCLAIMER


class FileMetadata(BaseModel):
    filename: str
    size_bytes: int
    sha256: str


class ExpertMaskMetadata(BaseModel):
    present: bool = False
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class SpatialSummary(BaseModel):
    shape: List[int]
    spacing_mm: List[float]


class CaseUploadResponse(BaseModel):
    case_id: str
    created_at: str
    files: Dict[str, Any]
    validation_status: str
    spatial_summary: Optional[SpatialSummary] = None
    warnings: Optional[List[str]] = None
    anatomical_sanity: Optional[Dict[str, Any]] = None
    volumes: Optional[Dict[str, Any]] = None
    disclaimer: str = MANDATORY_DISCLAIMER


class CaseValidationResponse(BaseModel):
    case_id: str
    is_valid: bool
    checks: Dict[str, str]
    reference_affine: Optional[List[List[float]]] = None
    warnings: Optional[List[str]] = None
    anatomical_sanity: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    disclaimer: str = MANDATORY_DISCLAIMER


class SegmentationTriggerResponse(BaseModel):
    case_id: str
    status: str
    message: str


class CaseStatusResponse(BaseModel):
    case_id: str
    status: str  # queued | processing | completed | failed
    progress_percentage: int = 0
    runtime_seconds: Optional[float] = None
    error: Optional[str] = None
    disclaimer: str = MANDATORY_DISCLAIMER


class BoundingBox(BaseModel):
    min: List[int]
    max: List[int]


class RegionVolume(BaseModel):
    voxel_count: int
    volume_ml: float
    voxel_volume_mm3: Optional[float] = None
    formula: Optional[str] = None
    measurement_status: Optional[str] = None
    bounding_box_voxel: Optional[BoundingBox] = None
    centroid_voxel: Optional[List[int]] = None
    centroid_voxel_fractional: Optional[List[float]] = None
    bounding_box_world_mm: Optional[Dict[str, List[float]]] = None
    centroid_world_mm: Optional[List[float]] = None


class CaseMetric(BaseModel):
    dice: Optional[float] = None
    sensitivity: Optional[float] = None
    precision: Optional[float] = None
    specificity: Optional[float] = None
    hd95_mm: Optional[float] = None
    predicted_volume_ml: Optional[float] = None
    reference_volume_ml: Optional[float] = None
    signed_diff_ml: Optional[float] = None
    absolute_diff_ml: Optional[float] = None
    relative_diff_percent: Optional[float] = None
    tp_voxels: Optional[int] = None
    fp_voxels: Optional[int] = None
    fn_voxels: Optional[int] = None
    tn_voxels: Optional[int] = None


class CaseEvaluation(BaseModel):
    available: bool
    reason: Optional[str] = None
    alignment_status: Optional[str] = None
    metric_conventions: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, CaseMetric]] = None


class Provenance(BaseModel):
    model_name: str
    model_license: str
    training_data_source: str
    git_commit: Optional[str] = None
    timestamp: str
    input_sha256: Dict[str, str]
    execution_mode: Optional[str] = None
    checkpoint_file: Optional[str] = None
    demonstration_notice: Optional[str] = None
    timing_breakdown: Optional[Dict[str, Any]] = None


class CaseOutputs(BaseModel):
    mask_url: str
    report_url: str
    agreement_mask_url: Optional[str] = None
    uncertainty_map_url: Optional[str] = None
    probability_map_url: Optional[str] = None
    boundary_mask_url: Optional[str] = None
    summary_url: Optional[str] = None


class SpatialMetadata(BaseModel):
    dimensions: List[int]
    voxel_spacing_mm: List[float]
    suggested_crosshair_voxel: Optional[List[int]] = None
    suggested_crosshair_world_mm: Optional[List[float]] = None


class CaseResultsResponse(BaseModel):
    case_id: str
    status: str
    runtime_seconds: Optional[float] = None
    spatial_metadata: SpatialMetadata
    regions: Dict[str, RegionVolume]  # WT, TC, ET
    case_evaluation: Optional[CaseEvaluation] = None
    held_out_benchmark_reference: Optional[Dict[str, Any]] = None
    provenance: Provenance
    outputs: CaseOutputs
    boundary_watch: Optional[Dict[str, Any]] = None
    disclaimer: str = MANDATORY_DISCLAIMER


class StandardErrorResponse(BaseModel):
    status: str = "error"
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    disclaimer: str = MANDATORY_DISCLAIMER
