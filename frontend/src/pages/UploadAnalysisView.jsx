import React from 'react';
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Play,
  Eye,
  FileCheck,
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  BarChart3
} from 'lucide-react';
import UploadPanel from '../components/UploadPanel';

export default function UploadAnalysisView({
  onCaseUploaded,
  onSegmentationCompleted,
  onBaseScanSelected,
  onValidationError,
  activeCase,
  caseResults,
  onOpenWorkstation,
  onOpenAnalytics,
  ingestionFiles,
  onIngestionFilesChange,
  uploadResult,
  onUploadResultChange,
}) {
  const effectiveCaseId = caseResults?.case_id || activeCase?.case_id;
  const isSegmented = caseResults?.status === 'completed';
  const isValidated = activeCase?.validation_status === 'passed';

  return (
    <div className="upload-analysis-page-canvas">
      {/* 1. Header with Step Progress Bar */}
      <div className="upload-page-header">
        <div>
          <div className="upload-page-badge">
            <UploadCloud size={14} className="text-primary-blue" />
            <span>4-MODAL MRI INGESTION &amp; SEGRESNET INFERENCE PIPELINE</span>
          </div>
          <h1 className="upload-page-title">Study Ingestion, Validation &amp; Segmentation</h1>
          <p className="upload-page-desc">
            Standardized BraTS multi-parametric MRI upload. Pre-inference affine check guarantees identical
            matrix dimensions and spatial calibration before neural network evaluation.
          </p>
        </div>

        {effectiveCaseId && (
          <div className="upload-header-actions">
            <button
              type="button"
              className="btn-primary"
              onClick={onOpenWorkstation}
              style={{ fontSize: '11px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Eye size={14} />
              <span>Open in 3D Workstation</span>
            </button>
            {isSegmented && (
              <button
                type="button"
                className="btn-secondary"
                onClick={onOpenAnalytics}
                style={{ fontSize: '11px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <BarChart3 size={14} />
                <span>Volumetric Breakdown</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* 2. Visual 4-Step Stepper Header */}
      <div className="upload-stepper-bar">
        {/* Step 1 */}
        <div className={`stepper-step ${activeCase ? 'completed' : 'active'}`}>
          <div className="stepper-step-num">
            {activeCase ? <CheckCircle2 size={14} /> : '1'}
          </div>
          <div className="stepper-step-info">
            <span className="stepper-step-title">1. File Ingestion</span>
            <span className="stepper-step-subtitle">4 Modalities + GT</span>
          </div>
        </div>

        <div className="stepper-connector" />

        {/* Step 2 */}
        <div className={`stepper-step ${isValidated ? 'completed' : activeCase ? 'active' : 'pending'}`}>
          <div className="stepper-step-num">
            {isValidated ? <CheckCircle2 size={14} /> : '2'}
          </div>
          <div className="stepper-step-info">
            <span className="stepper-step-title">2. Spatial Validation</span>
            <span className="stepper-step-subtitle">Affine &amp; Dimensions</span>
          </div>
        </div>

        <div className="stepper-connector" />

        {/* Step 3 */}
        <div className={`stepper-step ${isSegmented ? 'completed' : isValidated ? 'active' : 'pending'}`}>
          <div className="stepper-step-num">
            {isSegmented ? <CheckCircle2 size={14} /> : '3'}
          </div>
          <div className="stepper-step-info">
            <span className="stepper-step-title">3. Neural Inference</span>
            <span className="stepper-step-subtitle">MONAI SegResNet</span>
          </div>
        </div>

        <div className="stepper-connector" />

        {/* Step 4 */}
        <div className={`stepper-step ${isSegmented ? 'completed' : 'pending'}`}>
          <div className="stepper-step-num">
            {isSegmented ? <CheckCircle2 size={14} /> : '4'}
          </div>
          <div className="stepper-step-info">
            <span className="stepper-step-title">4. Quantitation &amp; 3D</span>
            <span className="stepper-step-subtitle">WT, TC, ET Volumes</span>
          </div>
        </div>
      </div>

      {/* 3. Main Step Content Container */}
      <div className="upload-step-content-grid">
        {/* Left Column: The Upload & Action Panel */}
        <div className="upload-panel-main-column">
          <UploadPanel
            onCaseUploaded={onCaseUploaded}
            onSegmentationCompleted={onSegmentationCompleted}
            onBaseScanSelected={onBaseScanSelected}
            onValidationError={onValidationError}
            files={ingestionFiles}
            onFilesChange={onIngestionFilesChange}
            uploadResult={uploadResult}
            onUploadResultChange={onUploadResultChange}
            activeCase={activeCase}
          />
        </div>

        {/* Right Column: Pre-Inference Validation Specifications & Hardware Guidance */}
        <div className="upload-panel-guidance-column">
          {/* Spatial Requirements Card */}
          <div className="guidance-card">
            <div className="guidance-card-header">
              <FileCheck size={16} className="text-primary-blue" />
              <h3 className="guidance-card-title">Pre-Inference Validation Rules</h3>
            </div>
            <ul className="guidance-list">
              <li>
                <strong>4 Co-Registered Volumes:</strong> Native T1, Contrast-Enhanced T1ce, T2-Weighted, and FLAIR are mandatory.
              </li>
              <li>
                <strong>Identical Matrix Geometry:</strong> All 4 volumes must possess matching spatial dimensions (e.g. 240 × 240 × 155).
              </li>
              <li>
                <strong>Isotropic Voxel Spacing:</strong> Standard 1.0mm × 1.0mm × 1.0mm grid ensures uniform physical millilitre quantitation.
              </li>
              <li>
                <strong>Skull-Stripped Inputs:</strong> Scalp fat and bone marrow should be extracted to avoid false-positive perimeter oedema.
              </li>
            </ul>
          </div>

          {/* Model Pipeline Execution Spec */}
          <div className="guidance-card">
            <div className="guidance-card-header">
              <Cpu size={16} className="text-secondary-blue" />
              <h3 className="guidance-card-title">SegResNet Execution Pipeline</h3>
            </div>
            <div className="guidance-pipeline-steps">
              <div className="pipeline-step-item">
                <span className="pipeline-step-badge">TRANSFORM</span>
                <span className="pipeline-step-desc">Z-score brain intensity normalization &amp; 4-channel tensor assembly</span>
              </div>
              <div className="pipeline-step-item">
                <span className="pipeline-step-badge">WINDOW</span>
                <span className="pipeline-step-desc">128³ sliding-window inference with 50% Gaussian overlap blending</span>
              </div>
              <div className="pipeline-step-item">
                <span className="pipeline-step-badge">HIERARCHY</span>
                <span className="pipeline-step-desc">Multi-compartment decoding: WT (1,2,4), TC (1,4), ET (4)</span>
              </div>
              <div className="pipeline-step-item">
                <span className="pipeline-step-badge">PHYSICAL</span>
                <span className="pipeline-step-desc">Deterministic affine matrix determinant volume quantitation</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
