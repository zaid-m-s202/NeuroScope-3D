import React, { useState, useEffect } from 'react';
import {
  Layers,
  UploadCloud,
  ChevronRight,
  ChevronLeft,
  Eye,
  Sliders,
  BarChart3,
  RefreshCw,
  FolderOpen,
  PanelRightClose,
  PanelRightOpen,
  Sparkles
} from 'lucide-react';
import UploadPanel from '../components/UploadPanel';
import NiiVueViewer from '../components/NiiVueViewer';
import VolumetricCards from '../components/VolumetricCards';
import BoundaryWatchPanel from '../components/BoundaryWatchPanel';
import EvaluationPanel from '../components/EvaluationPanel';
import DownloadsPanel from '../components/DownloadsPanel';
import { loadSampleStudy, triggerSegmentation, getCaseResults, calculateCaseVolumes, getMaskDownloadUrl } from '../services/api';

export default function Dashboard({
  activeCase,
  caseResults,
  baseScanFile,
  baseScanUrl,
  maskUrl,
  activeSequenceKey,
  onSelectSequence,
  onCaseUploaded,
  onSegmentationCompleted,
  onBaseScanSelected,
  onValidationError,
  workstationSubTab = 'viewer',
  onWorkstationSubTabChange,
  ingestionFiles,
  onIngestionFilesChange,
  uploadResult,
  onUploadResultChange,
}) {
  const [selectedCentroid, setSelectedCentroid] = useState(null);
  const [overlayMode, setOverlayMode] = useState('prediction'); // 'prediction' | 'agreement'
  const [agreementRegion, setAgreementRegion] = useState('WT'); // 'WT' | 'TC' | 'ET'
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [sidePanelTab, setSidePanelTab] = useState('quant'); // 'quant' | 'upload'
  const [isSidePanelCollapsed, setIsSidePanelCollapsed] = useState(false);

  const [layerVisibility, setLayerVisibility] = useState({
    WT: true,
    TC: true,
    ET: true,
  });

  const handleToggleLayer = (layerKey) => {
    setLayerVisibility((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  };

  const handleActivateAgreementLens = (reg) => {
    setAgreementRegion(reg || 'WT');
    setOverlayMode('agreement');
  };

  const handleExpertMaskUploaded = (uploadRes) => {
    if (uploadRes?.case_evaluation && caseResults) {
      onSegmentationCompleted({
        ...caseResults,
        case_evaluation: uploadRes.case_evaluation,
      });
    }
  };

  const effectiveVolumes = caseResults?.regions || activeCase?.volumes || activeCase?.regions;
  const effectiveCaseId = caseResults?.case_id || activeCase?.case_id;

  // Sync default centroid whenever new segmentation results arrive
  useEffect(() => {
    if (caseResults?.spatial_metadata?.suggested_crosshair_voxel) {
      setSelectedCentroid(caseResults.spatial_metadata.suggested_crosshair_voxel);
    } else if (effectiveVolumes?.WT?.centroid_voxel) {
      setSelectedCentroid(effectiveVolumes.WT.centroid_voxel);
    } else {
      setSelectedCentroid(null);
    }
  }, [caseResults?.spatial_metadata?.suggested_crosshair_voxel, effectiveVolumes?.WT?.centroid_voxel]);

  const seqList = [
    { key: 't1', label: 'T1 Native', code: 'T1' },
    { key: 't1gd', label: 'T1CE Contrast', code: 'T1Gd' },
    { key: 't2', label: 'T2-Weighted', code: 'T2' },
    { key: 'flair', label: 'T2 FLAIR', code: 'FLAIR' },
  ];

  const handleCalculateVolumes = async () => {
    const caseIdToUse = effectiveCaseId;
    if (!caseIdToUse) {
      throw new Error('Please upload and validate all 4 MRI sequences in Step 1 first.');
    }
    setIsSegmenting(true);
    try {
      if (caseResults?.status === 'completed' || activeCase?.has_mask) {
        await calculateCaseVolumes(caseIdToUse);
        const results = await getCaseResults(caseIdToUse);
        onSegmentationCompleted(results);
      } else {
        await triggerSegmentation(caseIdToUse);
        const results = await getCaseResults(caseIdToUse);
        onSegmentationCompleted(results);
      }
    } catch (err) {
      console.error('Volume calculation failed:', err);
      throw err;
    } finally {
      setIsSegmenting(false);
    }
  };

  const handleLoadSampleDirect = async () => {
    setIsSegmenting(true);
    try {
      const resp = await loadSampleStudy();
      onCaseUploaded(resp);
      await triggerSegmentation(resp.case_id);
      const results = await getCaseResults(resp.case_id);
      onSegmentationCompleted(results);
    } catch (err) {
      console.error('Failed to load sample study:', err);
      throw err;
    } finally {
      setIsSegmenting(false);
    }
  };

  return (
    <div className="viewer-first-workstation-layout">
      {/* 1. CENTRAL / PRIMARY AREA: Viewer-First Canvas */}
      <main className="workstation-primary-canvas">
        {/* Sequence Switcher Pill Bar (Placed near the viewer as requested) */}
        <div className="viewer-sequence-strip">
          <div className="sequence-selector-group">
            <span className="sequence-strip-label">Sequence:</span>
            {seqList.map((seq) => {
              const isActive = activeSequenceKey === seq.key;
              return (
                <button
                  key={seq.key}
                  type="button"
                  className={`sequence-pill-btn ${isActive ? 'active' : ''}`}
                  onClick={() => onSelectSequence && onSelectSequence(seq.key)}
                  title={`Display ${seq.label} in viewport`}
                >
                  <span className="sequence-pill-code">{seq.code}</span>
                  <span className="sequence-pill-name">{seq.label}</span>
                </button>
              );
            })}
          </div>

          <div className="sequence-strip-actions">
            <button
              type="button"
              className="side-panel-toggle-btn"
              onClick={() => setIsSidePanelCollapsed((prev) => !prev)}
              title={isSidePanelCollapsed ? 'Open side panel' : 'Collapse side panel for full-width viewer'}
            >
              {isSidePanelCollapsed ? (
                <>
                  <PanelRightOpen size={14} />
                  <span>Show Side Panel</span>
                </>
              ) : (
                <>
                  <PanelRightClose size={14} />
                  <span>Expand Viewer</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* The 2D / 3D NiiVue Interactive Viewport */}
        <div className="viewer-canvas-frame">
          <NiiVueViewer
            baseScanFile={baseScanFile}
            baseScanUrl={baseScanUrl}
            maskUrl={maskUrl || (effectiveCaseId && (caseResults?.status === 'completed' || activeCase?.files?.expert_mask?.present || activeCase?.has_mask || effectiveVolumes) ? getMaskDownloadUrl(effectiveCaseId) : null)}
            suggestedCrosshair={selectedCentroid}
            activeSequenceKey={activeSequenceKey}
            activeSequenceLabel={seqList.find((s) => s.key === activeSequenceKey)?.label || 'T1CE'}
            onSelectSequence={onSelectSequence}
            caseId={effectiveCaseId}
            activeCase={activeCase}
            caseResults={caseResults}
            hasExpertMask={caseResults?.case_evaluation?.available === true || activeCase?.files?.expert_mask?.present === true}
            layerVisibility={layerVisibility}
            onToggleLayer={handleToggleLayer}
            overlayMode={overlayMode}
            agreementRegion={agreementRegion}
            onSetOverlayMode={setOverlayMode}
            onSetAgreementRegion={setAgreementRegion}
            spatialMetadata={caseResults?.spatial_metadata || activeCase?.spatial_summary || activeCase?.spatial_metadata}
          />
        </div>
      </main>

      {/* 2. DEDICATED SIDE PANEL: Region Visibility, Opacity, Volumes, Boundary Watch, Export */}
      {!isSidePanelCollapsed && (
        <aside className="workstation-side-panel">
          {/* Side Panel Tabs Header */}
          <div className="side-panel-tabs-header">
            <button
              type="button"
              className={`side-panel-tab-btn ${sidePanelTab === 'quant' ? 'active' : ''}`}
              onClick={() => setSidePanelTab('quant')}
            >
              <BarChart3 size={14} />
              <span>Quantitation &amp; Layers</span>
            </button>
            <button
              type="button"
              className={`side-panel-tab-btn ${sidePanelTab === 'upload' ? 'active' : ''}`}
              onClick={() => setSidePanelTab('upload')}
            >
              <UploadCloud size={14} />
              <span>Study Ingestion</span>
            </button>
          </div>

          {/* Side Panel Body */}
          <div className="side-panel-scrollable-body">
            <div style={{ display: sidePanelTab === 'quant' ? 'flex' : 'none', flexDirection: 'column', gap: '14px' }}>
              <VolumetricCards
                volumes={effectiveVolumes}
                spatialMetadata={caseResults?.spatial_metadata || activeCase?.spatial_summary}
                onSelectCentroid={(vox) => {
                  setSelectedCentroid([...vox]);
                }}
                layerVisibility={layerVisibility}
                onToggleLayer={handleToggleLayer}
                onRunSegmentation={handleCalculateVolumes}
                onLoadSample={handleLoadSampleDirect}
                isSegmenting={isSegmenting}
                caseId={effectiveCaseId}
              />

              {caseResults && (
                <BoundaryWatchPanel
                  caseId={caseResults.case_id}
                  boundaryWatchData={caseResults.boundary_watch}
                  onNavigateToCentroid={(coords) => {
                    setSelectedCentroid([...coords]);
                  }}
                  onNavigateToSlice={(sliceZ) => {
                    const currentX = selectedCentroid?.[0] ?? 32;
                    const currentY = selectedCentroid?.[1] ?? 32;
                    setSelectedCentroid([currentX, currentY, sliceZ]);
                  }}
                />
              )}

              <EvaluationPanel
                caseId={effectiveCaseId}
                evaluation={caseResults?.case_evaluation}
                benchmarkReference={caseResults?.held_out_benchmark_reference}
                volumes={effectiveVolumes}
                onExpertMaskUploaded={handleExpertMaskUploaded}
                onActivateAgreementLens={(reg) => {
                  handleActivateAgreementLens(reg);
                }}
              />

              {caseResults && (
                <DownloadsPanel
                  caseId={caseResults.case_id}
                  provenance={caseResults.provenance}
                  caseResults={caseResults}
                />
              )}
            </div>

            <div style={{ display: sidePanelTab === 'upload' ? 'block' : 'none' }}>
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
          </div>
        </aside>
      )}
    </div>
  );
}
