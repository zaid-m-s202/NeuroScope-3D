import React, { useState, useCallback } from 'react';
import {
  UploadCloud, FileCheck, AlertCircle, CheckCircle2,
  Loader2, Play, FolderUp, X, RefreshCw, AlertTriangle, Info, FileArchive
} from 'lucide-react';
import { uploadStudy, triggerSegmentation, getCaseResults, loadSampleStudy, uploadDicomStudy } from '../services/api';

// ── BraTS filename patterns for auto-detection ──────────────────────────────
const SEQUENCE_PATTERNS = [
  { id: 'expert_mask', patterns: ['seg', 'mask', 'label', 'gt'], label: 'Expert Mask' },
  { id: 't1gd',        patterns: ['t1ce', 't1gd', 't1c', 'ce'],  label: 'T1Gd' },
  { id: 'flair',       patterns: ['flair'],                        label: 'FLAIR' },
  { id: 't2',          patterns: ['t2'],                           label: 'T2' },
  { id: 't1',          patterns: ['t1'],                           label: 'T1' },
];

function detectSequenceId(filename) {
  const lower = filename.toLowerCase();
  for (const { id, patterns } of SEQUENCE_PATTERNS) {
    if (patterns.some(p => lower.includes(p))) return id;
  }
  return null;
}

const SEQUENCE_SPECS = [
  { id: 't1',     code: 'T1',   label: 'T1 Native',                    req: true,  hint: 'Ends with _t1.nii.gz' },
  { id: 't1gd',   code: 'T1Gd', label: 'T1Gd Contrast (T1CE)',         req: true,  hint: 'Ends with _t1ce.nii.gz' },
  { id: 't2',     code: 'T2',   label: 'T2-Weighted',                  req: true,  hint: 'Ends with _t2.nii.gz' },
  { id: 'flair',  code: 'FLAIR','label': 'FLAIR Volumetric',           req: true,  hint: 'Ends with _flair.nii.gz' },
  {
    id: 'expert_mask', code: 'MASK', label: 'BraTS 2021 Expert Mask (Optional)',
    req: false, hint: 'Optional — ends with _seg.nii.gz'
  },
];

export default function UploadPanel({
  onCaseUploaded,
  onSegmentationCompleted,
  onBaseScanSelected,
  onValidationError,
  files: externalFiles,
  onFilesChange,
  uploadResult: externalUploadResult,
  onUploadResultChange,
  activeCase,
}) {
  const [internalFiles, setInternalFiles] = useState({
    t1: null, t1gd: null, t2: null, flair: null, expert_mask: null,
  });
  const [internalUploadResult, setInternalUploadResult] = useState(null);

  const files = externalFiles !== undefined ? externalFiles : internalFiles;
  const setFiles = useCallback((updater) => {
    if (onFilesChange) {
      if (typeof updater === 'function') {
        onFilesChange((prev) => updater(prev || files));
      } else {
        onFilesChange(updater);
      }
    } else {
      setInternalFiles(updater);
    }
  }, [onFilesChange, files]);

  const uploadResult = externalUploadResult !== undefined
    ? externalUploadResult
    : (internalUploadResult || activeCase || null);

  const setUploadResult = useCallback((updater) => {
    if (onUploadResultChange) {
      if (typeof updater === 'function') {
        onUploadResultChange((prev) => updater(prev || uploadResult));
      } else {
        onUploadResultChange(updater);
      }
    } else {
      setInternalUploadResult(updater);
    }
  }, [onUploadResultChange, uploadResult]);

  // Tracks which sequence is being manually re-assigned via drag
  const [pendingAssign, setPendingAssign] = useState(null);   // file waiting for slot click
  const [conflicts, setConflicts]         = useState([]);     // [{file, detectedAs, existingAs}]
  const [warnings, setWarnings]           = useState([]);
  const [isUploading, setIsUploading]     = useState(false);
  const [isSegmenting, setIsSegmenting]   = useState(false);
  const [errorMessage, setErrorMessage]   = useState(null);

  // ── Helpers ─────────────────────────────────────────────────────────────
  const isNiftiFile = (file) => {
    const n = file.name.toLowerCase();
    return n.endsWith('.nii') || n.endsWith('.nii.gz');
  };

  const clearState = () => {
    setErrorMessage(null);
    setConflicts([]);
    setWarnings([]);
    setUploadResult(null);
  };

  // ── Bulk file assignment with conflict detection ─────────────────────────
  const handleBulkFiles = useCallback(async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    clearState();

    // Direct clinical DICOM ZIP archive upload
    if (fileList.length === 1 && fileList[0].name.toLowerCase().endsWith('.zip')) {
      const zipFile = fileList[0];
      setIsUploading(true);
      try {
        const resp = await uploadDicomStudy(zipFile);
        setUploadResult(resp);
        onCaseUploaded(resp);
        if (resp.warnings && resp.warnings.length > 0) {
          setWarnings(resp.warnings);
        }
      } catch (err) {
        setErrorMessage(buildErrorMessage(err));
        if (onValidationError) onValidationError(err);
      } finally {
        setIsUploading(false);
      }
      return;
    }

    const updated = { ...files };
    const newConflicts = [];
    const rejected = [];

    Array.from(fileList).forEach((file) => {
      if (!isNiftiFile(file)) {
        rejected.push(file.name);
        return;
      }
      const detected = detectSequenceId(file.name);
      if (!detected) {
        // Cannot auto-detect — queue for manual assignment
        setPendingAssign(file);
        return;
      }
      // Conflict: slot already occupied by a different file
      if (updated[detected] && updated[detected].name !== file.name) {
        newConflicts.push({ file, detectedAs: detected, existing: updated[detected] });
        return;
      }
      updated[detected] = file;
    });

    setFiles(updated);
    setConflicts(newConflicts);

    if (rejected.length > 0) {
      setWarnings(w => [...w, `Ignored non-NIfTI files: ${rejected.join(', ')}`]);
    }

    // Auto-select viewer base
    if (updated.t1gd) onBaseScanSelected(updated.t1gd);
    else if (updated.t1) onBaseScanSelected(updated.t1);
  }, [files, onBaseScanSelected]);

  const resolveConflict = (conflict, keepNew) => {
    const updated = { ...files };
    if (keepNew) updated[conflict.detectedAs] = conflict.file;
    setFiles(updated);
    setConflicts(c => c.filter(x => x !== conflict));
  };

  // ── Individual slot assignment ────────────────────────────────────────────
  const handleIndividualFileChange = (seqId, file) => {
    if (!file) return;
    if (!isNiftiFile(file)) {
      setErrorMessage(`'${file.name}' is not a NIfTI file (.nii or .nii.gz). Please select a valid file.`);
      return;
    }
    // Warn if the auto-detected type doesn't match the target slot (not an error, just informational)
    const detected = detectSequenceId(file.name);
    const newWarnings = [];
    if (detected && detected !== seqId && detected !== 'expert_mask') {
      newWarnings.push(`'${file.name}' filename suggests ${detected.toUpperCase()} but assigned to ${seqId.toUpperCase()}.`);
    }
    setWarnings(newWarnings);

    const updated = { ...files, [seqId]: file };
    setFiles(updated);
    setErrorMessage(null);
    if (uploadResult) {
      setUploadResult({
        ...uploadResult,
        validation_status: 'pending',
        files: {
          ...uploadResult.files,
          [seqId]: { filename: file.name, size_bytes: file.size }
        }
      });
    } else {
      setUploadResult(null);
    }

    if (seqId === 't1gd' && file) onBaseScanSelected(file);
    else if (seqId === 't1' && file && !files.t1gd) onBaseScanSelected(file);
  };

  const handleRemoveFile = (seqId) => {
    setFiles(prev => ({ ...prev, [seqId]: null }));
    if (uploadResult?.files) {
      const updatedFiles = { ...uploadResult.files };
      delete updatedFiles[seqId];
      setUploadResult({
        ...uploadResult,
        files: updatedFiles,
        validation_status: 'pending',
      });
    } else {
      setUploadResult(null);
    }
    setErrorMessage(null);
  };

  // ── Drag & Drop ──────────────────────────────────────────────────────────
  const handleFileDrop = (e) => {
    e.preventDefault();
    handleBulkFiles(e.dataTransfer.files);
  };

  // ── Upload ───────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    setErrorMessage(null);

    const requiredKeys = ['t1', 't1gd', 't2', 'flair'];
    const missing = requiredKeys.filter(k => !files[k]);
    if (missing.length > 0) {
      setErrorMessage(
        `Missing required sequences: ${missing.map(s => s.toUpperCase()).join(', ')}. ` +
        `All four sequences (T1, T1Gd, T2, FLAIR) must be provided.`
      );
      return null;
    }

    if (conflicts.length > 0) {
      setErrorMessage('Please resolve all sequence conflicts before uploading.');
      return null;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('t1', files.t1);
      formData.append('t1gd', files.t1gd);
      formData.append('t2', files.t2);
      formData.append('flair', files.flair);
      if (files.expert_mask && files.expert_mask.size > 0) {
        formData.append('expert_mask', files.expert_mask);
      }

      const resp = await uploadStudy(formData);
      setUploadResult(resp);
      onCaseUploaded(resp);

      if (resp.warnings && resp.warnings.length > 0) {
        setWarnings(resp.warnings);
      }
      return resp;
    } catch (err) {
      const msg = err.message || 'Validation failed during case upload';
      setErrorMessage(buildErrorMessage(err));
      if (onValidationError) onValidationError(err);
      return null;
    } finally {
      setIsUploading(false);
    }
  };

  // ── Sample study ─────────────────────────────────────────────────────────
  const handleLoadSample = async () => {
    setIsUploading(true);
    clearState();
    try {
      const resp = await loadSampleStudy();
      setUploadResult(resp);
      onCaseUploaded(resp);
      if (resp.warnings && resp.warnings.length > 0) setWarnings(resp.warnings);

      // Automatically execute SegResNet segmentation on sample study for instant demonstration
      setIsUploading(false);
      setIsSegmenting(true);
      await triggerSegmentation(resp.case_id);
      const results = await getCaseResults(resp.case_id);
      onSegmentationCompleted(results);
    } catch (err) {
      setErrorMessage(`Sample load failed: ${err.message}`);
    } finally {
      setIsUploading(false);
      setIsSegmenting(false);
    }
  };

  // ── Run segmentation ─────────────────────────────────────────────────────
  const handleRunSegmentation = async () => {
    let currentCase = uploadResult;
    if (!currentCase?.case_id || currentCase?.validation_status !== 'passed') {
      currentCase = await handleUpload();
      if (!currentCase?.case_id) return;
    }
    setIsSegmenting(true);
    setErrorMessage(null);
    try {
      await triggerSegmentation(currentCase.case_id);
      const results = await getCaseResults(currentCase.case_id);
      onSegmentationCompleted(results);
    } catch (err) {
      setErrorMessage(err.message || 'Segmentation inference encountered an error');
    } finally {
      setIsSegmenting(false);
    }
  };

  // ── Error message builder ────────────────────────────────────────────────
  function buildErrorMessage(err) {
    const code = err.code || '';
    const base = err.message || 'Upload failed';
    const remedies = {
      MISSING_SEQUENCE:          'Ensure all four sequences (T1, T1Gd, T2, FLAIR) are assigned before uploading.',
      MISMATCHED_SHAPES:         'All four sequences must have identical spatial dimensions. Re-export with co-registration.',
      MISMATCHED_AFFINES:        'All sequences must share the same coordinate system. Use co-registered volumes.',
      DIMENSION_MISMATCH_NOT_3D: 'Each file must be a 3D NIfTI volume. Reject 2D slices or 4D time series.',
      NON_FINITE_VALUES:         'The image contains NaN or infinite values. Re-export or choose a different file.',
      CONSTANT_VOLUME:           'The image has no signal variation (all-zero or uniform). Verify the correct file was selected.',
      CORRUPT_GZIP:              'The gzip archive is corrupt. Re-download or re-compress the file.',
      INVALID_NIFTI:             'The file cannot be read as a NIfTI image. Verify the file format.',
      FILE_TOO_LARGE:            'Each file must be under 1 GB. Compress the file or crop the field of view.',
      INVALID_EXPERT_MASK_LABELS:'Expert mask contains labels not in {0, 1, 2, 4}. Use BraTS 2021 label convention.',
      REFUSAL_NOT_A_BRAIN:       'The image does not appear to be a skull-stripped brain MRI. Verify the scan.',
    };
    const hint = remedies[code];
    return hint ? `${base}\n\n→ ${hint}` : base;
  }

  // ── Derived state ────────────────────────────────────────────────────────
  const assignedCount = ['t1', 't1gd', 't2', 'flair'].filter(
    k => files[k] || uploadResult?.files?.[k]?.filename || uploadResult?.files?.[k]?.present
  ).length;
  const requiredReady  = ['t1', 't1gd', 't2', 'flair'].every(
    k => files[k] !== null || uploadResult?.files?.[k]?.filename || uploadResult?.files?.[k]?.present
  );
  const validationPassed = uploadResult?.validation_status === 'passed' ||
    (uploadResult?.status === 'completed' && ['t1', 't1gd', 't2', 'flair'].every(k => uploadResult?.files?.[k]));
  const segButtonReady = !isUploading && !isSegmenting;

  const spatialMeta = uploadResult?.spatial_summary || uploadResult?.spatial_metadata;

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div className="clinical-section-header">
        <h2 className="section-title-text" style={{ display: 'flex', alignItems: 'center' }}>
          <span className="section-number-badge">1</span>
          Study Ingestion &amp; Upload
        </h2>
        <span className="font-mono" style={{ fontSize: '11px', color: 'var(--on-surface-variant)' }}>
          {assignedCount}/4
          {validationPassed ? ' ✓ VALIDATED' : requiredReady ? ' READY' : ' PENDING'}
        </span>
      </div>

      {/* Drag & Drop Bulk Upload Zone */}
      <div
        className="dropzone-container"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
        onClick={() => document.getElementById('bulk-folder-input')?.click()}
      >
        <FolderUp size={28} style={{ color: 'var(--primary-accent)', marginBottom: '4px' }} />
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--on-surface-dark)' }}>
          Auto-Assign: Select All Patient MRI Files
        </span>
        <span style={{ fontSize: '11px', color: 'var(--on-surface-variant)', marginTop: '2px' }}>
          Supports uncompressed or gzip NIfTI (.nii, .nii.gz) — auto-detected from BraTS filenames
        </span>
        <span style={{ fontSize: '11px', color: 'var(--primary-accent)', fontWeight: 600, marginTop: '4px' }}>
          Batch Drag &amp; Drop Enabled
        </span>
        <input
          id="bulk-folder-input"
          type="file"
          multiple
          accept=".nii,.nii.gz"
          style={{ display: 'none' }}
          onChange={(e) => handleBulkFiles(e.target.files)}
        />
      </div>

      {/* Pending manual assignment prompt */}
      {pendingAssign && (
        <div style={{
          backgroundColor: 'var(--surface-container-low)',
          border: '1px solid var(--secondary-blue)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 10px',
          fontSize: '11px',
          color: 'var(--on-surface-dark)',
          display: 'flex', alignItems: 'flex-start', gap: '6px'
        }}>
          <Info size={14} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--secondary-blue)' }} />
          <div>
            <strong>'{pendingAssign.name}'</strong> could not be auto-detected.
            Click <strong>Assign</strong> on the correct sequence slot below, then dismiss.
            <button
              style={{ marginLeft: '8px', fontSize: '10px', cursor: 'pointer', color: 'var(--error-red)' }}
              onClick={() => setPendingAssign(null)}
            >Dismiss</button>
          </div>
        </div>
      )}

      {/* Conflict resolution panel */}
      {conflicts.length > 0 && (
        <div style={{
          backgroundColor: 'var(--warning-container, #fff3cd)',
          border: '1px solid var(--warning-border, #f0ad4e)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 10px',
          display: 'flex', flexDirection: 'column', gap: '6px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700 }}>
            <AlertTriangle size={13} />
            Sequence Conflicts — please resolve:
          </div>
          {conflicts.map((c, i) => (
            <div key={i} style={{ fontSize: '10px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span>
                <strong>{c.file.name}</strong> auto-detected as <strong>{c.detectedAs.toUpperCase()}</strong>,
                but that slot already has <strong>{c.existing.name}</strong>.
              </span>
              <button
                style={{ fontSize: '10px', cursor: 'pointer', padding: '1px 6px', border: '1px solid #888', borderRadius: 3 }}
                onClick={() => resolveConflict(c, true)}
              >Use new</button>
              <button
                style={{ fontSize: '10px', cursor: 'pointer', padding: '1px 6px', border: '1px solid #888', borderRadius: 3 }}
                onClick={() => resolveConflict(c, false)}
              >Keep existing</button>
            </div>
          ))}
        </div>
      )}

      {/* Sequence Slots */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {SEQUENCE_SPECS.map((spec) => {
          const selectedFile  = files[spec.id];
          const serverInfo    = uploadResult?.files?.[spec.id];
          const isAssigned    = !!(selectedFile || serverInfo?.filename);
          const fileName      = selectedFile?.name || serverInfo?.filename || 'Not assigned';
          const sizeLabel     = selectedFile?.size
            ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`
            : serverInfo?.size_bytes
            ? `${(serverInfo.size_bytes / (1024 * 1024)).toFixed(1)} MB`
            : '--';

          // Per-slot validation status
          let slotStatus = 'empty';
          if (isAssigned && validationPassed) slotStatus = 'validated';
          else if (isAssigned) slotStatus = 'assigned';

          return (
            <div key={spec.id} className="sequence-slot-card">
              {/* Left: badge + label + filename */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
                <span className="seq-code-badge"
                  style={slotStatus === 'validated' ? { backgroundColor: 'var(--success-green, #2ecc71)', color: '#fff' } : {}}
                >
                  {spec.code}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--on-surface-dark)', lineHeight: 1.2 }}>
                    {spec.label}
                    {spec.req && <span style={{ color: 'var(--error-red)', marginLeft: 3 }}>*</span>}
                  </span>
                  <span
                    className="font-mono"
                    style={{
                      fontSize: '10px',
                      color: isAssigned ? 'var(--on-surface-variant)' : 'var(--outline)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      maxWidth: '190px',
                    }}
                    title={fileName}
                  >
                    {fileName}
                  </span>
                </div>
              </div>

              {/* Right: size + status + actions */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                {isAssigned && (
                  <span className="font-mono" style={{ fontSize: '10px', color: 'var(--on-surface-variant)' }}>
                    {sizeLabel}
                  </span>
                )}

                {slotStatus === 'validated' && (
                  <span className="status-badge-valid">✓ Validated</span>
                )}
                {slotStatus === 'assigned' && (
                  <span style={{
                    fontSize: '10px', fontWeight: 600,
                    color: 'var(--secondary-blue)',
                    border: '1px solid var(--secondary-blue)',
                    padding: '2px 5px', borderRadius: 'var(--radius-xs)',
                  }}>Selected</span>
                )}

                {/* Assign from pending file */}
                {pendingAssign && (
                  <button
                    style={{
                      fontSize: '10px', fontWeight: 600, cursor: 'pointer',
                      color: '#fff', backgroundColor: 'var(--secondary-blue)',
                      border: 'none', padding: '2px 6px', borderRadius: 'var(--radius-xs)',
                    }}
                    onClick={() => {
                      handleIndividualFileChange(spec.id, pendingAssign);
                      setPendingAssign(null);
                    }}
                  >Assign</button>
                )}

                {/* Browse / Replace button */}
                <label
                  style={{
                    fontSize: '10px', fontWeight: 600,
                    color: 'var(--primary-accent)',
                    border: '1px solid var(--secondary-blue)',
                    padding: '2px 6px', borderRadius: 'var(--radius-xs)',
                    cursor: 'pointer',
                    backgroundColor: 'var(--surface-container-low)',
                  }}
                >
                  {isAssigned ? 'Replace' : 'Browse'}
                  <input
                    type="file"
                    accept=".nii,.nii.gz"
                    style={{ display: 'none' }}
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleIndividualFileChange(spec.id, e.target.files[0]);
                      }
                      e.target.value = '';   // allow re-selecting same file
                    }}
                  />
                </label>

                {/* Remove button */}
                {isAssigned && (
                  <button
                    title="Remove this file"
                    aria-label={`Remove ${spec.label}`}
                    className="remove-seq-btn"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--on-surface-variant)', padding: '2px',
                    }}
                    onClick={() => handleRemoveFile(spec.id)}
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Validation summary after successful upload */}
      {validationPassed && spatialMeta && (
        <div style={{
          backgroundColor: 'var(--surface-container-low)',
          border: '1px solid var(--success-green, #2ecc71)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 10px',
          fontSize: '11px',
          color: 'var(--on-surface-dark)',
        }}>
          <div style={{ fontWeight: 700, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <CheckCircle2 size={13} style={{ color: 'var(--success-green, #2ecc71)' }} />
            Study Validated — Ready for Segmentation
          </div>
          <div className="font-mono" style={{ fontSize: '10px', color: 'var(--on-surface-variant)' }}>
            Dimensions: {spatialMeta.shape?.join(' × ')} voxels
            &nbsp;|&nbsp;
            Spacing: {spatialMeta.spacing_mm?.map(s => s.toFixed(3)).join(' × ')} mm
          </div>
          {uploadResult?.anatomical_sanity && (
            <div className="font-mono" style={{ fontSize: '10px', color: 'var(--on-surface-variant)' }}>
              Anatomical sanity: {uploadResult.anatomical_sanity.sanity_status}
              &nbsp;|&nbsp;Foreground: {(uploadResult.anatomical_sanity.foreground_ratio * 100).toFixed(1)}%
            </div>
          )}
          {uploadResult?.has_expert_mask && (
            <div style={{ fontSize: '10px', color: 'var(--secondary-blue)', marginTop: '2px' }}>
              Expert mask: present — comparison metrics will be calculated
            </div>
          )}
          <div style={{
            marginTop: '8px',
            padding: '6px 8px',
            backgroundColor: '#EFF6FF',
            border: '1px solid #93C5FD',
            borderRadius: 'var(--radius-xs)',
            fontSize: '11px',
            color: '#1D4ED8',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontWeight: 500
          }}>
            <span>👉 <strong>Next Step:</strong> Click <strong>"Run SegResNet &amp; Compute Volumes"</strong> below or in Section 2 to generate 3D tumor masks and compute exact tissue volumes.</span>
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div style={{
          backgroundColor: 'var(--warning-container, #fff3cd)',
          border: '1px solid var(--warning-border, #f0ad4e)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 10px',
          fontSize: '11px',
          color: 'var(--on-surface-dark)',
          display: 'flex', flexDirection: 'column', gap: '3px',
        }}>
          {warnings.map((w, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '5px' }}>
              <AlertTriangle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Error message */}
      {errorMessage && (
        <div style={{
          backgroundColor: 'var(--error-container)',
          border: '1px solid var(--error-red)',
          padding: '8px 10px',
          borderRadius: 'var(--radius-sm)',
          fontSize: '11px',
          color: 'var(--error-dark)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '6px',
          whiteSpace: 'pre-wrap',
        }}>
          <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>{errorMessage}</div>
        </div>
      )}

      {/* Missing sequences summary */}
      {!requiredReady && !errorMessage && (
        <div style={{
          fontSize: '10px', color: 'var(--outline)',
          display: 'flex', gap: '4px', flexWrap: 'wrap'
        }}>
          {['t1', 't1gd', 't2', 'flair'].filter(k => !files[k] && !uploadResult?.files?.[k]?.filename && !uploadResult?.files?.[k]?.present).map(k => (
            <span key={k} style={{
              border: '1px dashed var(--outline)',
              borderRadius: 'var(--radius-xs)',
              padding: '1px 5px',
              fontWeight: 600,
            }}>
              {k.toUpperCase()} missing
            </span>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
        {/* Upload/Validate button — visible when files selected but not yet validated */}
        {!validationPassed && requiredReady && (
          <button
            type="button"
            className="btn-secondary"
            onClick={handleUpload}
            disabled={isUploading || conflicts.length > 0}
          >
            {isUploading ? (
              <><Loader2 size={16} className="spin" /><span>Validating Study Geometry...</span></>
            ) : (
              <><FileCheck size={16} /><span>Validate &amp; Upload Study</span></>
            )}
          </button>
        )}

        {/* Segmentation button — disabled until validation passes */}
        <button
          type="button"
          className="btn-primary"
          onClick={handleRunSegmentation}
          disabled={!segButtonReady || (!validationPassed && !requiredReady)}
          title={!validationPassed ? 'Upload and validate all four sequences first' : 'Run AI segmentation and calculate physical volumes'}
        >
          {isSegmenting ? (
            <><Loader2 size={16} className="spin" /><span>Calculating Tumor Volumes (SegResNet)...</span></>
          ) : isUploading ? (
            <><Loader2 size={16} className="spin" /><span>Waiting for Validation...</span></>
          ) : (
            <><Play size={16} /><span>Run SegResNet &amp; Compute Volumes (mL)</span></>
          )}
        </button>

        <button
          type="button"
          className="btn-secondary"
          onClick={handleLoadSample}
          disabled={isUploading || isSegmenting}
          title="Load pre-packaged BraTS2021 multi-sequence case"
        >
          <span>Load Sample BraTS Study</span>
        </button>
      </div>
    </section>
  );
}
