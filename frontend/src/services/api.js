/**
 * NeuroScope 3D - REST API Service Client
 * Interacts directly with the FastAPI backend defined in docs/API_CONTRACT.md.
 */

const RENDER_BACKEND_URL = 'https://neuroscope-3d-backend.onrender.com';

export const API_BASE = (() => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.startsWith('http')) {
    return envUrl.replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') {
      return '';
    }
  }
  return RENDER_BACKEND_URL;
})();

async function safeParseJson(res, defaultError = 'Request failed') {
  const contentType = res.headers?.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return await res.json().catch(() => ({}));
  }
  const text = await res.text().catch(() => '');
  if (res.status === 405) {
    throw new Error('Endpoint returned HTTP 405 Method Not Allowed. Ensure backend routing is active and not returning static HTML.');
  }
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new Error('Backend service is waking up from sleep on Render (free tier cold start takes ~50s). Please wait a moment and retry.');
  }
  if (!res.ok) {
    throw new Error(`${defaultError} (HTTP ${res.status})`);
  }
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error('Backend service is warming up or returned non-JSON. Please allow 30–60 seconds for Render cold start.');
  }
}

/**
 * Backend operational status and model readiness
 */
export async function getHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) {
      throw new Error(`Health check failed: ${res.statusText}`);
    }
    return await safeParseJson(res, 'Health check failed');
  } catch (err) {
    throw new Error(`Cannot reach FastAPI backend server (${err.message})`);
  }
}

/**
 * Upload 4 required MRI sequences and optional expert mask
 */
export async function uploadStudy(formData) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases`, {
      method: 'POST',
      body: formData,
    });
  } catch (netErr) {
    throw new Error(`Cannot reach FastAPI backend server. If using Render free tier, the backend may be cold-starting (${netErr.message})`);
  }

  const data = await safeParseJson(res, 'Upload failed');
  if (!res.ok) {
    const errorMsg = data.message || `Upload failed with HTTP ${res.status}`;
    const err = new Error(errorMsg);
    err.code = data.code || 'UPLOAD_FAILED';
    err.details = data.details || {};
    throw err;
  }
  return data;
}

/**
 * Validate case geometry
 */
export async function validateCase(caseId) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/validate`);
  } catch (netErr) {
    throw new Error(`Validation check failed: could not reach backend (${netErr.message})`);
  }
  if (!res.ok) {
    throw new Error(`Validation request failed with HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Trigger real model inference
 */
export async function triggerSegmentation(caseId) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/segment`, {
      method: 'POST',
    });
  } catch (netErr) {
    throw new Error(`Inference request failed: could not reach backend (${netErr.message})`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.message || `Segmentation failed with HTTP ${res.status}`);
    err.code = data.code || 'SEGMENTATION_ERROR';
    throw err;
  }
  return data;
}

/**
 * Check processing status
 */
export async function getCaseStatus(caseId) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/status`);
  } catch (netErr) {
    throw new Error(`Failed to query case status: could not reach backend (${netErr.message})`);
  }
  if (!res.ok) {
    throw new Error(`Failed to query case status: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Get results, volumes, metrics, and provenance
 */
export async function getCaseResults(caseId) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/results`);
  } catch (netErr) {
    throw new Error(`Failed to fetch case results: could not reach backend (${netErr.message})`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || `Failed to fetch results with HTTP ${res.status}`);
  }
  return data;
}

/**
 * Return download URLs
 */
export function getMaskDownloadUrl(caseId) {
  return `${API_BASE}/api/cases/${caseId}/mask`;
}

export function getReportDownloadUrl(caseId) {
  return `${API_BASE}/api/cases/${caseId}/report`;
}

export function getSequenceDownloadUrl(caseId, seqKey) {
  const cleanKey = String(seqKey).replace(/\.nii(\.gz)?$/, '');
  return `${API_BASE}/api/cases/${caseId}/sequences/${cleanKey}.nii.gz`;
}

export function getAgreementMaskUrl(caseId, region = 'WT') {
  return `${API_BASE}/api/cases/${caseId}/agreement-mask?region=${encodeURIComponent(region)}`;
}

export function getProbabilityMapUrl(caseId, channel = 'max') {
  return `${API_BASE}/api/cases/${caseId}/probabilities/${encodeURIComponent(channel)}`;
}

export function getBoundaryMaskUrl(caseId) {
  return `${API_BASE}/api/cases/${caseId}/boundary-mask`;
}

export async function getProbabilityMetadata(caseId) {
  try {
    const res = await fetch(`${API_BASE}/api/cases/${caseId}/probabilities`);
    if (!res.ok) return { available: false };
    return await res.json();
  } catch (_) {
    return { available: false };
  }
}

/**
 * Initiates binary download of the 4-class expert agreement NIfTI volume (TP/FP/FN).
 */
export async function downloadAgreementMaskFile(caseId, region = 'WT') {
  const url = getAgreementMaskUrl(caseId, region);
  const res = await fetch(url);
  if (!res.ok) {
    let errMsg = `Agreement mask download failed (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errJson.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }
  const blob = await res.blob();
  if (blob.size === 0) {
    throw new Error('Downloaded agreement mask is empty.');
  }

  let filename = `${caseId}_${region}_agreement.nii.gz`;
  const disposition = res.headers.get('Content-Disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^";]+)"?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(blobUrl);

  return { filename, sizeBytes: blob.size };
}

/**
 * Initiates binary download of the clinical provenance PDF report.
 * Uses blob streaming to verify content before saving to the user's filesystem.
 */
export async function downloadReportFile(caseId) {
  const url = getReportDownloadUrl(caseId);
  const res = await fetch(url);
  if (!res.ok) {
    let errMsg = `Report download failed (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errJson.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }
  const blob = await res.blob();
  if (blob.size === 0) {
    throw new Error('Downloaded PDF report is empty.');
  }

  let filename = `neuroscope3d_report_${caseId}.pdf`;
  const disposition = res.headers.get('Content-Disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^";]+)"?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(blobUrl);

  return { filename, sizeBytes: blob.size };
}

/**
 * Initiates binary download of the predicted BraTS 2021 segmentation mask.
 */
export async function downloadMaskFile(caseId) {
  const url = getMaskDownloadUrl(caseId);
  const res = await fetch(url);
  if (!res.ok) {
    let errMsg = `Mask download failed (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errJson.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }
  const blob = await res.blob();
  if (blob.size === 0) {
    throw new Error('Downloaded segmentation mask is empty.');
  }

  let filename = `${caseId}_seg_brats2021.nii.gz`;
  const disposition = res.headers.get('Content-Disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^";]+)"?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(blobUrl);

  return { filename, sizeBytes: blob.size };
}

/**
 * Load bundled sample BraTS study
 */
export async function loadSampleStudy() {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/sample`, {
      method: 'POST',
    });
  } catch (netErr) {
    throw new Error(`Cannot reach FastAPI backend server. Ensure backend is running at http://127.0.0.1:8000 (${netErr.message})`);
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const errorMsg = data.message || `Sample load failed with HTTP ${res.status}`;
    const err = new Error(errorMsg);
    err.code = data.code || 'SAMPLE_LOAD_FAILED';
    err.details = data.details || {};
    throw err;
  }
  return data;
}

/**
 * Upload clinical DICOM archive (.zip)
 */
export async function uploadDicomStudy(zipFile) {
  const formData = new FormData();
  formData.append('file', zipFile);
  const res = await fetch(`${API_BASE}/api/cases/dicom`, {
    method: 'POST',
    body: formData,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.message || `DICOM upload failed with HTTP ${res.status}`);
    err.code = data.code || 'DICOM_UPLOAD_FAILED';
    err.details = data.details || {};
    throw err;
  }
  return data;
}

/**
 * Query algorithmic uncertainty quantification profile (Shannon entropy & boundary ambiguity)
 */
export async function getUncertaintyProfile(caseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/uncertainty`);
  if (!res.ok) {
    throw new Error(`Failed to load uncertainty profile: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Query modality ablation sensitivity profile
 */
export async function getModalityAblation(caseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/ablation`);
  if (!res.ok) {
    throw new Error(`Failed to load modality ablation profile: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Query 10-case held-out benchmark cohort manifest & overlap disclosure
 */
export async function getCohortBenchmark() {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/evaluation/cohort`);
  } catch (netErr) {
    throw new Error(`Cannot reach backend server. If using Render free tier, the backend may be cold-starting (takes ~50s): ${netErr.message}`);
  }
  return await safeParseJson(res, 'Failed to load cohort benchmark');
}

/**
 * Compare two longitudinal timepoints for the same patient (RANO delta)
 */
export async function compareLongitudinalStudies(caseId, followupCaseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/longitudinal?followup_case_id=${encodeURIComponent(followupCaseId)}`, {
    method: 'POST',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || `Longitudinal comparison failed (HTTP ${res.status})`);
  }
  return data;
}

/**
 * Get 3D surface mesh download URL
 */
export function getMeshDownloadUrl(caseId, region = 'WT', format = 'glb') {
  return `${API_BASE}/api/cases/${caseId}/mesh/${region}?format=${format}`;
}

/**
 * Download 3D polygonal surface mesh in world mm space (.glb or .obj)
 */
export async function downloadMeshFile(caseId, region = 'WT', format = 'glb') {
  const url = getMeshDownloadUrl(caseId, region, format);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Mesh download failed with HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const filename = `${caseId}_${region.toLowerCase()}_mesh.${format}`;
  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(blobUrl);
  return { filename, sizeBytes: blob.size };
}

/**
 * Get machine-readable research summary JSON URL
 */
export function getSummaryDownloadUrl(caseId) {
  return `${API_BASE}/api/cases/${caseId}/summary`;
}

/**
 * Download machine-readable research summary JSON file
 */
export async function downloadSummaryFile(caseId) {
  const url = getSummaryDownloadUrl(caseId);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Summary JSON download failed with HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const filename = `neuroscope3d_summary_${caseId}.json`;
  const blobUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(blobUrl);
  return { filename, sizeBytes: blob.size };
}

/**
 * Upload an expert reference mask to an existing case
 */
export async function uploadExpertMask(caseId, file) {
  const formData = new FormData();
  formData.append('expert_mask', file);

  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/expert-mask`, {
      method: 'POST',
      body: formData,
    });
  } catch (netErr) {
    throw new Error(`Failed to upload expert mask: could not reach backend (${netErr.message})`);
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const errorMsg = data.detail || data.message || `Upload failed with HTTP ${res.status}`;
    throw new Error(errorMsg);
  }
  return data;
}

/**
 * Get detailed case volumetric measurements and formulas
 */
export async function getCaseVolumetrics(caseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/volumetrics`);
  if (!res.ok) {
    throw new Error(`Failed to fetch volumetrics (HTTP ${res.status})`);
  }
  return res.json();
}

/**
 * Triggers or computes physical tumor volumetrics on demand (WT, TC, ET)
 */
export async function calculateCaseVolumes(caseId) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/volumetrics`, {
      method: 'POST',
    });
  } catch (netErr) {
    throw new Error(`Volumetric calculation request failed: could not reach backend (${netErr.message})`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || `Volume calculation failed with HTTP ${res.status}`);
  }
  return data;
}

/**
 * Get quantitative evaluation against expert reference
 */
export async function getCaseEvaluation(caseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/evaluation`);
  if (!res.ok) {
    throw new Error(`Failed to fetch evaluation (HTTP ${res.status})`);
  }
  return res.json();
}

/**
 * List all active sessions in temporary storage
 */
export async function getActiveCases() {
  const res = await fetch(`${API_BASE}/api/cases`);
  if (!res.ok) {
    throw new Error(`Failed to fetch active cases (HTTP ${res.status})`);
  }
  return res.json();
}

/**
 * Get Boundary Watch quality flags, component analysis, and slice continuity
 */
export async function getBoundaryWatch(caseId) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/boundary-watch`);
  if (!res.ok) {
    let errMsg = `Failed to fetch Boundary Watch flags (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errJson.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }
  return res.json();
}

/**
 * Generates an authoritative two-timepoint longitudinal research demonstration pair
 */
export async function loadSampleLongitudinalPair() {
  const res = await fetch(`${API_BASE}/api/cases/sample-longitudinal-pair`, {
    method: 'POST',
  });
  if (!res.ok) {
    let errMsg = `Failed to create longitudinal sample pair (HTTP ${res.status})`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errJson.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }
  return res.json();
}
