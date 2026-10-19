import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Niivue, NVImage, cmapper } from '@niivue/niivue';
import * as nifti from 'nifti-reader-js';
import { Eye, Check, Layers, Sliders, Target, RotateCcw, Activity } from 'lucide-react';
import { getAgreementMaskUrl, getMaskDownloadUrl, getProbabilityMapUrl, getBoundaryMaskUrl, getProbabilityMetadata } from '../services/api';
import { voxelToWorld, normalizeAffine, getAnatomicalOrientation } from '../utils/coordinates';

// Medical Imaging Heat Map Color Scale:
// 0.00: Transparent (Background healthy brain)
// 0.10: Dark Blue (very low predicted probability)
// 0.30: Cyan (low-to-moderate probability)
// 0.55: Green/Yellow (moderate-to-high probability)
// 0.85: Orange/Red (high predicted probability)
const MEDICAL_HEATMAP_CMAP = {
  R: [0,   0,   0,   0, 255, 255, 230],
  G: [0,  20, 210, 230, 215,  80,  20],
  B: [0, 150, 255,  30,   0,   0,  20],
  A: [0,  80, 140, 180, 210, 235, 255],
  I: [0,  25,  75, 130, 185, 225, 255]
};

if (cmapper && typeof cmapper.addColormap === 'function') {
  try {
    cmapper.addColormap('medical_heatmap', MEDICAL_HEATMAP_CMAP);
  } catch (_) {}
}

function updateMaskLut(maskVol, mode, reg, layers, opacity) {
  if (!maskVol) return;
  const alphaVal = Math.round(opacity * 255);

  if (mode === 'agreement') {
    const agreementLut = {
      R: [0, 34, 239, 168],
      G: [0, 197, 68, 85],
      B: [0, 94, 68, 247],
      A: [0, alphaVal, alphaVal, alphaVal],
      I: [0, 1, 2, 3],
      labels: ['TN (Transparent)', 'TP (Consensus)', 'FP (Over-predicted)', 'FN (Missed)'],
    };
    if (typeof maskVol.setColormapLabel === 'function') {
      maskVol.setColormapLabel(agreementLut);
    }
    maskVol.opacity = opacity;
  } else if (mode === 'heatmap') {
    // Continuous probability heat map with smooth color gradient
    maskVol.colormapLabel = null;
    maskVol.colormap = 'medical_heatmap';
    maskVol.cal_min = 0.01;
    maskVol.cal_max = 1.0;
    maskVol.opacity = opacity;
  } else if (mode === 'boundary') {
    // Discrete boundary contour lines (1-voxel thickness for crisp delineation)
    const showWT = layers?.WT ?? true;
    const showTC = layers?.TC ?? true;
    const showET = layers?.ET ?? true;
    const boundaryLut = {
      R: [0,   6,  34, 0, 250],
      G: [0, 182, 197, 0, 204],
      B: [0, 212,  94, 0,  21],
      A: [
        0,                                 // 0: BG transparent
        showTC ? 255 : 0,                  // 1: TC contour (Cyan)
        showWT ? 255 : 0,                  // 2: WT contour (Emerald Green)
        0,                                 // 3: Unused
        showET ? 255 : 0,                  // 4: ET contour (Electric Gold)
      ],
      I: [0, 1, 2, 3, 4],
      labels: ['Background', 'TC Boundary', 'WT Boundary', 'Unused', 'ET Boundary'],
    };
    if (typeof maskVol.setColormapLabel === 'function') {
      maskVol.setColormapLabel(boundaryLut);
    }
    maskVol.opacity = 1.0;
  } else {
    // BraTS 2021 labels (Segmentation mode):
    // 0: Background (100% transparent)
    // 1: Necrotic Core (NCR) -> part of WT, TC
    // 2: Edema (ED) -> part of WT
    // 4: Enhancing Tumour (ET) -> part of WT, TC, ET
    const showWT = layers?.WT ?? true;
    const showTC = layers?.TC ?? true;
    const showET = layers?.ET ?? true;

    // HIGH-VISIBILITY CLINICAL COLOR PALETTE
    // 1: Necrotic Core (NCR) / Tumour Core -> Crisp Vivid Sky Blue / Cyan (#06B6D4)
    // 2: Peritumoral Edema (ED) / Whole Tumour -> High-Visibility Vivid Emerald Green (#22C55E)
    // 4: Enhancing Tumour (ET) -> High-Visibility Electric Gold / Yellow (#FACC15)
    const bratsLut = {
      R: [0,   6,  34, 0, 250],
      G: [0, 182, 197, 0, 204],
      B: [0, 212,  94, 0,  21],
      A: [
        0,                                 // 0: BG transparent
        showTC ? alphaVal : 0,             // 1: NCR / TC component (Cyan)
        showWT ? alphaVal : 0,             // 2: ED / WT component (Emerald Green)
        0,                                 // 3: Unused
        showET ? alphaVal : 0,             // 4: ET (Vivid Electric Gold/Yellow)
      ],
      I: [0, 1, 2, 3, 4],
      labels: [
        'Background',
        'Necrotic Core (NCR)',
        'Peritumoral Edema (ED)',
        'Unused',
        'Enhancing Tumour (ET)',
      ],
    };
    if (typeof maskVol.setColormapLabel === 'function') {
      maskVol.setColormapLabel(bratsLut);
    }
    maskVol.opacity = opacity;
  }
}

export default function NiiVueViewer({
  baseScanFile,
  baseScanUrl,
  maskUrl,
  suggestedCrosshair,
  activeSequenceKey = 't1gd',
  activeSequenceLabel = 'T1CE',
  caseId,
  hasExpertMask = false,
  layerVisibility: propLayerVisibility,
  onToggleLayer: propOnToggleLayer,
  overlayMode: propOverlayMode,
  agreementRegion: propAgreementRegion,
  onSetOverlayMode: propOnSetOverlayMode,
  onSetAgreementRegion: propOnSetAgreementRegion,
  caseResults,
  activeCase,
  spatialMetadata,
}) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);

  const [currentSliceType, setCurrentSliceType] = useState('multi');
  const [crosshairPos, setCrosshairPos] = useState(null);
  const [volumeDims, setVolumeDims] = useState(null);
  const [isViewerReady, setIsViewerReady] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [wlPreset, setWlPreset] = useState('Brain');
  const [statusMessage, setStatusMessage] = useState('Standby • Ready for multi-sequence MRI study');
  const [internalOverlayMode, setInternalOverlayMode] = useState('prediction'); // 'prediction' | 'heatmap' | 'combined' | 'agreement'
  const [internalAgreementRegion, setInternalAgreementRegion] = useState('WT'); // 'WT' | 'TC' | 'ET'
  const [probChannel, setProbChannel] = useState('max'); // 'max' | 'wt' | 'tc' | 'et'
  const [hasProbabilities, setHasProbabilities] = useState(true);
  const [baseLoadedCount, setBaseLoadedCount] = useState(0);

  const overlayMode = propOverlayMode || internalOverlayMode;
  const agreementRegion = propAgreementRegion || internalAgreementRegion;

  const setOverlayMode = (mode) => {
    if (propOnSetOverlayMode) {
      propOnSetOverlayMode(mode);
    } else {
      setInternalOverlayMode(mode);
    }
  };

  const setAgreementRegion = (reg) => {
    if (propOnSetAgreementRegion) {
      propOnSetAgreementRegion(reg);
    } else {
      setInternalAgreementRegion(reg);
    }
  };

  // Inspect probability availability for current active case
  useEffect(() => {
    if (!caseId) return;
    let isCancelled = false;
    getProbabilityMetadata(caseId)
      .then((meta) => {
        if (!isCancelled) {
          setHasProbabilities(meta?.available === true);
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setHasProbabilities(false);
        }
      });
    return () => {
      isCancelled = true;
    };
  }, [caseId]);

  const [internalLayerVisibility, setInternalLayerVisibility] = useState({
    WT: true,
    TC: true,
    ET: true,
  });

  const layerVisibility = propLayerVisibility || internalLayerVisibility;

  // Reset Camera / Scene action to canonical orientations
  const handleResetCamera = () => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady) return;
    try {
      if (typeof nv.resetScene === 'function') {
        nv.resetScene();
      }
      if (currentSliceType === 'multi') {
        nv.setMultiplanarLayout(2);
        nv.opts.multiplanarForceRender = true;
        nv.setCornerOrientationText(true);
      }
      if (suggestedCrosshair && Array.isArray(suggestedCrosshair) && suggestedCrosshair.length >= 3) {
        navigateToCentroid(suggestedCrosshair);
      } else if (typeof nv.vox2frac === 'function' && volumeDims) {
        const center = volumeDims.map((d) => Math.floor(d / 2));
        nv.scene.crosshairPos = nv.vox2frac(center, 0);
      }
      nv.updateGLVolume();
      setStatusMessage('Camera and multiplanar orientation reset to canonical views');
    } catch (e) {
      console.warn('Reset scene failed:', e.message);
    }
  };

  // Comprehensive Reset View action: restores multiplanar, camera, crosshair, opacity, and Brain W/L
  const handleResetView = () => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady) return;
    try {
      if (typeof nv.resetScene === 'function') {
        nv.resetScene();
      }
      handleSliceTypeChange('multi');
      setOverlayOpacity(0.75);
      setWlPreset('Brain');
      applyWlPreset('Brain');
      if (suggestedCrosshair && Array.isArray(suggestedCrosshair) && suggestedCrosshair.length >= 3) {
        navigateToCentroid(suggestedCrosshair);
      } else if (typeof nv.vox2frac === 'function' && volumeDims) {
        const center = volumeDims.map((d) => Math.floor(d / 2));
        nv.scene.crosshairPos = nv.vox2frac(center, 0);
      }
      nv.updateGLVolume();
      setStatusMessage('View reset: Multiplanar orientation, canonical camera, Brain W/L (75% opacity) restored');
    } catch (e) {
      console.warn('Reset view failed:', e.message);
    }
  };

  const getActiveAffine = useCallback(() => {
    const metaAff = spatialMetadata?.reference_affine
      || spatialMetadata?.affine
      || caseResults?.spatial_metadata?.reference_affine
      || caseResults?.spatial_metadata?.affine
      || activeCase?.spatial_metadata?.reference_affine
      || activeCase?.spatial_summary?.reference_affine;

    const spacing = spatialMetadata?.spacing_mm
      || caseResults?.spatial_metadata?.spacing_mm
      || activeCase?.spatial_metadata?.spacing_mm
      || activeCase?.spatial_summary?.spacing_mm
      || [1.0, 1.0, 1.0];

    if (metaAff) {
      return normalizeAffine(metaAff, spacing);
    }

    const nv = nvRef.current;
    if (nv && nv.volumes && nv.volumes.length > 0) {
      const vol = nv.volumes[0];
      if (vol.hdr?.affine) return normalizeAffine(vol.hdr.affine, spacing);
      if (vol.matRAS) return normalizeAffine(vol.matRAS, spacing);
      if (vol.hdr?.sform) return normalizeAffine(vol.hdr.sform, spacing);
    }

    return normalizeAffine(null, spacing);
  }, [spatialMetadata, caseResults, activeCase]);

  // Navigate crosshair to specific voxel indices and synchronize all views
  const navigateToCentroid = (coords) => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady || !nv.volumes || nv.volumes.length === 0) return;
    const target = coords || suggestedCrosshair;
    if (!target || !Array.isArray(target) || target.length < 3) return;

    try {
      const targetVox = [
        Math.round(Number(target[0]) || 0),
        Math.round(Number(target[1]) || 0),
        Math.round(Number(target[2]) || 0),
      ];
      const aff = getActiveAffine();
      const worldMm = voxelToWorld(targetVox, aff);

      if (typeof nv.vox2frac === 'function') {
        const frac = nv.vox2frac(targetVox, 0);
        if (frac && Array.isArray(frac) && frac.every((v) => Number.isFinite(v))) {
          nv.scene.crosshairPos = frac;
          nv.updateGLVolume();
        }
      }

      setCrosshairPos({
        vox: targetVox,
        mm: worldMm,
        val: 0,
      });
    } catch (e) {
      console.warn('Centroid navigation failed:', e.message);
    }
  };

  // Initialize NiiVue instance on mount
  useEffect(() => {
    if (!canvasRef.current) return;
    let isCancelled = false;

    const nv = new Niivue({
      backColor: [0.0, 0.0, 0.0, 1.0], // Clinical diagnostic pure black
      crosshairColor: [0.39, 0.82, 0.96, 0.9], // Cyan #65D0F4
      show3Dcrosshair: true,
      isColorbar: false,
      loadingText: 'Rendering Multi-Planar Voxel Grid...',
      isOrientCube: true,
    });

    async function initViewer() {
      try {
        await nv.attachToCanvas(canvasRef.current);
        if (isCancelled) return;
        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.setMultiplanarLayout(2); // MULTIPLANAR_TYPE.GRID = 2 (Forces 4 equal partitions)
        nv.opts.multiplanarForceRender = true; // Always render 3D Volume in 4th partition
        nv.opts.multiplanarPadPixels = 6;
        nv.setCornerOrientationText(true);

        // Crosshair location updates synchronized across multiplanar views
        nv.onLocationChange = (data) => {
          if (data && data.vox && Array.isArray(data.vox) && data.vox.length >= 3) {
            const vx = Math.round(Number(data.vox[0]) || 0);
            const vy = Math.round(Number(data.vox[1]) || 0);
            const vz = Math.round(Number(data.vox[2]) || 0);
            const vox = [vx, vy, vz];
            const aff = getActiveAffine();
            const worldMm = voxelToWorld(vox, aff);

            setCrosshairPos({
              vox,
              mm: worldMm,
              val: data.values && data.values.length > 0 ? (Number(data.values[0]) || 0) : 0,
            });
          }
        };

        nvRef.current = nv;
        setIsViewerReady(true);
      } catch (err) {
        console.error('Failed to attach Niivue to canvas:', err);
        setStatusMessage(`Viewer WebGL Error: ${err.message}`);
      }
    }

    initViewer();

    return () => {
      isCancelled = true;
      try {
        nv.destroy();
      } catch (e) {
        // ignore on unmount
      }
    };
  }, []);

  // Load Base Scan (either from File object or from URL)
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady) return;
    if (!baseScanFile && !baseScanUrl) return;

    let isCancelled = false;
    const name = baseScanFile ? baseScanFile.name : (baseScanUrl || 'Volume');
    setStatusMessage(`Ingesting sequence: ${activeSequenceLabel || name}...`);

    async function loadBase() {
      try {
        let volume;
        if (baseScanFile) {
          volume = await NVImage.loadFromFile({
            file: baseScanFile,
            colormap: 'gray',
            opacity: 1.0,
          });
        } else if (baseScanUrl) {
          volume = await NVImage.loadFromUrl({
            url: baseScanUrl,
            name: `${activeSequenceKey || 'sequence'}.nii.gz`,
            colormap: 'gray',
            opacity: 1.0,
          });
        }
        if (isCancelled || !volume) return;

        // Preserve existing overlays (segmentation, heatmap, or combined boundaries)
        const existingOverlays = (nv.volumes && nv.volumes.length > 1) ? nv.volumes.slice(1) : [];

        while (nv.volumes && nv.volumes.length > 0) {
          nv.removeVolumeByIndex(0);
        }

        nv.addVolume(volume);
        if (volume.robust_min !== undefined && volume.robust_max !== undefined) {
          volume.cal_min = volume.robust_min;
          volume.cal_max = Math.max(volume.robust_max * 1.05, volume.robust_min + 1.0);
        }
        for (let i = 0; i < existingOverlays.length; i++) {
          const ov = existingOverlays[i];
          nv.addVolume(ov);
          const volIdx = nv.volumes.length - 1;
          if (overlayMode === 'combined' && i === 1) {
            updateMaskLut(ov, 'boundary', agreementRegion, layerVisibility, 1.0);
            nv.setOpacity(volIdx, 1.0);
          } else if (overlayMode === 'heatmap' || (overlayMode === 'combined' && i === 0)) {
            updateMaskLut(ov, 'heatmap', agreementRegion, layerVisibility, overlayOpacity);
            nv.setOpacity(volIdx, overlayOpacity);
          } else {
            updateMaskLut(ov, overlayMode, agreementRegion, layerVisibility, overlayOpacity);
            nv.setOpacity(volIdx, overlayOpacity);
          }
        }

        if (volume.dims && volume.dims.length >= 4) {
          setVolumeDims([volume.dims[1], volume.dims[2], volume.dims[3]]);
        }

        nv.updateGLVolume();
        setBaseLoadedCount((c) => c + 1);
        setStatusMessage(`Active sequence: ${activeSequenceLabel || name}`);
      } catch (err) {
        if (!isCancelled) {
          console.error('Failed to display volume:', err);
          setStatusMessage(`Failed to display volume: ${err.message}`);
        }
      }
    }

    loadBase();

    return () => {
      isCancelled = true;
    };
  }, [baseScanFile, baseScanUrl, activeSequenceKey, activeSequenceLabel, isViewerReady]);

  // Helper to fetch and decompress NIfTI buffer
  const fetchNiftiBuffer = async (url, label) => {
    const res = await fetch(url);
    if (!res.ok) {
      if (res.status === 404) {
        throw new Error(`NOT_FOUND: ${label} not available (404)`);
      }
      let errDetail = res.statusText;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errJson.message || errDetail;
      } catch (_) {}
      throw new Error(`Failed to retrieve ${label} (HTTP ${res.status}: ${errDetail})`);
    }

    const arrayBuf = await res.arrayBuffer();
    if (arrayBuf.byteLength < 348) {
      throw new Error(`Retrieved ${label} payload is empty or corrupted.`);
    }

    let niftiBuffer = arrayBuf;
    if (nifti.isCompressed(arrayBuf)) {
      niftiBuffer = nifti.decompress(arrayBuf);
    }
    if (!nifti.isNIFTI(niftiBuffer)) {
      throw new Error(`Retrieved ${label} is not a valid NIfTI medical image.`);
    }

    const hdr = nifti.readHeader(niftiBuffer);
    if (!hdr || !hdr.dims) {
      throw new Error(`NIfTI volume header for ${label} is corrupt: missing dimensions.`);
    }

    return { niftiBuffer, hdr };
  };

  // Load and Validate Segmentation, Heatmap, Combined, or Agreement Overlay
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady) return;

    const isAgreement = overlayMode === 'agreement' && hasExpertMask && caseId;
    const isHeatmap = overlayMode === 'heatmap' && caseId;
    const isCombined = overlayMode === 'combined' && caseId;

    const hasCompletedResults = caseResults?.status === 'completed' || caseResults?.regions || activeCase?.volumes || activeCase?.files?.expert_mask?.present || activeCase?.has_mask;

    if (!isAgreement && !isHeatmap && !isCombined && (!maskUrl && (!caseId || !hasCompletedResults))) {
      // Clear any leftover overlay volumes when switching cases or resetting
      while (nv.volumes && nv.volumes.length > 1) {
        nv.removeVolumeByIndex(1);
      }
      nv.updateGLVolume();
      return;
    }

    let isCancelled = false;
    if (isHeatmap) {
      setStatusMessage(`Loading predicted probability heat map (${probChannel.toUpperCase()})...`);
    } else if (isCombined) {
      setStatusMessage(`Loading combined probability heat map & boundaries (${probChannel.toUpperCase()})...`);
    } else if (isAgreement) {
      setStatusMessage(`Loading Expert Agreement Map (${agreementRegion})...`);
    } else {
      setStatusMessage('Loading predicted BraTS 2021 segmentation overlay...');
    }

    async function loadMask() {
      try {
        if (isHeatmap) {
          // Heat Map Mode: Load single continuous probability volume
          const probUrl = getProbabilityMapUrl(caseId, probChannel);
          let probData;
          try {
            probData = await fetchNiftiBuffer(probUrl, `probability map (${probChannel})`);
          } catch (fetchErr) {
            if (fetchErr.message.includes('NOT_FOUND')) {
              setStatusMessage('Model probability maps are unavailable for this study. Retaining standard BraTS segmentation overlay.');
              setOverlayMode('prediction');
              return;
            }
            throw fetchErr;
          }

          if (isCancelled) return;

          const probVolume = await NVImage.loadFromUrl({
            url: probData.niftiBuffer,
            name: `prob_${probChannel}.nii`,
            opacity: overlayOpacity,
          });

          if (isCancelled) return;

          updateMaskLut(probVolume, 'heatmap', null, null, overlayOpacity);

          while (nv.volumes && nv.volumes.length > 1) {
            nv.removeVolumeByIndex(1);
          }
          nv.addVolume(probVolume);
          nv.setOpacity(1, overlayOpacity);
          nv.updateGLVolume();

          setStatusMessage(
            `Heat Map Active • Continuous Probability (${probChannel.toUpperCase()} [0.0 - 1.0]) • Blue (Low) -> Cyan -> Green/Yellow -> Red (High)`
          );
        } else if (isCombined) {
          // Combined Mode: Multi-overlay with Probability Heat Map (Vol 1) + Discrete Boundaries (Vol 2)
          const probUrl = getProbabilityMapUrl(caseId, probChannel);
          const boundUrl = getBoundaryMaskUrl(caseId);

          let probData;
          try {
            probData = await fetchNiftiBuffer(probUrl, `probability map (${probChannel})`);
          } catch (fetchErr) {
            if (fetchErr.message.includes('NOT_FOUND')) {
              setStatusMessage('Model probability maps are unavailable for this study. Retaining standard BraTS segmentation overlay.');
              setOverlayMode('prediction');
              return;
            }
            throw fetchErr;
          }

          if (isCancelled) return;

          let boundData = null;
          try {
            boundData = await fetchNiftiBuffer(boundUrl, 'boundary mask');
          } catch (boundErr) {
            console.warn('Boundary mask could not be loaded:', boundErr.message);
          }

          if (isCancelled) return;

          const probVolume = await NVImage.loadFromUrl({
            url: probData.niftiBuffer,
            name: `prob_${probChannel}.nii`,
            opacity: overlayOpacity,
          });

          let boundVolume = null;
          if (boundData) {
            boundVolume = await NVImage.loadFromUrl({
              url: boundData.niftiBuffer,
              name: 'boundary_seg.nii',
              opacity: 1.0,
            });
          }

          if (isCancelled) return;

          updateMaskLut(probVolume, 'heatmap', null, null, overlayOpacity);
          if (boundVolume) {
            updateMaskLut(boundVolume, 'boundary', null, layerVisibility, 1.0);
          }

          while (nv.volumes && nv.volumes.length > 1) {
            nv.removeVolumeByIndex(1);
          }
          nv.addVolume(probVolume);
          nv.setOpacity(1, overlayOpacity);

          if (boundVolume) {
            nv.addVolume(boundVolume);
            nv.setOpacity(2, 1.0);
          }

          nv.updateGLVolume();

          setStatusMessage(
            `Combined Active • Continuous Heat Map (${probChannel.toUpperCase()}) + Delineated Segmentation Boundaries`
          );
        } else {
          // Discrete Segmentation or Expert Agreement Mode
          const targetUrl = isAgreement
            ? getAgreementMaskUrl(caseId, agreementRegion)
            : (maskUrl || (caseId ? getMaskDownloadUrl(caseId) : null));

          if (!targetUrl) return;

          const maskData = await fetchNiftiBuffer(
            targetUrl,
            isAgreement ? `agreement (${agreementRegion})` : 'predicted segmentation'
          );

          if (isCancelled) return;

          const maskVolume = await NVImage.loadFromUrl({
            url: maskData.niftiBuffer,
            name: isAgreement ? `agreement_${agreementRegion}.nii` : 'predicted_seg.nii',
            opacity: overlayOpacity,
          });

          if (isCancelled) return;

          updateMaskLut(maskVolume, overlayMode, agreementRegion, layerVisibility, overlayOpacity);

          while (nv.volumes && nv.volumes.length > 1) {
            nv.removeVolumeByIndex(1);
          }
          nv.addVolume(maskVolume);
          nv.setOpacity(1, overlayOpacity);
          nv.updateGLVolume();

          setStatusMessage(
            isAgreement
              ? `Expert Agreement Active (${agreementRegion}) • 🟢 TP Consensus | 🔴 FP Over-pred | 🟣 FN Missed`
              : 'Segmentation overlay active • 🟡 ET (Enhancing) | 🔵 TC (Core) | 🟢 WT (Edema)'
          );
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('Overlay loading error:', err.message);
          setStatusMessage(`Overlay notice: ${err.message}`);
        }
      }
    }

    loadMask();

    return () => {
      isCancelled = true;
    };
  }, [maskUrl, overlayMode, agreementRegion, probChannel, caseId, hasExpertMask, isViewerReady, baseLoadedCount, caseResults, activeCase]);

  // Synchronize colormap LUT whenever tissue visibility toggles or opacity changes
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !isViewerReady || !nv.volumes || nv.volumes.length < 2) return;

    if (overlayMode === 'combined') {
      const heatVol = nv.volumes[1];
      if (heatVol) {
        updateMaskLut(heatVol, 'heatmap', null, null, overlayOpacity);
        nv.setOpacity(1, overlayOpacity);
      }
      if (nv.volumes.length >= 3) {
        const boundVol = nv.volumes[2];
        updateMaskLut(boundVol, 'boundary', null, layerVisibility, 1.0);
        nv.setOpacity(2, 1.0);
      }
    } else {
      const maskVol = nv.volumes[1];
      updateMaskLut(maskVol, overlayMode, agreementRegion, layerVisibility, overlayOpacity);
      nv.setOpacity(1, overlayOpacity);
    }
    nv.updateGLVolume();
  }, [layerVisibility, overlayOpacity, overlayMode, agreementRegion, probChannel, isViewerReady]);

  // Handle Suggested Tumor Crosshair Positioning
  useEffect(() => {
    if (suggestedCrosshair && Array.isArray(suggestedCrosshair) && suggestedCrosshair.length >= 3) {
      navigateToCentroid(suggestedCrosshair);
    }
  }, [suggestedCrosshair, isViewerReady]);

  // Handle Slice View Switching
  const handleSliceTypeChange = (type) => {
    const nv = nvRef.current;
    if (!nv) return;

    setCurrentSliceType(type);
    switch (type) {
      case 'multi':
        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.setMultiplanarLayout(2); // 4 equal partitions (2x2 grid)
        nv.opts.multiplanarForceRender = true;
        nv.opts.multiplanarPadPixels = 6;
        nv.setCornerOrientationText(true);
        break;
      case 'axial':
        nv.setSliceType(nv.sliceTypeAxial);
        break;
      case 'coronal':
        nv.setSliceType(nv.sliceTypeCoronal);
        break;
      case 'sagittal':
        nv.setSliceType(nv.sliceTypeSagittal);
        break;
      case 'render':
        nv.setSliceType(nv.sliceTypeRender);
        break;
      default:
        nv.setSliceType(nv.sliceTypeMultiplanar);
        nv.setMultiplanarLayout(2);
        nv.opts.multiplanarForceRender = true;
    }
    nv.updateGLVolume();
  };

  // Handle Opacity Slider
  const handleOpacityChange = (e) => {
    const val = parseFloat(e.target.value);
    setOverlayOpacity(val);
    const nv = nvRef.current;
    if (nv && nv.volumes && nv.volumes.length > 1) {
      nv.setOpacity(1, val);
    }
  };

  // Handle Window/Level Presets
  const applyWlPreset = (preset) => {
    setWlPreset(preset);
    const nv = nvRef.current;
    if (!nv || !nv.volumes || nv.volumes.length === 0) return;

    const baseVol = nv.volumes[0];
    const robustMin = baseVol.robust_min || 0;
    const robustMax = baseVol.robust_max || 255;
    const range = Math.max(robustMax - robustMin, 1);

    switch (preset) {
      case 'Brain':
        baseVol.cal_min = robustMin;
        baseVol.cal_max = robustMax * 1.05;
        break;
      case 'Subdural':
        baseVol.cal_min = robustMin + range * 0.05;
        baseVol.cal_max = robustMax * 0.95;
        break;
      case 'Bone':
        baseVol.cal_min = robustMin;
        baseVol.cal_max = robustMax * 1.35;
        break;
      case 'Stroke':
        baseVol.cal_min = robustMin;
        baseVol.cal_max = robustMax * 0.85;
        break;
      default:
        baseVol.cal_min = robustMin;
        baseVol.cal_max = robustMax;
    }
    nv.updateGLVolume();
  };

  // Toggle Tissue Layer Visibility
  const toggleLayer = (layerKey) => {
    if (propOnToggleLayer) {
      propOnToggleLayer(layerKey);
    } else {
      setInternalLayerVisibility((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      {/* Viewer Main Toolbar */}
      <div className="viewer-toolbar">
        {/* Viewport Switcher */}
        <div className="segmented-control mobile-scrollable-segmented">
          <button
            type="button"
            className={`segmented-btn ${currentSliceType === 'multi' ? 'active' : ''}`}
            onClick={() => handleSliceTypeChange('multi')}
            title="Multiplanar 2x2 Grid"
          >
            <span className="desktop-inline">Multiplanar (Axial, Coronal, Sagittal)</span>
            <span className="mobile-inline">Multi</span>
          </button>
          <button
            type="button"
            className={`segmented-btn ${currentSliceType === 'render' ? 'active' : ''}`}
            onClick={() => handleSliceTypeChange('render')}
            title="3D Volume MIP / Surface"
          >
            <span className="desktop-inline">3D Volume</span>
            <span className="mobile-inline">3D</span>
          </button>
          <button
            type="button"
            className={`segmented-btn ${currentSliceType === 'axial' ? 'active' : ''}`}
            onClick={() => handleSliceTypeChange('axial')}
            title="Axial Single Slice"
          >
            <span className="desktop-inline">Axial</span>
            <span className="mobile-inline">Ax</span>
          </button>
          <button
            type="button"
            className={`segmented-btn ${currentSliceType === 'coronal' ? 'active' : ''}`}
            onClick={() => handleSliceTypeChange('coronal')}
            title="Coronal Single Slice"
          >
            <span className="desktop-inline">Coronal</span>
            <span className="mobile-inline">Cor</span>
          </button>
          <button
            type="button"
            className={`segmented-btn ${currentSliceType === 'sagittal' ? 'active' : ''}`}
            onClick={() => handleSliceTypeChange('sagittal')}
            title="Sagittal Single Slice"
          >
            <span className="desktop-inline">Sagittal</span>
            <span className="mobile-inline">Sag</span>
          </button>
        </div>

        {/* Visualization Mode Switcher */}
        <div className="segmented-control mobile-scrollable-segmented">
          <button
            type="button"
            className={`segmented-btn ${overlayMode === 'prediction' ? 'active' : ''}`}
            onClick={() => setOverlayMode('prediction')}
            title="Discrete BraTS segmentation labels (WT, TC, ET)"
          >
            Segmentation
          </button>
          <button
            type="button"
            className={`segmented-btn ${overlayMode === 'heatmap' ? 'active' : ''}`}
            onClick={() => setOverlayMode('heatmap')}
            title="Continuous voxel-wise probability heat map from SegResNet sigmoid outputs"
          >
            Heat Map
          </button>
          <button
            type="button"
            className={`segmented-btn ${overlayMode === 'combined' ? 'active' : ''}`}
            onClick={() => setOverlayMode('combined')}
            title="Combined: continuous probability heat map with discrete segmentation boundaries"
          >
            Combined
          </button>
          <button
            type="button"
            className={`segmented-btn ${overlayMode === 'agreement' ? 'active' : ''}`}
            disabled={!hasExpertMask}
            title={!hasExpertMask ? 'Upload expert ground truth mask to enable consensus overlay' : 'Color-coded TP/FP/FN consensus map'}
            onClick={() => setOverlayMode('agreement')}
            style={{ opacity: hasExpertMask ? 1 : 0.45, cursor: hasExpertMask ? 'pointer' : 'not-allowed' }}
          >
            Expert Agreement
          </button>
        </div>

        {/* Center Toolset: Mask Opacity & Color Chips / Agreement Legend / Heat Map Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {/* Opacity Scrubber */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: '#0F172A', whiteSpace: 'nowrap' }}>
              Opacity: {Math.round(overlayOpacity * 100)}%
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={overlayOpacity}
              onChange={handleOpacityChange}
              style={{ width: '80px', accentColor: 'var(--primary-accent)', cursor: 'pointer' }}
            />
          </div>

          <div style={{ height: '18px', width: '1px', backgroundColor: 'var(--border-subtle)' }} />

          {overlayMode === 'heatmap' ? (
            /* Continuous Probability Controls & Legend */
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              {/* Channel Selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '10px', fontWeight: 600, color: '#475569' }}>Prob:</span>
                {[
                  { id: 'max', label: 'Max' },
                  { id: 'wt', label: 'WT' },
                  { id: 'tc', label: 'TC' },
                  { id: 'et', label: 'ET' },
                ].map((ch) => (
                  <button
                    key={ch.id}
                    type="button"
                    onClick={() => setProbChannel(ch.id)}
                    style={{
                      padding: '2px 7px',
                      fontSize: '10px',
                      fontWeight: probChannel === ch.id ? 700 : 500,
                      backgroundColor: probChannel === ch.id ? 'var(--primary-accent)' : '#FFFFFF',
                      color: probChannel === ch.id ? '#FFFFFF' : '#334155',
                      border: '1px solid #CBD5E1',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                    title={`Predicted voxel-wise sigmoid activation: ${ch.label}`}
                  >
                    {ch.label}
                  </button>
                ))}
              </div>

              {/* Continuous Probability Legend Bar 0.0 to 1.0 */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '150px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '9px', fontWeight: 600, color: '#334155' }}>
                  <span>Tumour Probability</span>
                  <span style={{ fontFamily: 'monospace', color: '#64748B' }}>0.0 — 1.0</span>
                </div>
                <div
                  style={{
                    height: '8px',
                    width: '100%',
                    borderRadius: '2px',
                    background: 'linear-gradient(to right, rgb(0, 20, 150) 0%, rgb(0, 210, 255) 25%, rgb(0, 230, 30) 50%, rgb(255, 215, 0) 75%, rgb(255, 20, 20) 100%)',
                    border: '1px solid rgba(0,0,0,0.15)',
                    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.15)',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8px', color: '#64748B' }}>
                  <span>0.0 (Low)</span>
                  <span>0.5</span>
                  <span>1.0 (High)</span>
                </div>
              </div>
            </div>
          ) : overlayMode === 'combined' ? (
            /* Combined Mode: Continuous Heat Map + Discrete Boundary Toggles */
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              {/* Channel Selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '10px', fontWeight: 600, color: '#475569' }}>Prob:</span>
                {[
                  { id: 'max', label: 'Max' },
                  { id: 'wt', label: 'WT' },
                  { id: 'tc', label: 'TC' },
                  { id: 'et', label: 'ET' },
                ].map((ch) => (
                  <button
                    key={ch.id}
                    type="button"
                    onClick={() => setProbChannel(ch.id)}
                    style={{
                      padding: '2px 7px',
                      fontSize: '10px',
                      fontWeight: probChannel === ch.id ? 700 : 500,
                      backgroundColor: probChannel === ch.id ? 'var(--primary-accent)' : '#FFFFFF',
                      color: probChannel === ch.id ? '#FFFFFF' : '#334155',
                      border: '1px solid #CBD5E1',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                    title={`Continuous probability channel: ${ch.label}`}
                  >
                    {ch.label}
                  </button>
                ))}
              </div>

              {/* Boundary Contour Toggles */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '10px', fontWeight: 600, color: '#475569' }}>Contours:</span>
                <button
                  type="button"
                  className="roi-chip-btn"
                  style={{
                    backgroundColor: 'var(--tissue-wt)',
                    color: '#FFFFFF',
                    borderColor: 'var(--tissue-wt-border)',
                    opacity: layerVisibility.WT ? 1 : 0.4,
                    padding: '2px 6px',
                    fontSize: '10px'
                  }}
                  onClick={() => toggleLayer('WT')}
                  title="Toggle WT boundary contour"
                >
                  <Check size={10} strokeWidth={3} />
                  <span>WT</span>
                </button>
                <button
                  type="button"
                  className="roi-chip-btn"
                  style={{
                    backgroundColor: 'var(--tissue-tc)',
                    color: '#FFFFFF',
                    borderColor: 'var(--tissue-tc-border)',
                    opacity: layerVisibility.TC ? 1 : 0.4,
                    padding: '2px 6px',
                    fontSize: '10px'
                  }}
                  onClick={() => toggleLayer('TC')}
                  title="Toggle TC boundary contour"
                >
                  <Check size={10} strokeWidth={3} />
                  <span>TC</span>
                </button>
                <button
                  type="button"
                  className="roi-chip-btn"
                  style={{
                    backgroundColor: 'var(--tissue-et)',
                    color: '#0F172A',
                    borderColor: 'var(--tissue-et-border)',
                    opacity: layerVisibility.ET ? 1 : 0.4,
                    padding: '2px 6px',
                    fontSize: '10px'
                  }}
                  onClick={() => toggleLayer('ET')}
                  title="Toggle ET boundary contour"
                >
                  <Check size={10} strokeWidth={3} />
                  <span>ET</span>
                </button>
              </div>

              {/* Mini Continuous Probability Legend */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '120px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '9px', fontWeight: 600, color: '#334155' }}>
                  <span>Probability</span>
                  <span style={{ fontFamily: 'monospace', color: '#64748B' }}>0.0 — 1.0</span>
                </div>
                <div
                  style={{
                    height: '7px',
                    width: '100%',
                    borderRadius: '2px',
                    background: 'linear-gradient(to right, rgb(0, 20, 150) 0%, rgb(0, 210, 255) 25%, rgb(0, 230, 30) 50%, rgb(255, 215, 0) 75%, rgb(255, 20, 20) 100%)',
                    border: '1px solid rgba(0,0,0,0.15)',
                  }}
                />
              </div>
            </div>
          ) : overlayMode === 'agreement' ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '10px', fontWeight: 600, color: '#475569' }}>Region:</span>
              {['WT', 'TC', 'ET'].map((reg) => (
                <button
                  key={reg}
                  type="button"
                  onClick={() => setAgreementRegion(reg)}
                  style={{
                    padding: '2px 7px',
                    fontSize: '10px',
                    fontWeight: agreementRegion === reg ? 700 : 500,
                    backgroundColor: agreementRegion === reg ? 'var(--primary-accent)' : '#FFFFFF',
                    color: agreementRegion === reg ? '#FFFFFF' : '#334155',
                    border: '1px solid #CBD5E1',
                    borderRadius: '3px',
                    cursor: 'pointer',
                  }}
                >
                  {reg}
                </button>
              ))}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '10px', marginLeft: '4px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#065F46', fontWeight: 600 }}>
                  <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#10B981', display: 'inline-block' }} />
                  TP
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#991B1B', fontWeight: 600 }}>
                  <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#EF4444', display: 'inline-block' }} />
                  FP
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px', color: '#5B21B6', fontWeight: 600 }}>
                  <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#8B5CF6', display: 'inline-block' }} />
                  FN
                </span>
              </div>
            </div>
          ) : (
            /* Standard BraTS Layer Toggles */
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <button
                type="button"
                className="roi-chip-btn"
                style={{
                  backgroundColor: 'var(--tissue-wt)',
                  color: '#FFFFFF',
                  borderColor: 'var(--tissue-wt-border)',
                  opacity: layerVisibility.WT ? 1 : 0.5
                }}
                onClick={() => toggleLayer('WT')}
              >
                <Check size={11} strokeWidth={3} />
                <span>WT</span>
              </button>

              <button
                type="button"
                className="roi-chip-btn"
                style={{
                  backgroundColor: 'var(--tissue-tc)',
                  color: '#FFFFFF',
                  borderColor: 'var(--tissue-tc-border)',
                  opacity: layerVisibility.TC ? 1 : 0.5
                }}
                onClick={() => toggleLayer('TC')}
              >
                <Check size={11} strokeWidth={3} />
                <span>TC</span>
              </button>

              <button
                type="button"
                className="roi-chip-btn"
                style={{
                  backgroundColor: 'var(--tissue-et)',
                  color: '#0F172A',
                  borderColor: 'var(--tissue-et-border)',
                  opacity: layerVisibility.ET ? 1 : 0.5
                }}
                onClick={() => toggleLayer('ET')}
              >
                <Check size={11} strokeWidth={3} />
                <span>ET</span>
              </button>
            </div>
          )}
        </div>

        {/* Right Toolset: Presets & Live Coordinate Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {/* Centroid Jump Button */}
          <button
            type="button"
            className="segmented-btn"
            disabled={!suggestedCrosshair || !Array.isArray(suggestedCrosshair) || suggestedCrosshair.length < 3}
            title={
              suggestedCrosshair && Array.isArray(suggestedCrosshair) && suggestedCrosshair.length >= 3
                ? `Jump to Tumour Centroid [${suggestedCrosshair.join(', ')}]`
                : 'Tumour centroid not available (empty mask)'
            }
            onClick={() => navigateToCentroid(suggestedCrosshair)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 8px',
              fontSize: '11px',
              fontWeight: 600,
              backgroundColor: suggestedCrosshair ? '#F1F5F9' : '#F8FAFC',
              color: suggestedCrosshair ? 'var(--primary-accent)' : '#94A3B8',
              border: '1px solid var(--border-subtle)',
              borderRadius: '3px',
              cursor: suggestedCrosshair ? 'pointer' : 'not-allowed',
              opacity: suggestedCrosshair ? 1 : 0.6,
            }}
          >
            <Target size={12} strokeWidth={2.5} />
            <span>Centroid</span>
          </button>

          {/* Reset Camera / Multiplanar / Orientation Action */}
          <button
            type="button"
            className="segmented-btn"
            title="Reset 2D multiplanar crosshair, 3D camera, window/level, and opacity to default"
            onClick={handleResetView}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 8px',
              fontSize: '11px',
              fontWeight: 600,
              backgroundColor: '#F1F5F9',
              color: '#334155',
              border: '1px solid var(--border-subtle)',
              borderRadius: '3px',
              cursor: 'pointer',
            }}
          >
            <RotateCcw size={12} strokeWidth={2.5} />
            <span>Reset View</span>
          </button>

          {/* Window/Level Presets */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              border: '1px solid var(--border-subtle)',
              backgroundColor: '#FFFFFF',
              borderRadius: '2px',
              overflow: 'hidden'
            }}
          >
            <span
              className="font-mono"
              style={{
                padding: '2px 6px',
                backgroundColor: '#F1F5F9',
                fontSize: '10px',
                color: 'var(--on-surface-variant)'
              }}
            >
              W/L:
            </span>
            {['Brain', 'Subdural', 'Bone', 'Stroke'].map((preset) => (
              <button
                key={preset}
                type="button"
                style={{
                  padding: '2px 6px',
                  border: 'none',
                  fontSize: '10px',
                  fontWeight: wlPreset === preset ? 600 : 400,
                  backgroundColor: wlPreset === preset ? 'var(--primary-accent)' : 'transparent',
                  color: wlPreset === preset ? '#FFFFFF' : 'var(--on-surface-dark)',
                  cursor: 'pointer'
                }}
                onClick={() => applyWlPreset(preset)}
              >
                {preset}
              </button>
            ))}
          </div>

          {/* Live Coordinate Pill Badge */}
          <div className="coordinate-pill-badge" id="workstation-coordinates-badge">
            {crosshairPos?.vox && crosshairPos?.mm
              ? `Voxel: [${crosshairPos.vox.join(', ')}] | World: [${crosshairPos.mm.map((m) => `${m.toFixed(1)}mm`).join(', ')}]`
              : 'Voxel: [--, --, --] | World: [--, --, --]'}
          </div>
        </div>
      </div>

      {/* Viewport Canvas & HUD */}
      <div className="viewer-canvas-container">
        {/* Top HUD Header (for Single View Modes) */}
        {currentSliceType !== 'multi' && (
          <div className="viewport-hud-header">
            <span style={{ color: 'var(--tissue-wt)', fontWeight: 700 }}>
              {currentSliceType === 'render' ? '3D VOLUME' : currentSliceType.toUpperCase()}
            </span>
            <span style={{ color: '#94A3B8' }}>|</span>
            <span>
              Z: {crosshairPos?.vox?.[2] !== undefined ? crosshairPos.vox[2] : '--'}
              {volumeDims?.[2] ? ` / ${volumeDims[2]}` : ''}
            </span>
            <span style={{ color: '#94A3B8' }}>|</span>
            <span style={{ color: 'var(--tissue-et)' }}>
              {activeSequenceLabel}{maskUrl ? ' + MONAI Seg' : ''}
            </span>
          </div>
        )}

        {/* 4-Quadrant Partition Grid Overlay (when in Multiplanar mode) */}
        {currentSliceType === 'multi' && (
          <>
            {/* Vertical Partition Line */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: '50%',
                width: '1px',
                backgroundColor: 'rgba(51, 65, 85, 0.85)',
                zIndex: 10,
                pointerEvents: 'none'
              }}
            />
            {/* Horizontal Partition Line */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                right: 0,
                top: '50%',
                height: '1px',
                backgroundColor: 'rgba(51, 65, 85, 0.85)',
                zIndex: 10,
                pointerEvents: 'none'
              }}
            />
            {/* Center Intersection Reticle */}
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '14px',
                height: '14px',
                borderRadius: '50%',
                border: '1px solid #38BDF8',
                backgroundColor: 'rgba(11, 15, 25, 0.95)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 12,
                pointerEvents: 'none'
              }}
            >
              <div style={{ width: '4px', height: '4px', borderRadius: '50%', backgroundColor: '#38BDF8' }} />
            </div>

            {/* Partition 1 (Top-Left): CORONAL */}
            <div
              style={{
                position: 'absolute',
                top: '8px',
                left: '8px',
                zIndex: 15,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid #334155',
                borderRadius: '3px',
                padding: '2px 8px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                color: '#38BDF8',
                pointerEvents: 'none'
              }}
            >
              <span style={{ fontWeight: 700 }}>CORONAL</span>
              <span style={{ color: '#64748B' }}>|</span>
              <span style={{ color: '#E2E8F0' }}>Y: {crosshairPos?.vox?.[1] !== undefined ? crosshairPos.vox[1] : '--'}{volumeDims?.[1] ? ` / ${volumeDims[1]}` : ''}</span>
            </div>

            {/* Partition 2 (Top-Right): SAGITTAL */}
            <div
              style={{
                position: 'absolute',
                top: '8px',
                right: '8px',
                zIndex: 15,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid #334155',
                borderRadius: '3px',
                padding: '2px 8px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                color: '#38BDF8',
                pointerEvents: 'none'
              }}
            >
              <span style={{ fontWeight: 700 }}>SAGITTAL</span>
              <span style={{ color: '#64748B' }}>|</span>
              <span style={{ color: '#E2E8F0' }}>X: {crosshairPos?.vox?.[0] !== undefined ? crosshairPos.vox[0] : '--'}{volumeDims?.[0] ? ` / ${volumeDims[0]}` : ''}</span>
            </div>

            {/* Partition 3 (Bottom-Left): AXIAL */}
            <div
              style={{
                position: 'absolute',
                top: 'calc(50% + 8px)',
                left: '8px',
                zIndex: 15,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid #334155',
                borderRadius: '3px',
                padding: '2px 8px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                color: '#38BDF8',
                pointerEvents: 'none'
              }}
            >
              <span style={{ fontWeight: 700 }}>AXIAL</span>
              <span style={{ color: '#64748B' }}>|</span>
              <span style={{ color: '#E2E8F0' }}>Z: {crosshairPos?.vox?.[2] !== undefined ? crosshairPos.vox[2] : '--'}{volumeDims?.[2] ? ` / ${volumeDims[2]}` : ''}</span>
            </div>

            {/* Partition 4 (Bottom-Right): 3D VOLUME RENDER */}
            <div
              style={{
                position: 'absolute',
                top: 'calc(50% + 8px)',
                right: '8px',
                zIndex: 15,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid #334155',
                borderRadius: '3px',
                padding: '2px 8px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                color: '#F97316',
                pointerEvents: 'none'
              }}
            >
              <span style={{ fontWeight: 700 }}>3D VOLUME RENDER</span>
              <span style={{ color: '#64748B' }}>|</span>
              <span style={{ color: '#CBD5E1' }}>MIP / Surface</span>
            </div>

            {/* Center Sequence Info Pill */}
            <div
              style={{
                position: 'absolute',
                bottom: '8px',
                left: '50%',
                transform: 'translateX(-50%)',
                zIndex: 15,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                backgroundColor: 'rgba(15, 23, 42, 0.92)',
                border: '1px solid #334155',
                borderRadius: '4px',
                padding: '3px 12px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                color: 'var(--tissue-et)',
                pointerEvents: 'none'
              }}
            >
              <span>{activeSequenceLabel}{maskUrl ? ' + MONAI Seg' : ''}</span>
            </div>
          </>
        )}

        {/* Anatomical Orientation Badges (visible in full Axial 2D mode only) */}
        {currentSliceType === 'axial' && (
          <>
            <span className="anatomical-badge" style={{ top: '8px', left: '50%', transform: 'translateX(-50%)' }}>
              A
            </span>
            <span className="anatomical-badge" style={{ bottom: '8px', left: '50%', transform: 'translateX(-50%)' }}>
              P
            </span>
            <span className="anatomical-badge" style={{ left: '8px', top: '50%', transform: 'translateY(-50%)' }}>
              R
            </span>
            <span className="anatomical-badge" style={{ right: '8px', top: '50%', transform: 'translateY(-50%)' }}>
              L
            </span>
          </>
        )}

        {/* Actual WebGL Canvas attached to NiiVue with touch-action: none for zero mobile scroll jitter */}
        <canvas ref={canvasRef} className="niivue-canvas-el" style={{ touchAction: 'none' }} />
      </div>
    </div>
  );
}
