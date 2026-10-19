import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

// Mock NiiVueViewer to avoid WebGL context requirements in JSDOM
vi.mock('../components/NiiVueViewer', () => ({
  default: () => <div data-testid="mock-niivue-viewer">NiiVue Mock</div>,
}));

// Mock api services
vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', monai_loaded: true }),
    uploadStudy: vi.fn(),
    triggerSegmentation: vi.fn().mockResolvedValue({ status: 'completed' }),
    getCaseResults: vi.fn().mockResolvedValue({
      case_id: 'case_tested_456',
      status: 'completed',
      regions: { WT: { volume_ml: 15.0 }, TC: { volume_ml: 7.5 }, ET: { volume_ml: 3.2 } },
      spatial_metadata: { shape: [240, 240, 155], spacing_mm: [1.0, 1.0, 1.0] },
    }),
    loadSampleStudy: vi.fn().mockResolvedValue({
      case_id: 'case_sample_123',
      validation_status: 'passed',
      files: {
        t1: { filename: 'sample_t1.nii.gz', size_bytes: 1048576 },
        t1gd: { filename: 'sample_t1ce.nii.gz', size_bytes: 1048576 },
        t2: { filename: 'sample_t2.nii.gz', size_bytes: 1048576 },
        flair: { filename: 'sample_flair.nii.gz', size_bytes: 1048576 },
      },
      spatial_summary: { shape: [240, 240, 155], spacing_mm: [1.0, 1.0, 1.0] },
    }),
    getBoundaryWatch: vi.fn().mockResolvedValue({
      case_id: 'case_tested_456',
      risk_voxels: 100,
      expansion_risk_score: 0.1,
    }),
    uploadDicomStudy: vi.fn(),
    calculateCaseVolumes: vi.fn().mockResolvedValue({ status: 'calculated' }),
    getSequenceDownloadUrl: vi.fn((caseId, seq) => `/api/v1/cases/${caseId}/sequence/${seq}`),
    getMaskDownloadUrl: vi.fn((caseId) => `/api/v1/cases/${caseId}/mask`),
    getAgreementMaskUrl: vi.fn(() => '/api/v1/cases/test/agreement'),
  };
});

import App from '../App';
import * as api from '../services/api';

describe('MRI Workstation Tab Switching State Persistence', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.location.hash = '#workstation';
  });

  const createDummyFile = (name, size = 1024 * 1024) => {
    return new File(['mock content '.repeat(100)], name, { type: 'application/gzip' });
  };

  it('verifies all 12 regression requirements for workstation tab persistence', async () => {
    const { container } = render(<App />);

    // 1. Initial State: Switch to Study Ingestion panel
    const studyIngestionTabBtn = screen.getByRole('button', { name: /Study Ingestion/i });
    const quantLayersTabBtn = screen.getByRole('button', { name: /Quantitation & Layers/i });

    fireEvent.click(studyIngestionTabBtn);

    // Initial panel displays 0/4 PENDING
    expect(screen.getByText(/0\/4/)).toBeInTheDocument();
    expect(screen.getByText(/PENDING/)).toBeInTheDocument();

    // Create 4 BraTS MRI sequence files
    const fileT1 = createDummyFile('BraTS2021_00001_t1.nii.gz', 1048576);
    const fileT1ce = createDummyFile('BraTS2021_00001_t1ce.nii.gz', 1048576);
    const fileT2 = createDummyFile('BraTS2021_00001_t2.nii.gz', 1048576);
    const fileFlair = createDummyFile('BraTS2021_00001_flair.nii.gz', 1048576);

    // 1 & 2. Upload / auto-assign all four MRI sequences via batch dropzone
    const bulkInput = container.querySelector('#bulk-folder-input');
    expect(bulkInput).toBeTruthy();

    fireEvent.change(bulkInput, {
      target: { files: [fileT1, fileT1ce, fileT2, fileFlair] }
    });

    // 2. Confirm the panel displays 4/4 assigned (READY)
    await waitFor(() => {
      expect(screen.getByText(/4\/4/)).toBeInTheDocument();
      expect(screen.getByText(/READY/)).toBeInTheDocument();
    });

    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1ce.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t2.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_flair.nii.gz')).toBeInTheDocument();

    // 3. Switch to Quantitation & Layers
    fireEvent.click(quantLayersTabBtn);
    expect(quantLayersTabBtn).toHaveClass('active');

    // 4. Return to Study Ingestion
    fireEvent.click(studyIngestionTabBtn);
    expect(studyIngestionTabBtn).toHaveClass('active');

    // 5. Confirm all four assignments and filenames remain unchanged
    expect(screen.getByText(/4\/4/)).toBeInTheDocument();
    expect(screen.getByText(/READY/)).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1ce.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t2.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_flair.nii.gz')).toBeInTheDocument();

    // 6. Switch tabs repeatedly (10 times) and confirm assignments persist
    for (let i = 0; i < 5; i++) {
      fireEvent.click(quantLayersTabBtn);
      fireEvent.click(studyIngestionTabBtn);
    }
    expect(screen.getByText(/4\/4/)).toBeInTheDocument();
    expect(screen.getByText(/READY/)).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_flair.nii.gz')).toBeInTheDocument();

    // 7. Confirm that replacing one sequence updates only that sequence
    const replaceInputs = container.querySelectorAll('.sequence-slot-card input[type="file"]');
    // FLAIR is index 3 (T1=0, T1Gd=1, T2=2, FLAIR=3)
    const newFlairFile = createDummyFile('BraTS2021_00001_flair_v2.nii.gz', 2097152);
    fireEvent.change(replaceInputs[3], { target: { files: [newFlairFile] } });

    expect(screen.getByText('BraTS2021_00001_flair_v2.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1ce.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t2.nii.gz')).toBeInTheDocument();
    expect(screen.queryByText('BraTS2021_00001_flair.nii.gz')).not.toBeInTheDocument();
    expect(screen.getByText(/4\/4/)).toBeInTheDocument();

    // 8. Confirm that clearing a sequence explicitly clears only that sequence
    const removeButtons = container.querySelectorAll('.remove-seq-btn');
    // Click remove on FLAIR (the 4th slot with a remove button)
    fireEvent.click(removeButtons[3]);

    expect(screen.getByText(/3\/4/)).toBeInTheDocument();
    expect(screen.getByText(/PENDING/)).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1ce.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t2.nii.gz')).toBeInTheDocument();
    // FLAIR is cleared + Expert Mask is not assigned = 2 "Not assigned" slots
    expect(screen.getAllByText('Not assigned').length).toBe(2);

    // Re-assign FLAIR to restore 4/4
    fireEvent.change(replaceInputs[3], { target: { files: [fileFlair] } });
    expect(screen.getByText(/4\/4/)).toBeInTheDocument();
    expect(screen.getByText(/READY/)).toBeInTheDocument();

    // 9. Confirm that the Run SegResNet button receives the correct four browser File objects
    api.uploadStudy.mockResolvedValueOnce({
      case_id: 'case_tested_456',
      validation_status: 'passed',
      files: {
        t1: { filename: fileT1.name, size_bytes: fileT1.size },
        t1gd: { filename: fileT1ce.name, size_bytes: fileT1ce.size },
        t2: { filename: fileT2.name, size_bytes: fileT2.size },
        flair: { filename: fileFlair.name, size_bytes: fileFlair.size },
      },
      spatial_summary: { shape: [240, 240, 155], spacing_mm: [1.0, 1.0, 1.0] },
    });

    const runSegBtn = screen.getByRole('button', { name: /Run SegResNet/i });
    fireEvent.click(runSegBtn);

    await waitFor(() => {
      expect(api.uploadStudy).toHaveBeenCalledTimes(1);
    });

    // Inspect the FormData sent to uploadStudy
    const uploadedFormData = api.uploadStudy.mock.calls[0][0];
    expect(uploadedFormData instanceof FormData).toBe(true);
    expect(uploadedFormData.get('t1').name).toBe('BraTS2021_00001_t1.nii.gz');
    expect(uploadedFormData.get('t1gd').name).toBe('BraTS2021_00001_t1ce.nii.gz');
    expect(uploadedFormData.get('t2').name).toBe('BraTS2021_00001_t2.nii.gz');
    expect(uploadedFormData.get('flair').name).toBe('BraTS2021_00001_flair.nii.gz');

    // 10. Confirm that a successful backend upload retains the correct case ID and status
    await waitFor(() => {
      expect(screen.getByText(/4\/4 ✓ VALIDATED/)).toBeInTheDocument();
    });

    // Switch tabs to quant and back
    fireEvent.click(quantLayersTabBtn);
    fireEvent.click(studyIngestionTabBtn);

    // Retains validated state and filenames
    expect(screen.getByText(/4\/4 ✓ VALIDATED/)).toBeInTheDocument();
    expect(screen.getByText('BraTS2021_00001_t1.nii.gz')).toBeInTheDocument();

    // 12. Confirm that tab switching does not trigger duplicate API requests
    const uploadCallCount = api.uploadStudy.mock.calls.length;
    const triggerSegCallCount = api.triggerSegmentation.mock.calls.length;

    fireEvent.click(quantLayersTabBtn);
    fireEvent.click(studyIngestionTabBtn);
    fireEvent.click(quantLayersTabBtn);
    fireEvent.click(studyIngestionTabBtn);

    expect(api.uploadStudy).toHaveBeenCalledTimes(uploadCallCount);
    expect(api.triggerSegmentation).toHaveBeenCalledTimes(triggerSegCallCount);

    // 11. Confirm that Reset Study intentionally clears state according to existing reset behavior
    const resetStudyBtn = screen.getByRole('button', { name: /Reset current study/i });
    fireEvent.click(resetStudyBtn);

    // After reset, panel resets to 0/4 PENDING
    await waitFor(() => {
      expect(screen.getByText(/0\/4 PENDING/)).toBeInTheDocument();
    });

    // All slots show "Not assigned"
    const notAssignedSpans = screen.getAllByText('Not assigned');
    expect(notAssignedSpans.length).toBeGreaterThanOrEqual(4);
  });

  it('persists selected files and active case when navigating across routes (#workstation -> #analytics -> #audit -> #workstation)', async () => {
    const { container } = render(<App />);

    const studyIngestionTabBtn = screen.getByRole('button', { name: /Study Ingestion/i });
    fireEvent.click(studyIngestionTabBtn);

    const fileT1 = createDummyFile('patient_test_t1.nii.gz');
    const fileT1ce = createDummyFile('patient_test_t1ce.nii.gz');
    const fileT2 = createDummyFile('patient_test_t2.nii.gz');
    const fileFlair = createDummyFile('patient_test_flair.nii.gz');

    const bulkInput = container.querySelector('#bulk-folder-input');
    fireEvent.change(bulkInput, {
      target: { files: [fileT1, fileT1ce, fileT2, fileFlair] }
    });

    await waitFor(() => {
      expect(screen.getByText('patient_test_t1.nii.gz')).toBeInTheDocument();
      expect(screen.getByText(/4\/4/)).toBeInTheDocument();
    });

    // 1. Navigate to Volumetric Analytics via nav rail
    const analyticsRailBtn = screen.getByRole('button', { name: /Volumetric Analytics/i });
    fireEvent.click(analyticsRailBtn);

    // 2. Navigate to Provenance & Reports via nav rail
    const provenanceRailBtn = screen.getByRole('button', { name: /Provenance & Reports/i });
    fireEvent.click(provenanceRailBtn);

    // 3. Navigate back to MRI Workstation
    const workstationRailBtn = screen.getByRole('button', { name: /MRI Workstation/i });
    fireEvent.click(workstationRailBtn);

    // 4. Return to Study Ingestion panel
    const studyIngestionBtnAgain = screen.getByRole('button', { name: /Study Ingestion/i });
    fireEvent.click(studyIngestionBtnAgain);

    // Verify all 4 sequence assignments, filenames, and ready status are still retained!
    expect(screen.getByText('patient_test_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('patient_test_t1ce.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('patient_test_t2.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('patient_test_flair.nii.gz')).toBeInTheDocument();
    expect(screen.getByText(/4\/4/)).toBeInTheDocument();
    expect(screen.getByText(/READY/)).toBeInTheDocument();
  });

  it('preserves validated sample case status and metadata across tab switches', async () => {
    render(<App />);

    const studyIngestionTabBtn = screen.getByRole('button', { name: /Study Ingestion/i });
    const quantLayersTabBtn = screen.getByRole('button', { name: /Quantitation & Layers/i });
    fireEvent.click(studyIngestionTabBtn);

    const loadSampleBtn = screen.getByRole('button', { name: /Load Sample BraTS Study/i });
    fireEvent.click(loadSampleBtn);

    await waitFor(() => {
      expect(api.loadSampleStudy).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByText(/4\/4 ✓ VALIDATED/)).toBeInTheDocument();
    });

    expect(screen.getByText('sample_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('sample_t1ce.nii.gz')).toBeInTheDocument();

    // Switch to Quantitation & Layers and back to Study Ingestion
    fireEvent.click(quantLayersTabBtn);
    fireEvent.click(studyIngestionTabBtn);

    // Should remain 4/4 VALIDATED with server filenames
    expect(screen.getByText(/4\/4 ✓ VALIDATED/)).toBeInTheDocument();
    expect(screen.getByText('sample_t1.nii.gz')).toBeInTheDocument();
    expect(screen.getByText('sample_flair.nii.gz')).toBeInTheDocument();
  });
});
