import React, { useState, useEffect } from 'react';
import NavRail from './components/NavRail';
import Header from './components/Header';
import SeriesExplorer from './components/SeriesExplorer';
import WorkspaceContextStrip from './components/WorkspaceContextStrip';
import Dashboard from './pages/Dashboard';
import DashboardHome from './pages/DashboardHome';
import VolumetricAnalyticsView from './pages/VolumetricAnalyticsView';
import LongitudinalView from './pages/LongitudinalView';
import CaseRegistryView from './pages/CaseRegistryView';
import UploadAnalysisView from './pages/UploadAnalysisView';
import ProvenanceAuditReport from './components/ProvenanceAuditReport';
import CaseValidationDiagnostics from './components/CaseValidationDiagnostics';
import CohortBenchmarkView from './components/CohortBenchmarkView';
import UncertaintyAndLongitudinalView from './components/UncertaintyAndLongitudinalView';
import {
  getHealth,
  getMaskDownloadUrl,
  getSequenceDownloadUrl,
  loadSampleStudy,
  triggerSegmentation,
  getCaseResults
} from './services/api';

export default function App() {
  const getInitialTab = () => {
    if (typeof window !== 'undefined' && window.location.hash) {
      const h = window.location.hash.replace('#', '').toLowerCase();
      const valid = ['dashboard', 'workstation', 'upload', 'analytics', 'longitudinal', 'trust', 'cohort', 'cases', 'audit', 'diagnostics'];
      if (valid.includes(h)) return h;
    }
    return 'workstation';
  };

  const [backendHealth, setBackendHealth] = useState(null);
  const [activeTab, setActiveTabState] = useState(getInitialTab);
  const [isNavCollapsed, setIsNavCollapsed] = useState(false);
  const [workstationSubTab, setWorkstationSubTab] = useState('viewer');

  const setActiveTab = (tab) => {
    setActiveTabState(tab);
    if (typeof window !== 'undefined') {
      window.location.hash = tab;
    }
  };

  useEffect(() => {
    const handleHash = () => {
      const h = window.location.hash.replace('#', '').toLowerCase();
      const valid = ['dashboard', 'workstation', 'upload', 'analytics', 'longitudinal', 'trust', 'cohort', 'cases', 'audit', 'diagnostics'];
      if (valid.includes(h)) setActiveTabState(h);
    };
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  const [activeCase, setActiveCase] = useState(null);
  const [caseResults, setCaseResults] = useState(null);
  const [baseScanFile, setBaseScanFile] = useState(null);
  const [baseScanUrl, setBaseScanUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [activeSequenceKey, setActiveSequenceKey] = useState('t1gd');
  const [validationError, setValidationError] = useState(null);
  const [isGlobalLoading, setIsGlobalLoading] = useState(false);

  // Ingestion state persistent across workstation session
  const [ingestionFiles, setIngestionFiles] = useState({
    t1: null,
    t1gd: null,
    t2: null,
    flair: null,
    expert_mask: null,
  });
  const [uploadResult, setUploadResult] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    getHealth()
      .then((data) => setBackendHealth(data))
      .catch((err) => {
        console.warn('Backend currently offline or starting:', err.message);
      });
  }, []);

  // Suppress transitions on theme / scheme switch (better-ui)
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = () => {
      const style = document.createElement('style');
      style.append(
        document.createTextNode(
          '*,*::before,*::after{transition:none !important}'
        )
      );
      document.head.append(style);

      // Force synchronous style flush
      void document.body.offsetHeight;

      requestAnimationFrame(() => {
        requestAnimationFrame(() => style.remove());
      });
    };

    mql.addEventListener('change', handleChange);
    return () => mql.removeEventListener('change', handleChange);
  }, []);

  const handleCaseUploaded = (uploadData) => {
    setActiveCase(uploadData);
    setUploadResult(uploadData);
    setValidationError(null);

    // Clear local file so server-synced sequence URL is loaded
    setBaseScanFile(null);
    if (uploadData?.case_id) {
      setBaseScanUrl(getSequenceDownloadUrl(uploadData.case_id, 't1gd'));
      setActiveSequenceKey('t1gd');
    }

    // If upload response already includes precomputed or expert volumes, populate them safely
    const hasExpertOrMask = uploadData?.has_mask || uploadData?.files?.expert_mask?.present || false;
    if (uploadData?.volumes && (hasExpertOrMask || uploadData?.case_id)) {
      setCaseResults({
        case_id: uploadData.case_id,
        status: hasExpertOrMask ? 'completed' : (uploadData.status || 'uploaded'),
        regions: uploadData.volumes,
        spatial_metadata: uploadData.spatial_summary || uploadData.spatial_metadata,
      });
      setMaskUrl(getMaskDownloadUrl(uploadData.case_id));
    } else if (uploadData?.volumes) {
      setCaseResults({
        case_id: uploadData.case_id,
        status: uploadData.status || 'uploaded',
        regions: uploadData.volumes,
        spatial_metadata: uploadData.spatial_summary || uploadData.spatial_metadata,
      });
      setMaskUrl(getMaskDownloadUrl(uploadData.case_id));
    } else {
      setCaseResults(null);
      setMaskUrl(null);
    }
  };

  const handleSegmentationCompleted = (resultsData) => {
    setCaseResults(resultsData);
    if (resultsData.case_id) {
      setMaskUrl(getMaskDownloadUrl(resultsData.case_id));
    }
  };

  const handleBaseScanSelected = (file) => {
    setBaseScanFile(file);
    setBaseScanUrl(null);
  };

  const handleValidationError = (err) => {
    setValidationError(err);
    setActiveTab('diagnostics');
  };

  const handleSelectSequence = (seqKey) => {
    setActiveSequenceKey(seqKey);
    const caseIdToUse = activeCase?.case_id || caseResults?.case_id;
    if (caseIdToUse) {
      setBaseScanFile(null);
      setBaseScanUrl(getSequenceDownloadUrl(caseIdToUse, seqKey));
    }
  };

  const handleResetStudy = () => {
    setActiveCase(null);
    setCaseResults(null);
    setBaseScanFile(null);
    setBaseScanUrl(null);
    setMaskUrl(null);
    setValidationError(null);
    setIngestionFiles({
      t1: null,
      t1gd: null,
      t2: null,
      flair: null,
      expert_mask: null,
    });
    setUploadResult(null);
    setActiveTab('workstation');
  };

  const handleLoadSampleDirect = async () => {
    setIsGlobalLoading(true);
    try {
      const resp = await loadSampleStudy();
      handleCaseUploaded(resp);
      setUploadResult(resp);
      await triggerSegmentation(resp.case_id);
      const results = await getCaseResults(resp.case_id);
      handleSegmentationCompleted(results);
      setActiveTab('workstation');
    } catch (err) {
      console.error('Failed to load sample study:', err);
      throw err;
    } finally {
      setIsGlobalLoading(false);
    }
  };

  const handleSelectPersistedCase = (persistedCase) => {
    if (!persistedCase) return;
    setActiveCase(persistedCase);
    setUploadResult(persistedCase);
    if (persistedCase.case_id) {
      setBaseScanFile(null);
      setBaseScanUrl(getSequenceDownloadUrl(persistedCase.case_id, 't1gd'));
      setActiveSequenceKey('t1gd');
      if (persistedCase.status === 'completed' || persistedCase.has_mask) {
        setMaskUrl(getMaskDownloadUrl(persistedCase.case_id));
        setCaseResults({
          case_id: persistedCase.case_id,
          status: 'completed',
          regions: persistedCase.regions || persistedCase.volumes,
          spatial_metadata: persistedCase.spatial_metadata || persistedCase.spatial_summary,
        });
      }
    }
  };

  return (
    <div className={`app-shell ${isNavCollapsed ? 'nav-collapsed' : 'nav-expanded'}`}>
      {/* 1. Global Left Navigation Rail */}
      <NavRail
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isCollapsed={isNavCollapsed}
        onToggleCollapse={() => setIsNavCollapsed((prev) => !prev)}
        activeCase={activeCase}
        caseResults={caseResults}
        backendHealth={backendHealth}
        hasValidationErrors={!!validationError}
      />

      {/* 2. Main Application Area: Header + Dynamic Canvas */}
      <div className="app-shell-body">
        {/* Compact Clinical Top Header */}
        <Header
          backendHealth={backendHealth}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onResetStudy={handleResetStudy}
          hasValidationErrors={!!validationError}
        />

        {/* Dynamic Main Content Canvas */}
        <div className="shell-main-canvas">
          {/* View 1: Bento-Grid Dashboard */}
          {activeTab === 'dashboard' && (
            <DashboardHome
              activeCase={activeCase}
              caseResults={caseResults}
              backendHealth={backendHealth}
              onNavigateTab={setActiveTab}
              onLoadSample={handleLoadSampleDirect}
              onCaseSelected={handleSelectPersistedCase}
              isProcessing={isGlobalLoading}
            />
          )}

          {/* View 2: Viewer-First MRI Workstation */}
          {activeTab === 'workstation' && (
            <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
              <WorkspaceContextStrip
                caseId={caseResults?.case_id || activeCase?.case_id}
                spatialMetadata={caseResults?.spatial_metadata || activeCase?.spatial_summary}
                backendHealth={backendHealth}
              />

              <div className={`app-layout-container workstation-layout mobile-subtab-${workstationSubTab}`}>
                <div className={`series-sidebar-wrapper ${workstationSubTab === 'series' ? 'mobile-active' : 'mobile-hidden'}`}>
                  <SeriesExplorer
                    activeSequenceKey={activeSequenceKey}
                    onSelectSequence={(seq) => {
                      handleSelectSequence(seq);
                      if (window.innerWidth < 1024) {
                        setWorkstationSubTab('viewer');
                      }
                    }}
                    volumes={caseResults?.regions || activeCase?.volumes || activeCase?.regions}
                    caseData={activeCase}
                    runtimeSeconds={caseResults?.runtime_seconds}
                  />
                </div>

                <div className={`workspace-content ${workstationSubTab === 'series' ? 'mobile-hidden' : 'mobile-active'}`}>
                  <Dashboard
                    activeCase={activeCase}
                    caseResults={caseResults}
                    baseScanFile={baseScanFile}
                    baseScanUrl={baseScanUrl}
                    maskUrl={maskUrl}
                    activeSequenceKey={activeSequenceKey}
                    onSelectSequence={handleSelectSequence}
                    onCaseUploaded={handleCaseUploaded}
                    onSegmentationCompleted={handleSegmentationCompleted}
                    onBaseScanSelected={handleBaseScanSelected}
                    onValidationError={handleValidationError}
                    workstationSubTab={workstationSubTab}
                    onWorkstationSubTabChange={setWorkstationSubTab}
                    ingestionFiles={ingestionFiles}
                    onIngestionFilesChange={setIngestionFiles}
                    uploadResult={uploadResult}
                    onUploadResultChange={setUploadResult}
                  />
                </div>
              </div>
            </div>
          )}

          {/* View 3: Step-Based Upload & Pipeline */}
          {activeTab === 'upload' && (
            <UploadAnalysisView
              onCaseUploaded={handleCaseUploaded}
              onSegmentationCompleted={handleSegmentationCompleted}
              onBaseScanSelected={handleBaseScanSelected}
              onValidationError={handleValidationError}
              activeCase={activeCase}
              caseResults={caseResults}
              onOpenWorkstation={() => setActiveTab('workstation')}
              onOpenAnalytics={() => setActiveTab('analytics')}
              ingestionFiles={ingestionFiles}
              onIngestionFilesChange={setIngestionFiles}
              uploadResult={uploadResult}
              onUploadResultChange={setUploadResult}
            />
          )}

          {/* View 4: Dedicated Volumetric Analytics */}
          {activeTab === 'analytics' && (
            <VolumetricAnalyticsView
              caseId={caseResults?.case_id || activeCase?.case_id}
              caseResults={caseResults}
              activeCase={activeCase}
              onNavigateToCentroid={(vox) => {
                setActiveTab('workstation');
              }}
              onOpenWorkstation={() => setActiveTab('workstation')}
            />
          )}

          {/* View 5: Longitudinal Comparison */}
          {activeTab === 'longitudinal' && (
            <LongitudinalView
              caseId={caseResults?.case_id || activeCase?.case_id}
              caseResults={caseResults}
              onBackToWorkstation={() => setActiveTab('workstation')}
            />
          )}

          {/* View 6: AI Trust & Uncertainty */}
          {activeTab === 'trust' && (
            <div className="app-layout-container" style={{ justifyContent: 'center' }}>
              <UncertaintyAndLongitudinalView
                caseId={caseResults?.case_id || activeCase?.case_id}
                caseResults={caseResults}
                onClose={() => setActiveTab('workstation')}
              />
            </div>
          )}

          {/* View 7: Held-Out Cohort Benchmarks */}
          {activeTab === 'cohort' && (
            <div className="app-layout-container" style={{ justifyContent: 'center' }}>
              <CohortBenchmarkView
                onClose={() => setActiveTab('workstation')}
              />
            </div>
          )}

          {/* View 8: Case Registry & Model Card */}
          {activeTab === 'cases' && (
            <CaseRegistryView
              activeCaseId={caseResults?.case_id || activeCase?.case_id}
              onSelectCase={handleSelectPersistedCase}
              onNavigateTab={setActiveTab}
            />
          )}

          {/* View 9: Provenance & Audit Report */}
          {activeTab === 'audit' && (
            <div className="app-layout-container" style={{ justifyContent: 'center' }}>
              <ProvenanceAuditReport
                caseId={caseResults?.case_id || activeCase?.case_id}
                caseData={activeCase}
                caseResults={caseResults}
                onClose={() => setActiveTab('workstation')}
              />
            </div>
          )}

          {/* View 10: Case Validation Diagnostics */}
          {activeTab === 'diagnostics' && (
            <div className="app-layout-container" style={{ justifyContent: 'center' }}>
              <CaseValidationDiagnostics
                caseId={activeCase?.case_id}
                caseData={activeCase}
                errorInfo={validationError}
                onBackToWorkstation={() => setActiveTab('workstation')}
                onUploadCorrected={() => setActiveTab('workstation')}
              />
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
