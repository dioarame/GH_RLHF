// static/js/main.js

import { NotificationSystem } from './ui/NotificationSystem.js';
import { APIClient } from './utils/APIClient.js';
import { ViewerBase } from './viewer/ViewerBase.js';
import { LightingManager } from './lighting/LightingManager.js';
import { MeshLoader } from './viewer/MeshLoader.js';
import { ViewportControls } from './viewer/ViewportControls.js';
import { LIGHT_PRESETS } from './core/constants.js';

class RLHFSystem {
    constructor() {
        this.viewers = { a: null, b: null };
        this.lightingManagers = { a: null, b: null };
        this.meshLoaders = { a: null, b: null };
        this.viewportControls = { a: null, b: null };
        this.currentDesigns = { a: null, b: null };
        this.environmentData = { contour: null, surface: null };
        this.systemTheme = 'light';
        this.selectionHistory = [];
        this.designStats = null;
        this.sessionStats = { 
            total_comparisons: 0, 
            target_comparisons: 100,
            start_time: Date.now()
        };
        
        this.init();
    }
    
    async init() {
        console.log('RLHF 시스템 초기화 중...');
        
        try {
            await this.loadEnvironmentData();
            this.initViewers();
            this.initLightingGUI();
            await this.showTargetSetupModal();
            
            console.log('초기화 완료');
        } catch (error) {
            console.error('초기화 오류:', error);
        }
    }
    
    initViewers() {
        ['a', 'b'].forEach(side => {
            try {
                // 뷰어 생성
                this.viewers[side] = new ViewerBase(`viewer-${side}`, side);
                
                // 조명 매니저 생성
                this.lightingManagers[side] = new LightingManager(this.viewers[side].scene);
                
                // 메시 로더 생성
                this.meshLoaders[side] = new MeshLoader(this.viewers[side].scene);
                
                // 뷰포트 컨트롤 생성
                this.viewportControls[side] = new ViewportControls(
                    this.viewers[side].scene,
                    this.viewers[side].camera,
                    this.viewers[side].controls
                );
                
                console.log(`뷰어 ${side.toUpperCase()} 초기화 완료`);
            } catch (error) {
                console.error(`뷰어 ${side} 초기화 오류:`, error);
            }
        });
        
        // 창 크기 변경 이벤트
        window.addEventListener('resize', () => this.handleResize());
    }
    
    handleResize() {
        Object.values(this.viewers).forEach(viewer => {
            if (viewer) viewer.handleResize();
        });
    }
    
    async loadEnvironmentData() {
        try {
            console.log('환경 데이터 로딩 중...');
            
            const [contour, surface] = await Promise.all([
                APIClient.loadEnvironmentData('Contour.json'),
                APIClient.loadEnvironmentData('Sur.json')
            ]);
            
            if (contour) {
                this.environmentData.contour = contour;
                console.log('Contour 데이터 로드 완료');
            }
            
            if (surface) {
                this.environmentData.surface = surface;
                console.log('Surface 데이터 로드 완료');
            }
        } catch (error) {
            console.warn('환경 데이터 로드 실패:', error);
        }
    }
    
    async showTargetSetupModal() {
        try {
            const modalElement = document.getElementById('targetSetupModal');
            
            if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
                const modal = new bootstrap.Modal(modalElement);
                modal.show();
            } else {
                modalElement.style.display = 'block';
                modalElement.classList.add('show');
                document.body.classList.add('modal-open');
            }
            
            await this.analyzeDesigns();
        } catch (error) {
            console.error('모달 표시 오류:', error);
            await this.analyzeDesigns();
        }
    }
    
    async analyzeDesigns() {
        try {
            console.log('디자인 데이터 분석 시작...');
            
            const data = await APIClient.getDesignStats();
            
            if (data.status === 'success') {
                this.designStats = data.stats;
                this.updateDesignAnalysisUI();
                this.setSmartDefaultTarget();
            } else {
                console.error('API 응답 오류:', data.message);
                this.showAnalysisError('API 응답 오류: ' + data.message);
            }
        } catch (error) {
            console.error('디자인 분석 오류:', error);
            this.showAnalysisError('네트워크 오류: ' + error.message);
        }
    }
    
    showAnalysisError(errorMessage) {
        document.getElementById('modal-loading').innerHTML = `
            <div class="text-center py-4">
                <i class="fa-solid fa-exclamation-triangle text-warning mb-3" style="font-size: 2rem;"></i>
                <div class="text-danger mb-2">분석 오류 발생</div>
                <small class="text-muted">${errorMessage}</small>
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="window.rlhfSystem.useDefaultSettings()">
                        기본 설정으로 계속
                    </button>
                </div>
            </div>
        `;
    }
    
    useDefaultSettings() {
        this.designStats = {
            top_designs: 15,
            random_designs: 15, 
            total_designs: 30,
            max_comparisons: 435
        };
        
        this.updateDesignAnalysisUI();
        this.setSmartDefaultTarget();
    }
    
    updateDesignAnalysisUI() {
        const stats = this.designStats;
        
        document.getElementById('modal-top-count').textContent = stats.top_designs;
        document.getElementById('modal-random-count').textContent = stats.random_designs;
        document.getElementById('modal-total-designs').textContent = stats.total_designs;
        document.getElementById('modal-max-pairs').textContent = stats.max_comparisons;
        
        const maxTarget = Math.min(stats.max_comparisons, 500);
        const slider = document.getElementById('modal-target-slider');
        slider.max = maxTarget;
        
        this.updateSmartButtons();
        
        document.getElementById('modal-loading').style.display = 'none';
        document.getElementById('modal-content').style.display = 'block';
        document.getElementById('modal-footer').style.display = 'flex';
        
        console.log('디자인 분석 완료:', stats);
    }
    
    updateSmartButtons() {
        if (!this.designStats) return;
        
        const maxComparisons = Math.min(this.designStats.max_comparisons, 500);
        
        const quickTarget = Math.max(50, Math.floor(maxComparisons * 0.15));
        const recommendedTarget = Math.max(100, Math.floor(maxComparisons * 0.35));
        const thoroughTarget = Math.max(150, Math.floor(maxComparisons * 0.60));
        
        document.getElementById('quick-target').textContent = `${quickTarget}회`;
        document.getElementById('recommended-target').textContent = `${recommendedTarget}회`;
        document.getElementById('thorough-target').textContent = `${thoroughTarget}회`;
        
        const slider = document.getElementById('modal-target-slider');
        slider.value = recommendedTarget;
        window.updateModalTarget();
    }
    
    setSmartDefaultTarget() {
        if (!this.designStats) return;
        
        const totalDesigns = this.designStats.total_designs;
        const maxComparisons = this.designStats.max_comparisons;
        
        let recommendedTarget;
        if (totalDesigns <= 10) {
            recommendedTarget = Math.min(50, maxComparisons);
        } else if (totalDesigns <= 20) {
            recommendedTarget = Math.min(100, maxComparisons);
        } else if (totalDesigns <= 30) {
            recommendedTarget = Math.min(150, maxComparisons);
        } else {
            recommendedTarget = Math.min(200, maxComparisons);
        }
        
        this.sessionStats.target_comparisons = recommendedTarget;
        this.updateStats();
        
        console.log(`스마트 기본 목표 설정: ${recommendedTarget}회 (총 디자인: ${totalDesigns}개)`);
    }
    
    async loadNextComparison() {
        try {
            this.showLoading(true);
            
            const data = await APIClient.getNextComparison();
            
            if (data.status === 'success') {
                await this.loadDesignPair(data.design_a, data.design_b);
                this.updateStats();
            } else {
                console.error('비교 쌍 로드 실패:', data.message);
                this.loadFallbackData();
            }
        } catch (error) {
            console.error('API 오류:', error);
            this.loadFallbackData();
        } finally {
            this.showLoading(false);
        }
    }
    
    loadFallbackData() {
        console.log('대체 데이터 로딩...');
        const mockDesigns = {
            design_a: {
                id: `fallback_a_${Date.now()}`,
                state: [0.45, 3.2, 85000, 0.78]
            },
            design_b: {
                id: `fallback_b_${Date.now()}`,
                state: [0.62, 4.1, 72000, 0.92]
            }
        };
        
        this.loadDesignPair(mockDesigns.design_a, mockDesigns.design_b);
        this.updateStats();
    }
    
    async loadDesignPair(designA, designB) {
        this.currentDesigns.a = designA;
        this.currentDesigns.b = designB;
        
        // 환경 메시 로드
        await Promise.all([
            this.meshLoaders.a.loadEnvironmentMeshes(this.environmentData),
            this.meshLoaders.b.loadEnvironmentMeshes(this.environmentData)
        ]);
        
        // 디자인 메시 로드
        await Promise.all([
            this.meshLoaders.a.loadDesignMesh(designA.id),
            this.meshLoaders.b.loadDesignMesh(designB.id)
        ]);
        
        // 초기 뷰 설정 (Perspective) - 타이밍 조정
        setTimeout(() => {
            if (this.viewportControls.a) {
                this.viewportControls.a.setPerspectiveView();
            }
            if (this.viewportControls.b) {
                this.viewportControls.b.setPerspectiveView();
            }
        }, 200); // 메시 로딩이 완전히 끝난 후 실행
        
        // 메트릭 업데이트
        this.updateMetrics('a', designA);
        this.updateMetrics('b', designB);
        
        // 현재 비교 쌍 업데이트
        const designALabel = this.formatDesignLabel(designA.id);
        const designBLabel = this.formatDesignLabel(designB.id);
        document.getElementById('current-pair').textContent = `${designALabel} vs ${designBLabel}`;
    }
    
    createDefaultMesh(side) {
        const viewer = this.viewers[side];
        if (!viewer) return;
        
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshStandardMaterial({
            color: 0xe0e0e0,
            roughness: 0.7,
            metalness: 0.1
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(0, 0.5, 0);
        
        viewer.scene.add(mesh);
        console.log(`${side} 기본 큐브 생성됨`);
    }
    
    formatDesignLabel(designId) {
        if (designId.includes('random')) {
            const match = designId.match(/random.*?(\d+)/i);
            const num = match ? match[1].padStart(2, '0') : '00';
            return `RANDOM ${num}`;
        } else if (designId.includes('top')) {
            const match = designId.match(/top.*?(\d+)/i);
            const num = match ? match[1].padStart(2, '0') : '00';
            return `TOP ${num}`;
        } else if (designId.includes('mock') || designId.includes('fallback')) {
            return designId.includes('_a') ? 'DEMO A' : 'DEMO B';
        } else {
            const parts = designId.split('_');
            if (parts.length >= 2) {
                return parts[0].toUpperCase() + ' ' + (parts[1] || '00').padStart(2, '0');
            }
            return designId.toUpperCase();
        }
    }
    
    updateMetrics(side, design) {
        const state = design.state || [0, 0, 0, 0];
        const reward = design.reward || 0;
        
        // 메트릭 업데이트
        const metrics = {
            bcr: this.formatMetric(state[0] * 100, '%', [0, 70]),
            far: this.formatMetric(state[1] * 100, '%', [200, 500]),
            sunlight: this.formatMetric(state[2], 'kWh/㎡', [80000, 100000]),
            svr: this.formatMetric(state[3], '', [0.7, 0.9])
        };
        
        Object.entries(metrics).forEach(([key, {value, className}]) => {
            const element = document.getElementById(`${key}-${side}`);
            if (element) {
                element.textContent = value;
                element.className = `metric-value ${className}`;
            }
        });
        
        // 보상값 표시
        this.updateRewardDisplay(side, reward);
    }
    
    formatMetric(value, unit, goodRange = null) {
        if (value === null || value === undefined || isNaN(value)) {
            return { value: 'N/A', className: '' };
        }
        
        let formattedValue;
        if (unit === '%') {
            formattedValue = `${value.toFixed(1)}${unit}`;
        } else if (unit === 'kWh/㎡') {
            formattedValue = `${(value/1000).toFixed(1)}k${unit}`;
        } else {
            formattedValue = `${value.toFixed(2)}${unit}`;
        }
        
        let className = '';
        if (goodRange) {
            if (value >= goodRange[0] && value <= goodRange[1]) {
                className = 'good';
            } else if (Math.abs(value - goodRange[0]) <= (goodRange[1] - goodRange[0]) * 0.2 ||
                      Math.abs(value - goodRange[1]) <= (goodRange[1] - goodRange[0]) * 0.2) {
                className = 'warning';
            } else {
                className = 'danger';
            }
        }
        
        return { value: formattedValue, className };
    }
    
    updateRewardDisplay(side, reward) {
        const container = document.getElementById(`viewer-${side}`);
        let rewardDisplay = container.querySelector('.reward-display');
        
        if (!rewardDisplay) {
            rewardDisplay = document.createElement('div');
            rewardDisplay.className = 'reward-display';
            container.appendChild(rewardDisplay);
        }
        
        let rewardClass = 'reward-neutral';
        if (reward > 2) rewardClass = 'reward-high';
        else if (reward > 0) rewardClass = 'reward-good';
        else if (reward < -2) rewardClass = 'reward-low';
        else if (reward < 0) rewardClass = 'reward-poor';
        
        rewardDisplay.innerHTML = `
            <div class="reward-badge ${rewardClass}">
                <i class="fa-solid fa-trophy"></i>
                <span>보상: ${reward.toFixed(3)}</span>
            </div>
        `;
    }
    
    async selectDesign(side) {
        try {
            const designA = this.currentDesigns.a;
            const designB = this.currentDesigns.b;
            
            if (!designA || !designB) return;
            
            this.showSelectionEffect(side);
            
            const feedbackData = {
                session_id: 'rlhf_session',
                design_a_id: designA.id,
                design_b_id: designB.id,
                selected_design: side === 'a' ? designA.id : designB.id,
                design_a_state: designA.state,
                design_b_state: designB.state,
                timestamp: Date.now()
            };
            
            const data = await APIClient.submitFeedback(feedbackData);
            
            if (data.status === 'success') {
                this.sessionStats.total_comparisons++;
                
                const selectedDesign = side === 'a' ? designA : designB;
                const notSelectedDesign = side === 'a' ? designB : designA;
                this.selectionHistory.push({
                    selected: selectedDesign,
                    notSelected: notSelectedDesign,
                    timestamp: Date.now()
                });
                
                this.updateStats();
                this.analyzeSelectionTendency();
                
                // 목표 도달 확인
                if (this.sessionStats.total_comparisons >= this.sessionStats.target_comparisons) {
                    this.showSessionComplete();
                } else {
                    setTimeout(() => {
                        this.loadNextComparison();
                    }, 1500);
                }
            }
        } catch (error) {
            console.error('선택 처리 오류:', error);
        }
    }
    
    showSelectionEffect(side) {
        const containerA = document.getElementById('viewer-a');
        const containerB = document.getElementById('viewer-b');
        
        containerA.classList.remove('selected');
        containerB.classList.remove('selected');
        
        const selectedContainer = document.getElementById(`viewer-${side}`);
        selectedContainer.classList.add('selected');
        
        setTimeout(() => {
            selectedContainer.classList.remove('selected');
        }, 1500);
    }
    
    updateStats() {
        document.getElementById('total-comparisons').textContent = this.sessionStats.total_comparisons;
        const progress = (this.sessionStats.total_comparisons / this.sessionStats.target_comparisons) * 100;
        const progressText = `${Math.round(progress)}% (${this.sessionStats.total_comparisons}/${this.sessionStats.target_comparisons})`;
        document.getElementById('session-progress').textContent = progressText;
    }
    
    analyzeSelectionTendency() {
        const historyCount = this.selectionHistory.length;
        const tendencyElement = document.getElementById('selection-tendency');
        
        if (historyCount < 3) {
            tendencyElement.textContent = '분석 중...';
            tendencyElement.className = 'card-title text-secondary';
            return;
        }
        
        // 선택된 디자인들의 메트릭 평균 계산
        const selectedMetrics = {
            bcr: 0, far: 0, sunlight: 0, svr: 0
        };
        
        this.selectionHistory.forEach(entry => {
            const state = entry.selected.state || [0, 0, 0, 0];
            selectedMetrics.bcr += state[0];
            selectedMetrics.far += state[1];
            selectedMetrics.sunlight += state[2];
            selectedMetrics.svr += state[3];
        });
        
        // 평균 계산
        Object.keys(selectedMetrics).forEach(key => {
            selectedMetrics[key] /= historyCount;
        });
        
        // 경향성 분석
        let tendency = '균형적';
        let tendencyClass = 'text-info';
        
        if (selectedMetrics.far > 3.5) {
            tendency = 'FAR 선호';
            tendencyClass = 'text-primary';
        } else if (selectedMetrics.bcr > 0.6) {
            tendency = '고밀도 선호';
            tendencyClass = 'text-danger';
        } else if (selectedMetrics.sunlight > 85000) {
            tendency = '일사량 중시';
            tendencyClass = 'text-warning';
        } else if (selectedMetrics.svr > 0.85) {
            tendency = 'SV비 중시';
            tendencyClass = 'text-success';
        }
        
        // 일관성 검사
        if (historyCount >= 5) {
            const recentSelections = this.selectionHistory.slice(-3);
            const consistency = this.calculateConsistency(recentSelections);
            
            if (consistency > 0.8) {
                tendency += ' (일관됨)';
            } else if (consistency < 0.4) {
                tendency += ' (혼재)';
            }
        }
        
        tendencyElement.textContent = tendency;
        tendencyElement.className = `card-title ${tendencyClass}`;
    }
    
    calculateConsistency(recentSelections) {
        if (recentSelections.length < 2) return 0;
        
        let consistencyScore = 0;
        
        for (let i = 0; i < recentSelections.length - 1; i++) {
            const current = recentSelections[i].selected.state || [0, 0, 0, 0];
            const next = recentSelections[i + 1].selected.state || [0, 0, 0, 0];
            
            let similarity = 0;
            for (let j = 0; j < 4; j++) {
                const diff = Math.abs(current[j] - next[j]);
                const range = Math.max(current[j], next[j]) || 1;
                similarity += 1 - (diff / range);
            }
            consistencyScore += similarity / 4;
        }
        
        return consistencyScore / (recentSelections.length - 1);
    }
    
    showSessionComplete() {
        // 세션 통계 계산
        const duration = Math.round((Date.now() - this.sessionStats.start_time) / 60000); // 분 단위
        const tendencyElement = document.getElementById('selection-tendency');
        const tendency = tendencyElement.textContent;
        
        // 모달에 정보 업데이트
        document.getElementById('final-comparisons').textContent = this.sessionStats.total_comparisons;
        document.getElementById('final-duration').textContent = `${duration}분`;
        document.getElementById('final-tendency').textContent = tendency;
        
        // 모달 표시
        const modalElement = document.getElementById('sessionCompleteModal');
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        } else {
            modalElement.style.display = 'block';
            modalElement.classList.add('show');
            document.body.classList.add('modal-open');
        }
    }
    
    downloadData() {
        // 수집된 데이터 정리
        const sessionData = {
            session_info: {
                total_comparisons: this.sessionStats.total_comparisons,
                target_comparisons: this.sessionStats.target_comparisons,
                start_time: new Date(this.sessionStats.start_time).toISOString(),
                end_time: new Date().toISOString(),
                duration_minutes: Math.round((Date.now() - this.sessionStats.start_time) / 60000)
            },
            design_stats: this.designStats,
            selection_history: this.selectionHistory,
            preference_pairs: this.selectionHistory.map(entry => ({
                preferred_id: entry.selected.id,
                rejected_id: entry.notSelected.id,
                preferred_state: entry.selected.state,
                rejected_state: entry.notSelected.state,
                timestamp: new Date(entry.timestamp).toISOString()
            })),
            analysis: {
                total_selections: this.selectionHistory.length,
                tendency: document.getElementById('selection-tendency').textContent,
                metrics_average: this.calculateAverageMetrics()
            }
        };
        
        // JSON 파일로 다운로드
        const dataStr = JSON.stringify(sessionData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `rlhf_session_${Date.now()}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
        
        NotificationSystem.show('세션 데이터가 다운로드되었습니다.', 'success');
    }
    
    calculateAverageMetrics() {
        if (this.selectionHistory.length === 0) return null;
        
        const avgMetrics = { bcr: 0, far: 0, sunlight: 0, svr: 0 };
        
        this.selectionHistory.forEach(entry => {
            const state = entry.selected.state || [0, 0, 0, 0];
            avgMetrics.bcr += state[0];
            avgMetrics.far += state[1];
            avgMetrics.sunlight += state[2];
            avgMetrics.svr += state[3];
        });
        
        const count = this.selectionHistory.length;
        Object.keys(avgMetrics).forEach(key => {
            avgMetrics[key] /= count;
        });
        
        return avgMetrics;
    }
    
    continueSession() {
        // 모달 닫기
        const modalElement = document.getElementById('sessionCompleteModal');
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) modal.hide();
        }
        
        // 목표를 50개 더 증가
        this.sessionStats.target_comparisons += 50;
        this.updateStats();
        
        // 다음 비교 로드
        this.loadNextComparison();
        
        NotificationSystem.show('세션이 연장되었습니다. 50개의 추가 비교를 진행합니다.', 'info');
    }
    showLoading(show) {
        ['a', 'b'].forEach(side => {
            const loading = document.getElementById(`loading-${side}`);
            if (loading) {
                loading.style.display = show ? 'flex' : 'none';
            }
        });
    }
    
    initLightingGUI() {
        if (typeof dat === 'undefined') {
            console.warn('dat.GUI가 로드되지 않았습니다.');
            return;
        }
        
        const gui = new dat.GUI({ width: 300 });
        gui.domElement.style.position = 'fixed';
        gui.domElement.style.top = '70px';
        gui.domElement.style.left = '10px';
        gui.domElement.style.zIndex = '1000';
        
        // 프리셋 버튼
        const presets = {
            '밝은 낮': () => this.applyLightPreset('bright'),
            '선명한 낮': () => this.applyLightPreset('vibrant'),
            '부드러운 낮': () => this.applyLightPreset('soft'),
            '흐린 날': () => this.applyLightPreset('cloudy'),
            '저녁': () => this.applyLightPreset('evening')
        };
        
        const presetFolder = gui.addFolder('조명 프리셋');
        Object.keys(presets).forEach(name => {
            presetFolder.add(presets, name);
        });
        presetFolder.open();
        
        this.gui = gui;
    }
    
    applyLightPreset(presetName) {
        ['a', 'b'].forEach(side => {
            const lightingManager = this.lightingManagers[side];
            const viewer = this.viewers[side];
            
            if (lightingManager && viewer) {
                const rendererSettings = lightingManager.applyPreset(presetName);
                if (rendererSettings && viewer.renderer) {
                    viewer.renderer.toneMappingExposure = rendererSettings.toneMappingExposure;
                }
            }
        });
        
        NotificationSystem.show(`조명 프리셋 '${presetName}' 적용됨`, 'success');
    }
}

// 전역 함수들 (HTML에서 호출하는 함수들)
window.rlhfSystem = null;

window.selectDesign = function(side) {
    if (window.rlhfSystem) window.rlhfSystem.selectDesign(side);
};

window.skipComparison = function() {
    if (window.rlhfSystem) window.rlhfSystem.loadNextComparison();
};

window.loadNextComparison = function() {
    if (window.rlhfSystem) window.rlhfSystem.loadNextComparison();
};

window.resetView = function(side) {
    if (window.rlhfSystem && window.rlhfSystem.viewportControls[side]) {
        window.rlhfSystem.viewportControls[side].resetView();
    }
};

window.toggleWireframe = function(side) {
    if (window.rlhfSystem && window.rlhfSystem.viewportControls[side]) {
        window.rlhfSystem.viewportControls[side].toggleWireframe();
    }
};

window.captureView = function(side) {
    if (window.rlhfSystem && window.rlhfSystem.viewers[side]) {
        const canvas = window.rlhfSystem.viewers[side].renderer.domElement;
        const dataUrl = canvas.toDataURL('image/png');
        
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = `design_${side}_${Date.now()}.png`;
        link.click();
        
        NotificationSystem.show(`디자인 ${side.toUpperCase()} 스크린샷이 저장되었습니다.`, 'success');
    }
};

window.setViewport = function(side, viewType) {
    if (window.rlhfSystem && window.rlhfSystem.viewportControls[side]) {
        window.rlhfSystem.viewportControls[side].setViewport(viewType);
    }
};

window.toggleViewerMode = function(side) {
    if (window.rlhfSystem && window.rlhfSystem.viewers[side]) {
        const viewer = window.rlhfSystem.viewers[side];
        const newMode = viewer.viewerMode === 'light' ? 'dark' : 'light';
        viewer.updateBackground(newMode);
    }
};

window.toggleSystemTheme = function() {
    if (!window.rlhfSystem) return;
    
    const newTheme = window.rlhfSystem.systemTheme === 'light' ? 'dark' : 'light';
    window.rlhfSystem.systemTheme = newTheme;
    
    document.documentElement.setAttribute('data-theme', newTheme);
    
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.className = newTheme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
    }
};

window.showHelp = function() {
    const helpMessage = `
🏗️ CAD 데이터 기반 RLHF 시스템 사용법

📋 기본 사용법:
• 두 디자인 중 더 선호하는 디자인을 선택하세요
• 건축 지표(건폐율, 용적률, 일사량, SV Ratio)를 참고하여 판단

🎮 뷰어 조작법:
• 마우스 왼쪽 버튼 + 드래그: 회전
• 마우스 휠: 확대/축소
• 마우스 오른쪽 버튼 + 드래그: 이동

💡 팁:
• 선택의 경향성은 3회 이상 선택 후부터 분석됩니다
    `;
    alert(helpMessage);
};

// 모달 관련 함수들
window.updateModalTarget = function() {
    const slider = document.getElementById('modal-target-slider');
    const display = document.getElementById('modal-target-display');
    const info = document.getElementById('modal-target-info');
    const time = document.getElementById('estimated-time');
    
    const value = parseInt(slider.value);
    display.textContent = value;
    
    if (window.rlhfSystem && window.rlhfSystem.designStats) {
        const percentage = Math.round((value / window.rlhfSystem.designStats.max_comparisons) * 100);
        info.textContent = `전체 비교의 ${percentage}%`;
    }
    
    const estimatedMinutes = Math.ceil(value / 4);
    time.textContent = `${estimatedMinutes}분`;
};

window.setModalTarget = function(type) {
    if (!window.rlhfSystem || !window.rlhfSystem.designStats) return;
    
    const maxComparisons = Math.min(window.rlhfSystem.designStats.max_comparisons, 500);
    let target;
    
    switch (type) {
        case 'quick':
            target = Math.max(50, Math.floor(maxComparisons * 0.15));
            break;
        case 'recommended':
            target = Math.max(100, Math.floor(maxComparisons * 0.35));
            break;
        case 'thorough':
            target = Math.max(150, Math.floor(maxComparisons * 0.60));
            break;
    }
    
    const slider = document.getElementById('modal-target-slider');
    slider.value = target;
    window.updateModalTarget();
};

window.startSession = function() {
    const slider = document.getElementById('modal-target-slider');
    const targetValue = parseInt(slider.value);
    
    if (window.rlhfSystem) {
        window.rlhfSystem.sessionStats.target_comparisons = targetValue;
        window.rlhfSystem.updateStats();
    }
    
    // 모달 닫기
    const modalElement = document.getElementById('targetSetupModal');
    
    try {
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) modal.hide();
        }
    } catch (error) {
        console.error('모달 닫기 오류:', error);
    }
    
    modalElement.style.display = 'none';
    modalElement.classList.remove('show');
    document.body.classList.remove('modal-open');
    
    const backdrop = document.querySelector('.modal-backdrop');
    if (backdrop) backdrop.remove();
    
    // 첫 번째 비교 로드
    if (window.rlhfSystem) {
        window.rlhfSystem.loadNextComparison();
    }
    
    console.log(`세션 시작: 목표 ${targetValue}회`);
};

// 키보드 단축키
document.addEventListener('keydown', (event) => {
    if (event.key === 'f' || event.key === 'F') {
        // F 키로 양쪽 뷰어 모드 동시 전환
        if (window.rlhfSystem) {
            const currentModeA = window.rlhfSystem.viewers.a.viewerMode;
            const newMode = currentModeA === 'light' ? 'dark' : 'light';
            
            // 양쪽 뷰어를 같은 모드로 설정
            window.rlhfSystem.viewers.a.updateBackground(newMode);
            window.rlhfSystem.viewers.b.updateBackground(newMode);
            
            console.log(`F키: 양쪽 뷰어를 ${newMode} 모드로 전환`);
        }
    }
});

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.rlhfSystem = new RLHFSystem();
});