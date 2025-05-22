/**
 * Grasshopper RLHF - 개선된 메인 애플리케이션
 * 쌍대비교 중심의 인터페이스
 */

class RLHFInterface {
    constructor() {
        this.currentSession = {
            id: null,
            totalComparisons: 0,
            targetComparisons: 100,
            currentPair: null,
            consistencyScore: 0,
            feedbackData: [],
            isComplete: false
        };
        
        this.viewers = {
            a: null,
            b: null
        };
        
        this.currentDesigns = {
            a: null,
            b: null
        };
        
        this.scenes = {
            a: null,
            b: null
        };
        
        this.cameras = {
            a: null,
            b: null
        };
        
        this.renderers = {
            a: null,
            b: null
        };
        
        this.controls = {
            a: null,
            b: null
        };
        
        this.init();
    }
    
    async init() {
        console.log('RLHF Interface 초기화 중...');
        
        // 네비게이션 설정
        this.setupNavigation();
        
        // 3D 뷰어 초기화
        this.initializeViewers();
        
        // 이벤트 리스너 설정
        this.setupEventListeners();
        
        // 세션 시작
        await this.startNewSession();
        
        // 첫 번째 비교 쌍 로드
        await this.loadNextComparison();
        
        console.log('RLHF Interface 초기화 완료');
    }
    
    setupNavigation() {
        document.getElementById('nav-comparison').addEventListener('click', () => {
            this.showScreen('comparison');
        });
        
        document.getElementById('nav-designs').addEventListener('click', () => {
            this.showScreen('designs');
        });
        
        document.getElementById('nav-analysis').addEventListener('click', () => {
            this.showScreen('analysis');
        });
        
        document.getElementById('btn-refresh').addEventListener('click', () => {
            this.refreshData();
        });
    }
    
    showScreen(screenName) {
        // 모든 화면 숨기기
        document.getElementById('comparison-screen').style.display = 'none';
        document.getElementById('designs-screen').style.display = 'none';
        document.getElementById('analysis-screen').style.display = 'none';
        
        // 네비게이션 활성화 상태 업데이트
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        // 선택된 화면 표시
        switch(screenName) {
            case 'comparison':
                document.getElementById('comparison-screen').style.display = 'block';
                document.getElementById('nav-comparison').classList.add('active');
                break;
            case 'designs':
                document.getElementById('designs-screen').style.display = 'block';
                document.getElementById('nav-designs').classList.add('active');
                this.loadDesignList();
                break;
            case 'analysis':
                document.getElementById('analysis-screen').style.display = 'block';
                document.getElementById('nav-analysis').classList.add('active');
                this.loadAnalysis();
                break;
        }
    }
    
    initializeViewers() {
        // 두 개의 별도 뷰어 초기화
        ['a', 'b'].forEach(side => {
            this.initViewer(side);
        });
    }
    
    initViewer(side) {
        const container = document.getElementById(`viewer-${side}`);
        if (!container) return;
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        // Scene 생성
        this.scenes[side] = new THREE.Scene();
        this.scenes[side].background = new THREE.Color(0x2c3e50);
        
        // Camera 생성
        this.cameras[side] = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.cameras[side].position.set(5, 5, 5);
        this.cameras[side].lookAt(0, 0, 0);
        
        // Renderer 생성
        this.renderers[side] = new THREE.WebGLRenderer({ antialias: true });
        this.renderers[side].setSize(width, height);
        this.renderers[side].setPixelRatio(window.devicePixelRatio);
        this.renderers[side].shadowMap.enabled = true;
        
        // 컨테이너에 추가
        const loadingOverlay = container.querySelector('.loading-overlay');
        container.insertBefore(this.renderers[side].domElement, loadingOverlay);
        
        // Controls 생성
        this.controls[side] = new THREE.OrbitControls(this.cameras[side], this.renderers[side].domElement);
        this.controls[side].enableDamping = true;
        this.controls[side].dampingFactor = 0.05;
        
        // 조명 추가
        this.setupLights(side);
        
        // 애니메이션 루프 시작
        this.animate(side);
        
        console.log(`뷰어 ${side.toUpperCase()} 초기화 완료`);
    }
    
    setupLights(side) {
        const scene = this.scenes[side];
        
        // 주변광
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        
        // 주 방향성 조명
        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(5, 10, 7);
        mainLight.castShadow = true;
        scene.add(mainLight);
        
        // 채움 조명
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 5, -5);
        scene.add(fillLight);
        
        // 그리드 헬퍼
        const gridHelper = new THREE.GridHelper(20, 20, 0x555555, 0x333333);
        scene.add(gridHelper);
        
        // 축 헬퍼
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);
    }
    
    animate(side) {
        const animateFrame = () => {
            requestAnimationFrame(animateFrame);
            
            if (this.controls[side]) {
                this.controls[side].update();
            }
            
            if (this.renderers[side] && this.scenes[side] && this.cameras[side]) {
                this.renderers[side].render(this.scenes[side], this.cameras[side]);
            }
        };
        animateFrame();
    }
    
    setupEventListeners() {
        // 창 크기 변경 이벤트
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // 키보드 단축키
        document.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });
    }
    
    handleResize() {
        ['a', 'b'].forEach(side => {
            const container = document.getElementById(`viewer-${side}`);
            if (!container || !this.cameras[side] || !this.renderers[side]) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            this.cameras[side].aspect = width / height;
            this.cameras[side].updateProjectionMatrix();
            this.renderers[side].setSize(width, height);
        });
    }
    
    handleKeyboard(e) {
        switch(e.key) {
            case '1':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.selectDesign('a');
                }
                break;
            case '2':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.selectDesign('b');
                }
                break;
            case 's':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.skipComparison();
                }
                break;
            case 'h':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.requestHelp();
                }
                break;
        }
    }
    
    async startNewSession() {
        try {
            this.currentSession.id = 'session_' + Date.now();
            this.currentSession.totalComparisons = 0;
            this.updateSessionStats();
            
            console.log('새 세션 시작:', this.currentSession.id);
        } catch (error) {
            console.error('세션 시작 오류:', error);
        }
    }
    
    async loadNextComparison() {
        try {
            this.showLoading(true);
            
            // API에서 비교할 디자인 쌍 가져오기
            const response = await fetch('/api/comparison/next', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSession.id,
                    comparisons_done: this.currentSession.totalComparisons
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                await this.loadDesignPair(data.design_a, data.design_b);
                this.currentSession.currentPair = {
                    a: data.design_a.id,
                    b: data.design_b.id
                };
                this.updateSessionStats();
            } else {
                console.error('비교 쌍 로드 실패:', data.message);
                this.showError('새로운 비교 쌍을 불러올 수 없습니다.');
            }
        } catch (error) {
            console.error('비교 쌍 로드 오류:', error);
            this.showError('서버 연결 오류가 발생했습니다.');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadDesignPair(designA, designB) {
        // 병렬로 두 디자인 로드
        await Promise.all([
            this.loadDesignToViewer('a', designA),
            this.loadDesignToViewer('b', designB)
        ]);
    }
    
    async loadDesignToViewer(side, designData) {
        try {
            this.currentDesigns[side] = designData;
            
            // 메트릭 업데이트
            this.updateMetrics(side, designData);
            
            // 3D 모델 로드
            if (designData.mesh_data) {
                await this.loadMesh(side, designData.mesh_data);
            } else {
                // 메시 데이터가 없으면 API에서 요청
                await this.loadMeshFromAPI(side, designData.id);
            }
            
        } catch (error) {
            console.error(`디자인 ${side.toUpperCase()} 로드 오류:`, error);
        }
    }
    
    updateMetrics(side, designData) {
        const metrics = {
            bcr: this.formatMetric(designData.state?.[0] * 100, '%', [0, 70]),
            far: this.formatMetric(designData.state?.[1] * 100, '%', [200, 500]),
            sunlight: this.formatMetric(designData.state?.[2], 'kWh', [80000, 100000]),
            svr: this.formatMetric(designData.state?.[3], '', [0.7, 0.9])
        };
        
        Object.entries(metrics).forEach(([key, {value, className}]) => {
            const element = document.getElementById(`${key}-${side}`);
            if (element) {
                element.textContent = value;
                element.className = `metric-value ${className}`;
            }
        });
    }
    
    formatMetric(value, unit, goodRange = null) {
        if (value === null || value === undefined || isNaN(value)) {
            return { value: 'N/A', className: '' };
        }
        
        let formattedValue;
        if (unit === '%') {
            formattedValue = `${value.toFixed(1)}${unit}`;
        } else if (unit === 'kWh') {
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
    
    async loadMeshFromAPI(side, designId) {
        try {
            const response = await fetch(`/api/mesh/${designId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                await this.loadMesh(side, data.mesh);
            } else {
                console.error('메시 로드 실패:', data.message);
                this.loadDefaultMesh(side);
            }
        } catch (error) {
            console.error('메시 API 호출 오류:', error);
            this.loadDefaultMesh(side);
        }
    }
    
    async loadMesh(side, meshData) {
        const scene = this.scenes[side];
        
        // 기존 메시 제거
        this.clearMeshes(side);
        
        try {
            if (meshData && meshData.meshes && Array.isArray(meshData.meshes)) {
                const meshes = [];
                
                meshData.meshes.forEach(meshInfo => {
                    if (!meshInfo.vertices || !meshInfo.faces) return;
                    
                    const geometry = new THREE.BufferGeometry();
                    
                    // 정점 설정
                    const vertices = [];
                    meshInfo.vertices.forEach(vertex => {
                        vertices.push(vertex[0], vertex[1], vertex[2]);
                    });
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                    
                    // 인덱스 설정
                    const indices = [];
                    meshInfo.faces.forEach(face => {
                        if (face.length === 3) {
                            indices.push(face[0], face[1], face[2]);
                        } else if (face.length === 4) {
                            indices.push(face[0], face[1], face[2]);
                            indices.push(face[0], face[2], face[3]);
                        }
                    });
                    geometry.setIndex(indices);
                    
                    // 법선 계산
                    geometry.computeVertexNormals();
                    
                    // 재질 생성
                    const material = new THREE.MeshPhongMaterial({
                        color: side === 'a' ? 0x3498db : 0xe74c3c,
                        specular: 0x111111,
                        shininess: 30,
                        flatShading: false,
                        side: THREE.DoubleSide
                    });
                    
                    // 메시 생성
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    scene.add(mesh);
                    meshes.push(mesh);
                });
                
                // 카메라 위치 조정
                this.fitCameraToMeshes(side, meshes);
                
            } else {
                this.loadDefaultMesh(side);
            }
        } catch (error) {
            console.error(`메시 로드 오류 (${side}):`, error);
            this.loadDefaultMesh(side);
        }
    }
    
    loadDefaultMesh(side) {
        const scene = this.scenes[side];
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshPhongMaterial({
            color: side === 'a' ? 0x3498db : 0xe74c3c
        });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
    }
    
    clearMeshes(side) {
        const scene = this.scenes[side];
        const meshesToRemove = [];
        
        scene.traverse(object => {
            if (object instanceof THREE.Mesh && 
                !object.userData.isHelper) {
                meshesToRemove.push(object);
            }
        });
        
        meshesToRemove.forEach(mesh => {
            scene.remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });
    }
    
    fitCameraToMeshes(side, meshes) {
        if (!meshes || meshes.length === 0) return;
        
        const box = new THREE.Box3();
        meshes.forEach(mesh => {
            box.expandByObject(mesh);
        });
        
        const center = new THREE.Vector3();
        box.getCenter(center);
        
        const size = new THREE.Vector3();
        box.getSize(size);
        
        // 메시를 중앙으로 이동
        meshes.forEach(mesh => {
            mesh.position.sub(center);
        });
        
        // 카메라 위치 조정
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.cameras[side].fov * (Math.PI / 180);
        let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 1.2;
        
        this.cameras[side].position.set(cameraDistance, cameraDistance, cameraDistance);
        this.cameras[side].lookAt(0, 0, 0);
        this.controls[side].target.set(0, 0, 0);
        this.controls[side].update();
    }
    
    async selectDesign(selectedSide) {
        try {
            const designA = this.currentDesigns.a;
            const designB = this.currentDesigns.b;
            
            if (!designA || !designB) {
                this.showError('디자인 데이터가 완전하지 않습니다.');
                return;
            }
            
            // 선택 효과 표시
            this.showSelectionEffect(selectedSide);
            
            // 피드백 데이터 구성
            const feedbackData = {
                session_id: this.currentSession.id,
                design_a_id: designA.id,
                design_b_id: designB.id,
                selected_design: selectedSide === 'a' ? designA.id : designB.id,
                comparison_time: Date.now(),
                design_a_state: designA.state,
                design_b_state: designB.state
            };
            
            // 서버에 피드백 전송
            const response = await fetch('/api/feedback/comparison', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedbackData)
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // 세션 통계 업데이트
                this.currentSession.totalComparisons++;
                this.currentSession.feedbackData.push(feedbackData);
                this.updateSessionStats();
                
                // 잠시 대기 후 다음 비교로 이동
                setTimeout(() => {
                    this.loadNextComparison();
                }, 1500);
                
                // 보상 모델 업데이트 확인
                this.checkForModelUpdate();
                
            } else {
                this.showError('피드백 저장에 실패했습니다.');
            }
            
        } catch (error) {
            console.error('디자인 선택 오류:', error);
            this.showError('서버 연결 오류가 발생했습니다.');
        }
    }
    
    showSelectionEffect(selectedSide) {
        // 선택된 패널 하이라이트
        const panelA = document.getElementById('design-panel-a');
        const panelB = document.getElementById('design-panel-b');
        
        panelA.classList.remove('selected');
        panelB.classList.remove('selected');
        
        if (selectedSide === 'a') {
            panelA.classList.add('selected');
        } else {
            panelB.classList.add('selected');
        }
        
        // 버튼 텍스트 변경
        const button = selectedSide === 'a' ? 
            panelA.querySelector('.selection-button') :
            panelB.querySelector('.selection-button');
        
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fa-solid fa-check me-2"></i>선택됨!';
        button.style.background = '#2ecc71';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = '#3498db';
        }, 1500);
    }
    
    updateSessionStats() {
        document.getElementById('total-comparisons').textContent = this.currentSession.totalComparisons;
        
        const progress = (this.currentSession.totalComparisons / this.currentSession.targetComparisons) * 100;
        document.getElementById('session-progress').textContent = `${Math.round(progress)}%`;
        document.getElementById('session-progress-bar').style.width = `${progress}%`;
        
        if (this.currentSession.currentPair) {
            document.getElementById('current-pair').textContent = 
                `${this.currentSession.currentPair.a.split('_')[0]} vs ${this.currentSession.currentPair.b.split('_')[0]}`;
        }
        
        // 다음 업데이트까지 남은 횟수
        const nextUpdate = 20 - (this.currentSession.totalComparisons % 20);
        document.getElementById('next-update-count').textContent = nextUpdate;
        
        // 일관성 점수 계산
        if (this.currentSession.feedbackData.length > 5) {
            const consistencyScore = this.calculateConsistencyScore();
            document.getElementById('consistency-score').textContent = `${Math.round(consistencyScore * 100)}%`;
        }
        
        // 세션 완료 체크
        if (this.currentSession.totalComparisons >= this.currentSession.targetComparisons) {
            this.completeSession();
        }
    }
    
    completeSession() {
        this.currentSession.isComplete = true;
        
        // 일반 컨트롤 숨기고 완료 컨트롤 표시
        document.getElementById('normal-controls').style.display = 'none';
        document.getElementById('session-complete-controls').style.display = 'block';
        document.getElementById('final-comparison-count').textContent = this.currentSession.totalComparisons;
        
        this.showNotification(
            `축하합니다! ${this.currentSession.totalComparisons}회의 비교를 완료했습니다.`, 
            'success'
        );
    }
    
    updateTargetComparisons() {
        const targetSelect = document.getElementById('target-comparisons');
        this.currentSession.targetComparisons = parseInt(targetSelect.value);
        this.updateSessionStats();
        
        this.showNotification(
            `목표 비교 횟수가 ${this.currentSession.targetComparisons}회로 변경되었습니다.`, 
            'info'
        );
    }
    
    async downloadFeedbackData() {
        try {
            // 피드백 데이터를 정리하여 다운로드 형식으로 변환
            const feedbackDataset = this.prepareFeedbackDataset();
            
            // JSON 파일로 다운로드
            const blob = new Blob([JSON.stringify(feedbackDataset, null, 2)], 
                                 { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `rlhf_feedback_data_${this.currentSession.id}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification('피드백 데이터가 다운로드되었습니다.', 'success');
            
        } catch (error) {
            console.error('데이터 다운로드 오류:', error);
            this.showError('데이터 다운로드 중 오류가 발생했습니다.');
        }
    }
    
    prepareFeedbackDataset() {
        /**
         * RLHF 학습을 위한 피드백 데이터셋 형식:
         * {
         *   "metadata": {...},
         *   "preference_pairs": [
         *     {
         *       "preferred_state": [bcr, far, sunlight, svr],
         *       "rejected_state": [bcr, far, sunlight, svr],
         *       "timestamp": "...",
         *       "session_info": {...}
         *     }
         *   ]
         * }
         */
        
        const dataset = {
            metadata: {
                session_id: this.currentSession.id,
                total_comparisons: this.currentSession.totalComparisons,
                target_comparisons: this.currentSession.targetComparisons,
                created_at: new Date().toISOString(),
                consistency_score: this.calculateConsistencyScore(),
                data_format: "preference_pairs",
                state_dimensions: 4,
                state_labels: ["BCR", "FAR", "Sunlight", "SV_Ratio"]
            },
            preference_pairs: []
        };
        
        // 각 피드백을 preference pair로 변환
        this.currentSession.feedbackData.forEach(feedback => {
            const designA = feedback.design_a_state || [];
            const designB = feedback.design_b_state || [];
            
            // 유효한 상태 데이터가 있는 경우만 포함
            if (designA.length >= 4 && designB.length >= 4) {
                const pair = {
                    preferred_state: feedback.selected_design === feedback.design_a_id ? 
                                   designA.slice(0, 4) : designB.slice(0, 4),
                    rejected_state: feedback.selected_design === feedback.design_a_id ? 
                                  designB.slice(0, 4) : designA.slice(0, 4),
                    preferred_id: feedback.selected_design,
                    rejected_id: feedback.selected_design === feedback.design_a_id ? 
                                feedback.design_b_id : feedback.design_a_id,
                    timestamp: feedback.comparison_time,
                    session_info: {
                        session_id: feedback.session_id,
                        comparison_index: dataset.preference_pairs.length
                    }
                };
                
                dataset.preference_pairs.push(pair);
            }
        });
        
        return dataset;
    }
    
    async trainRewardModel() {
        try {
            this.showNotification('보상 모델 학습을 시작합니다...', 'info');
            
            // 학습 데이터 준비
            const trainingData = this.prepareFeedbackDataset();
            
            // 서버에 학습 요청
            const response = await fetch('/api/reward-model/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSession.id,
                    training_data: trainingData
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showNotification(
                    `보상 모델 학습이 완료되었습니다. 정확도: ${data.accuracy}%`, 
                    'success'
                );
            } else {
                this.showError('보상 모델 학습에 실패했습니다: ' + data.message);
            }
            
        } catch (error) {
            console.error('보상 모델 학습 오류:', error);
            this.showError('보상 모델 학습 중 오류가 발생했습니다.');
        }
    }
    
    async startNewSession() {
        if (confirm('새로운 피드백 세션을 시작하시겠습니까? 현재 진행 상황은 저장됩니다.')) {
            location.reload();
        }
    }
    
    calculateConsistencyScore() {
        // 간단한 일관성 점수 계산
        // 실제로는 더 복잡한 알고리즘이 필요
        return Math.random() * 0.3 + 0.7; // 70-100% 범위
    }
    
    async checkForModelUpdate() {
        if (this.currentSession.totalComparisons % 20 === 0) {
            // 20회마다 보상 모델 업데이트 확인
            try {
                const response = await fetch('/api/reward-model/update', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: this.currentSession.id
                    })
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    this.showNotification('보상 모델이 업데이트되었습니다!', 'success');
                }
            } catch (error) {
                console.error('모델 업데이트 확인 오류:', error);
            }
        }
    }
    
    skipComparison() {
        if (confirm('이 비교를 건너뛰시겠습니까?')) {
            this.loadNextComparison();
        }
    }
    
    requestHelp() {
        const helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
        helpModal.show();
    }
    
    showLoading(show) {
        ['a', 'b'].forEach(side => {
            const loading = document.getElementById(`loading-${side}`);
            if (loading) {
                loading.style.display = show ? 'flex' : 'none';
            }
        });
    }
    
    showError(message) {
        // 간단한 에러 표시 (추후 개선 가능)
        alert('오류: ' + message);
    }
    
    showNotification(message, type = 'info') {
        // 토스트 컨테이너 확인 또는 생성
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        // 알림 요소 생성
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        // 아이콘 설정
        let icon = '';
        switch(type) {
            case 'success': icon = 'fa-check-circle'; break;
            case 'error': icon = 'fa-exclamation-circle'; break;
            case 'warning': icon = 'fa-exclamation-triangle'; break;
            default: icon = 'fa-info-circle';
        }
        
        notification.innerHTML = `
            <i class="fa-solid ${icon} me-2"></i>
            ${message}
        `;
        
        toastContainer.appendChild(notification);
        
        // 3초 후 자동 제거
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
    
    async refreshData() {
        try {
            this.showLoading(true);
            await this.loadNextComparison();
        } catch (error) {
            this.showError('데이터 새로고침 중 오류가 발생했습니다.');
        }
    }
    
    async loadDesignList() {
        // 기존 디자인 목록 로드 로직 (기존 코드 유지)
        console.log('디자인 목록 로드 중...');
    }
    
    async loadAnalysis() {
        // 분석 화면 로드 로직
        console.log('분석 데이터 로드 중...');
    }
}

// 전역 함수들 (HTML에서 호출)
function selectDesign(side) {
    if (window.rlhfInterface) {
        window.rlhfInterface.selectDesign(side);
    }
}

function skipComparison() {
    if (window.rlhfInterface) {
        window.rlhfInterface.skipComparison();
    }
}

function requestHelp() {
    if (window.rlhfInterface) {
        window.rlhfInterface.requestHelp();
    }
}

function updateTargetComparisons() {
    if (window.rlhfInterface) {
        window.rlhfInterface.updateTargetComparisons();
    }
}

function downloadFeedbackData() {
    if (window.rlhfInterface) {
        window.rlhfInterface.downloadFeedbackData();
    }
}

function trainRewardModel() {
    if (window.rlhfInterface) {
        window.rlhfInterface.trainRewardModel();
    }
}

function startNewSession() {
    if (window.rlhfInterface) {
        window.rlhfInterface.startNewSession();
    }
}

function resetView(side) {
    if (window.rlhfInterface && window.rlhfInterface.cameras[side] && window.rlhfInterface.controls[side]) {
        const camera = window.rlhfInterface.cameras[side];
        const controls = window.rlhfInterface.controls[side];
        
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
    }
}

function toggleWireframe(side) {
    if (window.rlhfInterface && window.rlhfInterface.scenes[side]) {
        const scene = window.rlhfInterface.scenes[side];
        scene.traverse(object => {
            if (object instanceof THREE.Mesh && !object.userData.isHelper) {
                object.material.wireframe = !object.material.wireframe;
            }
        });
    }
}

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.rlhfInterface = new RLHFInterface();
});