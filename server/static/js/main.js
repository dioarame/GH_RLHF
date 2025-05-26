/**
 * RLHF 인간 피드백 시스템 - 메인 애플리케이션
 */

function showNotification(message, type = 'info') {
    // 간단한 알림 표시
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'success' ? 'success' : 'info'} alert-dismissible`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    // 페이드 인 효과
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 100);
    
    // 3초 후 자동 제거
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }
    }, 3000);
}

class RLHFSystem {
    constructor() {
        this.currentDesigns = { a: null, b: null };
        this.scenes = { a: null, b: null };
        this.cameras = { a: null, b: null };
        this.renderers = { a: null, b: null };
        this.controls = { a: null, b: null };
        this.environmentData = { contour: null, surface: null }; // 환경 데이터
        this.viewerMode = { a: 'light', b: 'light' }; // 뷰어 모드 (light/dark)
        this.systemTheme = 'light'; // 시스템 전체 테마
        this.selectionHistory = []; // 사용자 선택 히스토리
        this.designStats = null; // 디자인 통계 정보
        this.sessionStats = { 
            total_comparisons: 0, 
            target_comparisons: 100 
        };
        
        this.init();
    }
    
    async init() {
        console.log('RLHF 시스템 초기화 중...');
        
        try {
            // 환경 데이터 로드
            await this.loadEnvironmentData();
            this.initViewers();
            
            // 목표 설정 모달 표시
            await this.showTargetSetupModal();
            
            console.log('초기화 완료');
        } catch (error) {
            console.error('초기화 오류:', error);
        }
    }
    
    async showTargetSetupModal() {
        try {
            // Bootstrap 모달 표시 (안전하게)
            const modalElement = document.getElementById('targetSetupModal');
            
            if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
                const modal = new bootstrap.Modal(modalElement);
                modal.show();
            } else {
                // Bootstrap이 없으면 수동으로 표시
                modalElement.style.display = 'block';
                modalElement.classList.add('show');
                document.body.classList.add('modal-open');
                
                // 배경 오버레이 추가
                const backdrop = document.createElement('div');
                backdrop.className = 'modal-backdrop fade show';
                document.body.appendChild(backdrop);
            }
            
            // 디자인 분석 수행
            await this.analyzeDesigns();
        } catch (error) {
            console.error('모달 표시 오류:', error);
            // 모달 없이도 분석 진행
            await this.analyzeDesigns();
        }
    }
    
    async analyzeDesigns() {
        try {
            console.log('디자인 데이터 분석 시작...');
            
            const response = await fetch('/api/designs/stats');
            console.log('API 응답 상태:', response.status);
            
            if (response.ok) {
                const data = await response.json();
                console.log('API 응답 데이터:', data);
                
                if (data.status === 'success') {
                    this.designStats = data.stats;
                    this.updateDesignAnalysisUI();
                    this.setSmartDefaultTarget();
                } else {
                    console.error('API 응답 오류:', data.message);
                    this.showAnalysisError('API 응답 오류: ' + data.message);
                }
            } else {
                console.error('HTTP 오류:', response.status);
                this.showAnalysisError('서버 연결 오류: ' + response.status);
            }
        } catch (error) {
            console.error('디자인 분석 오류:', error);
            this.showAnalysisError('네트워크 오류: ' + error.message);
        }
    }
    
    showAnalysisError(errorMessage) {
        // 오류 발생 시 기본값으로 진행
        document.getElementById('modal-loading').innerHTML = `
            <div class="text-center py-4">
                <i class="fa-solid fa-exclamation-triangle text-warning mb-3" style="font-size: 2rem;"></i>
                <div class="text-danger mb-2">분석 오류 발생</div>
                <small class="text-muted">${errorMessage}</small>
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="rlhfSystem.useDefaultSettings()">
                        기본 설정으로 계속
                    </button>
                </div>
            </div>
        `;
    }
    
    useDefaultSettings() {
        // 기본값으로 설정
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
        
        // 모달 내 통계 정보 업데이트
        document.getElementById('modal-top-count').textContent = stats.top_designs;
        document.getElementById('modal-random-count').textContent = stats.random_designs;
        document.getElementById('modal-total-designs').textContent = stats.total_designs;
        document.getElementById('modal-max-pairs').textContent = stats.max_comparisons;
        
        // 슬라이더 범위 업데이트
        const maxTarget = Math.min(stats.max_comparisons, 500);
        const slider = document.getElementById('modal-target-slider');
        slider.max = maxTarget;
        
        // 스마트 버튼 값들 계산 및 업데이트
        this.updateSmartButtons();
        
        // 로딩 숨기고 콘텐츠 표시
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
        
        // 추천값을 기본으로 설정
        const slider = document.getElementById('modal-target-slider');
        slider.value = recommendedTarget;
        updateModalTarget(); // this. 제거
    }
    
    setSmartDefaultTarget() {
        if (!this.designStats) return;
        
        const totalDesigns = this.designStats.total_designs;
        const maxComparisons = this.designStats.max_comparisons;
        
        // 스마트 기본값 계산
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
        
        // UI 업데이트
        const slider = document.getElementById('target-slider');
        const input = document.getElementById('target-input');
        const currentTarget = document.getElementById('current-target');
        
        if (slider) slider.value = recommendedTarget;
        if (input) input.value = recommendedTarget;
        if (currentTarget) currentTarget.textContent = recommendedTarget;
        
        this.updateStats();
        
        console.log(`스마트 기본 목표 설정: ${recommendedTarget}회 (총 디자인: ${totalDesigns}개)`);
    }
    
    async loadEnvironmentData() {
        try {
            console.log('환경 데이터 로딩 중...');
            
            // Contour.json 로드
            const contourResponse = await fetch('/data/environment/Contour.json');
            if (contourResponse.ok) {
                this.environmentData.contour = await contourResponse.json();
                console.log('Contour 데이터 로드 완료');
            }
            
            // Sur.json 로드
            const surfaceResponse = await fetch('/data/environment/Sur.json');
            if (surfaceResponse.ok) {
                this.environmentData.surface = await surfaceResponse.json();
                console.log('Surface 데이터 로드 완료');
            }
            
        } catch (error) {
            console.warn('환경 데이터 로드 실패:', error);
        }
    }
    
    loadEnvironmentMeshes(side) {
        const scene = this.scenes[side];
        
        // Contour 메시 로드 (진한 회색)
        if (this.environmentData.contour) {
            this.createEnvironmentMesh(scene, this.environmentData.contour, 0x555555, 'contour');
        }
        
        // Surface 메시 로드 (파스텔 갈색)
        if (this.environmentData.surface) {
            this.createEnvironmentMesh(scene, this.environmentData.surface, 0xD2B48C, 'surface');
        }
    }
    
    createEnvironmentMesh(scene, meshData, color, type) {
        try {
            if (!meshData.meshes || !Array.isArray(meshData.meshes)) return;
            
            meshData.meshes.forEach(meshInfo => {
                if (!meshInfo.vertices || !meshInfo.faces) return;
                
                const geometry = new THREE.BufferGeometry();
                
                // 정점 설정 (Grasshopper → Three.js 좌표계 변환)
                const vertices = [];
                for (let i = 0; i < meshInfo.vertices.length; i++) {
                    const vertex = meshInfo.vertices[i];
                    // Rhino/Grasshopper는 Z-up, Three.js는 Y-up
                    // 변환 시도: X → X, Z → Y, Y → Z (Z와 Y를 바꿈)
                    vertices.push(vertex[0], vertex[2], vertex[1]);
                }
                geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
                
                // 면 설정
                const indices = [];
                meshInfo.faces.forEach(face => {
                    if (face.length === 3) {
                        indices.push(face[0], face[1], face[2]);
                    } else if (face.length === 4) {
                        indices.push(face[0], face[1], face[2]);
                        indices.push(face[0], face[2], face[3]);
                    }
                });
                
                if (indices.length > 0) {
                    geometry.setIndex(indices);
                }
                
                geometry.computeVertexNormals();
                
                // 재질 생성 (환경에 맞는 색상)
                const material = new THREE.MeshLambertMaterial({
                    color: color,
                    transparent: type === 'surface',
                    opacity: type === 'surface' ? 0.6 : 1.0,
                    side: THREE.DoubleSide
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.isEnvironmentMesh = true;
                mesh.userData.envType = type;
                mesh.receiveShadow = true; // 환경 메시는 그림자를 받기만 함
                
                // 환경 메시를 X축 기준으로 뒤집기 (X-Z 평면에서 뒤집기)
                mesh.scale.set(1, 1, -1); // Z축을 뒤집음
                
                scene.add(mesh);
            });
            
        } catch (error) {
            console.error(`${type} 환경 메시 생성 오류:`, error);
        }
    }
    
    initViewers() {
        ['a', 'b'].forEach(side => {
            try {
                this.initSingleViewer(side);
            } catch (error) {
                console.error(`뷰어 ${side} 초기화 오류:`, error);
            }
        });
        
        // 창 크기 변경 이벤트
        window.addEventListener('resize', () => this.handleResize());
    }
    
    initSingleViewer(side) {
        const container = document.getElementById(`viewer-${side}`);
        if (!container) {
            console.error(`컨테이너 ${side}를 찾을 수 없습니다`);
            return;
        }
        
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        // Scene 생성
        this.scenes[side] = new THREE.Scene();
        // 라이트 모드 배경 (기본값)
        this.updateViewerBackground(side, 'light');
        
        // Camera 생성
        this.cameras[side] = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        // 초기 카메라 위치를 Perspective 뷰로 설정
        this.cameras[side].position.set(5, 3, 5);
        this.cameras[side].lookAt(0, 1, 0); // 약간 위쪽을 바라보도록
        
        // Renderer 생성
        this.renderers[side] = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance", // RTX 3090 활용
            precision: "highp"
        });
        this.renderers[side].setSize(width, height);
        this.renderers[side].setPixelRatio(window.devicePixelRatio);
        // 그림자 활성화
        this.renderers[side].shadowMap.enabled = true;
        this.renderers[side].shadowMap.type = THREE.PCFSoftShadowMap;
        // 톤 매핑 설정
        this.renderers[side].toneMapping = THREE.ACESFilmicToneMapping;
        this.renderers[side].toneMappingExposure = 1.2;
        
        // Controls 생성
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls[side] = new THREE.OrbitControls(
                this.cameras[side], 
                this.renderers[side].domElement
            );
            this.controls[side].enableDamping = true;
            this.controls[side].dampingFactor = 0.05;
            
            // 미터 단위 모델을 위한 컨트롤 조정
            this.controls[side].rotateSpeed = 0.5; // 회전 속도 감소
            this.controls[side].zoomSpeed = 0.5; // 줌 속도 적당히 조정
            this.controls[side].panSpeed = 0.3; // 팬 속도 감소
            this.controls[side].minDistance = 0.1; // 최소 거리
            this.controls[side].maxDistance = 1000; // 최대 거리 크게 증가
        } else {
            console.warn('OrbitControls를 사용할 수 없습니다');
            this.controls[side] = {
                update: () => {},
                dispose: () => {}
            };
        }
        
        // 조명 설정
        this.setupLights(side);
        
        // DOM에 추가
        const loadingOverlay = container.querySelector('.loading-overlay');
        if (loadingOverlay) {
            container.insertBefore(this.renderers[side].domElement, loadingOverlay);
        } else {
            container.appendChild(this.renderers[side].domElement);
        }
        
        // 애니메이션 루프 시작
        this.startAnimation(side);
        
        console.log(`뷰어 ${side.toUpperCase()} 초기화 완료`);
    }
    
    updateViewerBackground(side, mode) {
        this.viewerMode[side] = mode;
        const scene = this.scenes[side];
        
        if (mode === 'light') {
            scene.background = new THREE.Color(0x87CEEB); // 밝은 하늘색
        } else {
            scene.background = new THREE.Color(0x1a1a1a); // 어두운 회색
        }
    }
    
    toggleViewerMode(side) {
        const currentMode = this.viewerMode[side];
        const newMode = currentMode === 'light' ? 'dark' : 'light';
        this.updateViewerBackground(side, newMode);
        
        // 그리드 헬퍼 색상도 업데이트
        this.updateGridColors(side, newMode);
    }
    
    updateGridColors(side, mode) {
        const scene = this.scenes[side];
        scene.traverse(object => {
            if (object.isGridHelper) {
                scene.remove(object);
                object.dispose();
            }
        });
        
        // 새로운 그리드 추가
        if (mode === 'light') {
            const gridHelper = new THREE.GridHelper(20, 40, 0x555555, 0x333333);
            gridHelper.isGridHelper = true;
            scene.add(gridHelper);
        } else {
            const gridHelper = new THREE.GridHelper(20, 40, 0x888888, 0x666666);
            gridHelper.isGridHelper = true;
            scene.add(gridHelper);
        }
    }
    
    setupLights(side) {
        const scene = this.scenes[side];
        
        // 주변광 (약간 밝게, 따뜻한 톤)
        const ambientLight = new THREE.AmbientLight(0xfff5e6, 0.5);
        scene.add(ambientLight);
        
        // 반구광 (하늘색과 지면색)
        const hemisphereLight = new THREE.HemisphereLight(
            0x87CEEB, // 하늘색
            0x8B7355, // 지면색 (갈색)
            0.4
        );
        scene.add(hemisphereLight);
        
        // 태양광 (주 방향성 조명 - 그림자 생성)
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
        sunLight.position.set(15, 20, 10);
        sunLight.castShadow = true;
        
        // 고품질 그림자 설정 (RTX 3090용)
        sunLight.shadow.mapSize.width = 4096;
        sunLight.shadow.mapSize.height = 4096;
        sunLight.shadow.camera.near = 0.1;
        sunLight.shadow.camera.far = 100;
        sunLight.shadow.camera.left = -20;
        sunLight.shadow.camera.right = 20;
        sunLight.shadow.camera.top = 20;
        sunLight.shadow.camera.bottom = -20;
        
        // 부드러운 그림자를 위한 반경 설정
        sunLight.shadow.radius = 4;
        sunLight.shadow.blurSamples = 25;
        
        scene.add(sunLight);
        
        // 보조 조명 1 (북쪽에서 오는 부드러운 빛)
        const fillLight1 = new THREE.DirectionalLight(0xe6f2ff, 0.4);
        fillLight1.position.set(-10, 15, -10);
        scene.add(fillLight1);
        
        // 보조 조명 2 (측면 강조)
        const fillLight2 = new THREE.DirectionalLight(0xfff0e6, 0.3);
        fillLight2.position.set(10, 5, -5);
        scene.add(fillLight2);
        
        // 포인트 라이트 (근처 환경 조명 시뮬레이션)
        const pointLight = new THREE.PointLight(0xffffff, 0.2, 30);
        pointLight.position.set(0, 10, 0);
        scene.add(pointLight);
        
        // 그리드 헬퍼 (더 세밀하게)
        const gridHelper = new THREE.GridHelper(20, 40, 0x555555, 0x333333);
        gridHelper.isGridHelper = true;
        scene.add(gridHelper);
        
        // 축 헬퍼
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);
    }
    
    startAnimation(side) {
        const animate = () => {
            requestAnimationFrame(animate);
            
            if (this.controls[side]) {
                this.controls[side].update();
            }
            
            if (this.renderers[side] && this.scenes[side] && this.cameras[side]) {
                this.renderers[side].render(this.scenes[side], this.cameras[side]);
            }
        };
        animate();
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
    
    async loadNextComparison() {
        try {
            this.showLoading(true);
            
            // 실제 API에서 비교 쌍 가져오기
            const response = await fetch('/api/comparison/next', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: 'rlhf_session' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
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
        
        // 메시 로드
        await Promise.all([
            this.loadMesh('a', designA),
            this.loadMesh('b', designB)
        ]);
        
        // 메트릭 업데이트
        this.updateMetrics('a', designA);
        this.updateMetrics('b', designB);
        
        // 양쪽 뷰어 모두 Perspective 뷰로 설정
        setTimeout(() => {
            this.setInitialPerspectiveView('a');
            this.setInitialPerspectiveView('b');
        }, 100); // 메시 로딩이 완전히 끝난 후 실행
        
        // 현재 비교 쌍 업데이트 (개선된 형식)
        const designALabel = this.formatDesignLabel(designA.id);
        const designBLabel = this.formatDesignLabel(designB.id);
        document.getElementById('current-pair').textContent = `${designALabel} vs ${designBLabel}`;
    }
    
    setInitialPerspectiveView(side) {
        // setViewport 함수를 직접 호출
        if (window.setViewport) {
            setViewport(side, 'perspective');
            console.log(`${side} 뷰어에 초기 Perspective 뷰 적용`);
        }
    }
    
    formatDesignLabel(designId) {
        // 디자인 ID에서 카테고리와 번호 추출
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
            // 일반적인 경우
            const parts = designId.split('_');
            if (parts.length >= 2) {
                return parts[0].toUpperCase() + ' ' + (parts[1] || '00').padStart(2, '0');
            }
            return designId.toUpperCase();
        }
    }
    
    async loadMesh(side, design) {
        try {
            // 기존 디자인 메시 제거 (환경 메시는 유지)
            this.clearDesignMesh(side);
            
            // 환경 메시 로드 (처음에만)
            this.loadEnvironmentMeshes(side);
            
            // 실제 메시 데이터가 있으면 API에서 로드 시도
            if (design.id && !design.id.startsWith('fallback_')) {
                const response = await fetch(`/api/mesh/${design.id}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'success' && data.mesh) {
                        this.createMeshFromData(side, data.mesh);
                        // 실제 메시 로드 후 Perspective 뷰 적용
                        setTimeout(() => {
                            if (window.setViewport) {
                                setViewport(side, 'perspective');
                            }
                        }, 50);
                        return;
                    }
                }
            }
            
            // 대체: 기본 큐브 생성
            this.createDefaultMesh(side);
            // 기본 큐브에 대해서도 Perspective 뷰 적용
            setTimeout(() => {
                if (window.setViewport) {
                    setViewport(side, 'perspective');
                }
            }, 50);
            
        } catch (error) {
            console.error(`메시 로드 오류 (${side}):`, error);
            this.createDefaultMesh(side);
            // 오류 시에도 Perspective 뷰 적용
            setTimeout(() => {
                if (window.setViewport) {
                    setViewport(side, 'perspective');
                }
            }, 50);
        }
    }
    
    createMeshFromData(side, meshData) {
        try {
            if (!meshData.meshes || !Array.isArray(meshData.meshes)) {
                this.createDefaultMesh(side);
                return;
            }
            
            const allMeshes = [];
            
            // 파스텔 색상 팔레트 (입면용)
            const pastelColors = [
                0xFFE4E1, // 미스티 로즈
                0xE6E6FA, // 라벤더
                0xF0E68C, // 카키
                0xFFDAB9, // 피치 퍼프
                0xB0E0E6, // 파우더 블루
                0xD8BFD8, // 시슬
                0xF5DEB3, // 위트
                0xFAF0E6, // 리넨
                0xFFE4B5, // 모카신
                0xF0FFFF  // 아주르
            ];
            
            meshData.meshes.forEach((meshInfo, meshIndex) => {
                if (!meshInfo.vertices || !meshInfo.faces) return;
                
                const geometry = new THREE.BufferGeometry();
                
                // 정점 설정 (좌표계 변환 적용)
                const vertices = [];
                for (let i = 0; i < meshInfo.vertices.length; i++) {
                    const vertex = meshInfo.vertices[i];
                    // 환경 메시와 완전히 동일한 변환 적용
                    // Rhino/Grasshopper는 Z-up, Three.js는 Y-up
                    // 변환 시도: X → X, Z → Y, Y → Z (Z와 Y를 바꿈)
                    vertices.push(vertex[0], vertex[2], vertex[1]);
                }
                geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
                
                // 면 설정
                const indices = [];
                meshInfo.faces.forEach(face => {
                    if (face.length === 3) {
                        indices.push(face[0], face[1], face[2]);
                    } else if (face.length === 4) {
                        indices.push(face[0], face[1], face[2]);
                        indices.push(face[0], face[2], face[3]);
                    }
                });
                
                if (indices.length > 0) {
                    geometry.setIndex(indices);
                }
                
                geometry.computeVertexNormals();
                
                // 면의 방향을 분석하여 수평/수직 판단
                const normals = geometry.getAttribute('normal').array;
                let horizontalScore = 0;
                let verticalScore = 0;
                
                // 모든 정점의 법선 벡터를 분석
                for (let i = 0; i < normals.length; i += 3) {
                    const nx = Math.abs(normals[i]);
                    const ny = Math.abs(normals[i + 1]);
                    const nz = Math.abs(normals[i + 2]);
                    
                    // Y축이 위쪽을 향하는 Three.js 좌표계에서
                    // ny가 크면 수평면(천장/바닥), nx나 nz가 크면 수직면(벽)
                    if (ny > 0.7) {
                        horizontalScore++;
                    } else if (nx > 0.5 || nz > 0.5) {
                        verticalScore++;
                    }
                }
                
                const isHorizontal = horizontalScore > verticalScore;
                
                // 재질 생성
                let material;
                if (isHorizontal) {
                    // 수평면(천장/바닥): 짙은 갈색/검정 계열
                    const darkColors = [0x2C1810, 0x1A0E05, 0x0D0D0D, 0x1C1C1C, 0x2F2519];
                    const selectedColor = darkColors[meshIndex % darkColors.length];
                    
                    material = new THREE.MeshPhysicalMaterial({
                        color: selectedColor,
                        roughness: 0.9,
                        metalness: 0.1,
                        clearcoat: 0.3,
                        clearcoatRoughness: 0.7,
                        side: THREE.DoubleSide
                    });
                } else {
                    // 수직면(입면): 파스텔 톤 랜덤
                    const selectedColor = pastelColors[meshIndex % pastelColors.length];
                    
                    material = new THREE.MeshPhysicalMaterial({
                        color: selectedColor,
                        roughness: 0.3,
                        metalness: 0.0,
                        clearcoat: 0.5,
                        clearcoatRoughness: 0.3,
                        transmission: 0.1, // 약간의 투명도
                        thickness: 0.5,
                        side: THREE.DoubleSide
                    });
                }
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.isDesignMesh = true;
                mesh.userData.isHorizontal = isHorizontal;
                mesh.castShadow = true; // 그림자 생성
                mesh.receiveShadow = true; // 그림자 받기
                
                // 환경 메시와 동일하게 X축 기준으로 뒤집기 (X-Z 평면에서 뒤집기)
                mesh.scale.set(1, 1, -1); // Z축을 뒤집음
                
                this.scenes[side].add(mesh);
                allMeshes.push(mesh);
            });
            
            // 메시들을 바닥 기준으로 정렬
            if (allMeshes.length > 0) {
                // centerMeshes 호출 제거 - 원본 디자인의 원점 유지
                console.log(`${side} 메시 로드 완료: ${allMeshes.length}개 메시`);
            }
            
        } catch (error) {
            console.error('메시 데이터 처리 오류:', error);
            this.createDefaultMesh(side);
        }
    }
    
    centerMeshes(meshes) {
        if (!meshes || meshes.length === 0) return;
        
        // 모든 메시의 경계 상자 계산
        const box = new THREE.Box3();
        meshes.forEach(mesh => {
            box.expandByObject(mesh);
        });
        
        // 중심점 계산
        const center = new THREE.Vector3();
        box.getCenter(center);
        
        // 모든 메시를 중심점으로 이동
        meshes.forEach(mesh => {
            mesh.position.sub(center);
        });
    }
    
    zoomToFitAll(side) {
        console.log(`${side} 뷰어 줌 조정 시작`);
        
        // 디자인 메시만 대상으로 줌 조정 (환경 메시 제외)
        const designMeshes = [];
        this.scenes[side].traverse(object => {
            if (object.userData.isDesignMesh) {
                designMeshes.push(object);
            }
        });
        
        if (designMeshes.length === 0) {
            console.log(`${side} 뷰어에 디자인 메시가 없음`);
            return;
        }
        
        console.log(`${side} 뷰어에서 ${designMeshes.length}개 메시 발견`);
        
        // 디자인 메시들의 경계 상자 계산
        const box = new THREE.Box3();
        designMeshes.forEach(mesh => {
            box.expandByObject(mesh);
        });
        
        // 크기 계산
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        
        console.log(`${side} 메시 크기:`, {
            x: size.x.toFixed(2),
            y: size.y.toFixed(2),
            z: size.z.toFixed(2),
            maxDim: maxDim.toFixed(2)
        });
        
        if (maxDim > 0) {
            const camera = this.cameras[side];
            const fov = camera.fov * (Math.PI / 180);
            let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 2.5; // 여유 더 크게
            
            // 건축물 관찰에 적합한 각도로 카메라 위치 설정
            const buildingHeight = size.z;
            const cameraHeight = Math.max(buildingHeight * 0.4, cameraDistance * 0.3);
            
            // 초기 로딩 시 Perspective 뷰로 설정
            const newCameraPos = {
                x: cameraDistance * 0.7,
                y: cameraHeight,
                z: cameraDistance * 0.7
            };
            
            camera.position.set(newCameraPos.x, newCameraPos.y, newCameraPos.z);
            
            // 건물 중간 높이 정도를 바라보도록 설정
            const lookAtHeight = buildingHeight * 0.3;
            camera.lookAt(0, lookAtHeight, 0);
            
            console.log(`${side} 카메라 위치 (Perspective 뷰):`, {
                position: `${newCameraPos.x.toFixed(2)}, ${newCameraPos.y.toFixed(2)}, ${newCameraPos.z.toFixed(2)}`,
                lookAt: `0, ${lookAtHeight.toFixed(2)}, 0`,
                distance: cameraDistance.toFixed(2)
            });
            
            if (this.controls[side] && this.controls[side].target) {
                this.controls[side].target.set(0, lookAtHeight, 0);
                this.controls[side].update();
            }
        }
    }
    
    createDefaultMesh(side) {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshPhysicalMaterial({
            color: 0xBFBFBF,
            roughness: 0.5,
            metalness: 0.1,
            clearcoat: 0.3,
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.isDesignMesh = true;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // 큐브를 바닥에 위치시키기 (큐브 바닥이 Y=0이 되도록)
        mesh.position.set(0, 0.5, 0); // 큐브 높이의 절반만큼 위로
        
        this.scenes[side].add(mesh);
        console.log(`${side} 기본 큐브 생성됨 (바닥 기준 위치)`);
    }
    
    clearDesignMesh(side) {
        if (!this.scenes[side]) return;
        
        const meshesToRemove = [];
        this.scenes[side].traverse(object => {
            if (object.userData.isDesignMesh) {
                meshesToRemove.push(object);
            }
        });
        
        meshesToRemove.forEach(mesh => {
            this.scenes[side].remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });
    }
    
    clearMesh(side) {
        if (!this.scenes[side]) return;
        
        const meshesToRemove = [];
        this.scenes[side].traverse(object => {
            if (object.userData.isDesignMesh || object.userData.isEnvironmentMesh) {
                meshesToRemove.push(object);
            }
        });
        
        meshesToRemove.forEach(mesh => {
            this.scenes[side].remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });
    }
    
    updateMetrics(side, design) {
        const state = design.state || [0, 0, 0, 0];
        const reward = design.reward || 0;
        
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
        
        // 보상값 표시 추가
        this.updateRewardDisplay(side, reward);
    }
    
    updateRewardDisplay(side, reward) {
        // 보상값 표시를 위한 새로운 함수
        const container = document.getElementById(`viewer-${side}`);
        let rewardDisplay = container.querySelector('.reward-display');
        
        if (!rewardDisplay) {
            rewardDisplay = document.createElement('div');
            rewardDisplay.className = 'reward-display';
            container.appendChild(rewardDisplay);
        }
        
        // 보상값에 따른 색상 결정
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
    
    async selectDesign(side) {
        try {
            const designA = this.currentDesigns.a;
            const designB = this.currentDesigns.b;
            
            if (!designA || !designB) return;
            
            // 선택 효과
            this.showSelectionEffect(side);
            
            // 피드백 데이터 구성
            const feedbackData = {
                session_id: 'rlhf_session',
                design_a_id: designA.id,
                design_b_id: designB.id,
                selected_design: side === 'a' ? designA.id : designB.id,
                design_a_state: designA.state,
                design_b_state: designB.state,
                timestamp: Date.now()
            };
            
            // 서버에 피드백 전송
            const response = await fetch('/api/feedback/comparison', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success') {
                    this.sessionStats.total_comparisons++;
                    
                    // 선택 히스토리에 추가
                    const selectedDesign = side === 'a' ? designA : designB;
                    const notSelectedDesign = side === 'a' ? designB : designA;
                    this.selectionHistory.push({
                        selected: selectedDesign,
                        notSelected: notSelectedDesign,
                        timestamp: Date.now()
                    });
                    
                    this.updateStats();
                    this.analyzeSelectionTendency(); // 경향성 분석
                    
                    // 다음 비교로 이동
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
        
        // 주요 선호 메트릭 찾기
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
        
        // 일관성 검사 (최근 3개 선택의 유사성)
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
        
        console.log('선택 경향성 분석:', {
            count: historyCount,
            metrics: selectedMetrics,
            tendency: tendency
        });
    }
    
    calculateConsistency(recentSelections) {
        if (recentSelections.length < 2) return 0;
        
        let consistencyScore = 0;
        const metrics = ['bcr', 'far', 'sunlight', 'svr'];
        
        for (let i = 0; i < recentSelections.length - 1; i++) {
            const current = recentSelections[i].selected.state || [0, 0, 0, 0];
            const next = recentSelections[i + 1].selected.state || [0, 0, 0, 0];
            
            let similarity = 0;
            for (let j = 0; j < 4; j++) {
                const diff = Math.abs(current[j] - next[j]);
                const range = Math.max(current[j], next[j]) - Math.min(current[j], next[j]);
                similarity += range > 0 ? 1 - (diff / Math.max(current[j], next[j])) : 1;
            }
            consistencyScore += similarity / 4;
        }
        
        return consistencyScore / (recentSelections.length - 1);
    }
    
    showLoading(show) {
        ['a', 'b'].forEach(side => {
            const loading = document.getElementById(`loading-${side}`);
            if (loading) {
                loading.style.display = show ? 'flex' : 'none';
            }
        });
    }
}

// 전역 함수들
let rlhfSystem;

function selectDesign(side) {
    if (rlhfSystem) rlhfSystem.selectDesign(side);
}

function skipComparison() {
    if (rlhfSystem) rlhfSystem.loadNextComparison();
}

function loadNextComparison() {
    if (rlhfSystem) rlhfSystem.loadNextComparison();
}

function showHelp() {
    const helpMessage = `
🏗️ CAD 데이터 기반 RLHF 시스템 사용법

📋 기본 사용법:
• 두 디자인 중 더 선호하는 디자인을 선택하세요
• 건축 지표(건폐율, 용적률, 일사량, SV Ratio)를 참고하여 판단

⚙️ 목표 설정:
• 슬라이더 또는 숫자 입력으로 목표 비교 횟수 조정 (10~500회)
• 빠름(50회) / 보통(100회) / 긴(200회) 버튼으로 빠른 설정
• 진행 중에도 목표 변경 가능

🎮 뷰어 조작법:
• 마우스 왼쪽 버튼 + 드래그: 회전
• 마우스 휠: 확대/축소
• 마우스 오른쪽 버튼 + 드래그: 이동

🔧 뷰어 버튼:
• ⬜ Top 뷰: 위에서 내려다보기
• ⏹️ Front 뷰: 정면에서 보기  
• 🧊 Perspective 뷰: 3D 관찰 각도
• 🏠 뷰 리셋: ZoomSelected (전체 보기)
• 📐 와이어프레임: 메시 윤곽선 보기
• 💡 뷰어 배경 모드: 개별 뷰어 배경 변경
• 📷 스크린샷 저장

🎨 테마:
• 헤더 오른쪽 버튼: 시스템 전체 라이트/다크 테마
• F 키: 양쪽 뷰어 배경 동시 전환

💡 팁:
• 미터 단위 모델로 작은 크기일 수 있으니 천천히 조작하세요
• Top/Front 뷰로 좌표계 정렬을 쉽게 확인할 수 있습니다
• 선택의 경향성은 3회 이상 선택 후부터 분석됩니다
    `;
    alert(helpMessage);
}

// 뷰어 컨트롤 함수들
function resetView(side) {
    if (rlhfSystem && rlhfSystem.cameras[side] && rlhfSystem.controls[side]) {
        // ZoomSelected 유사 기능 - 디자인에 맞춰 줌
        rlhfSystem.zoomToFitAll(side);
    }
}

function toggleWireframe(side) {
    if (rlhfSystem && rlhfSystem.scenes[side]) {
        const scene = rlhfSystem.scenes[side];
        scene.traverse(object => {
            if (object.userData.isDesignMesh && object.material) {
                object.material.wireframe = !object.material.wireframe;
            }
        });
    }
}

function captureView(side) {
    if (rlhfSystem && rlhfSystem.renderers[side]) {
        const canvas = rlhfSystem.renderers[side].domElement;
        const dataUrl = canvas.toDataURL('image/png');
        
        // 다운로드 링크 생성
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = `design_${side}_${Date.now()}.png`;
        link.click();
        
        // 성공 메시지 표시
        showNotification(`디자인 ${side.toUpperCase()} 스크린샷이 저장되었습니다.`, 'success');
    }
}

function toggleSystemTheme() {
    if (!rlhfSystem) return;
    
    const newTheme = rlhfSystem.systemTheme === 'light' ? 'dark' : 'light';
    rlhfSystem.systemTheme = newTheme;
    
    // HTML 문서에 테마 클래스 적용
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // 아이콘 변경
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        if (newTheme === 'dark') {
            themeIcon.className = 'fa-solid fa-moon';
        } else {
            themeIcon.className = 'fa-solid fa-sun';
        }
    }
    
    console.log(`시스템 테마를 ${newTheme} 모드로 변경`);
}

function setViewport(side, viewType) {
    if (!rlhfSystem || !rlhfSystem.cameras[side]) return;
    
    const camera = rlhfSystem.cameras[side];
    const controls = rlhfSystem.controls[side];
    
    // 디자인 메시들의 경계 상자를 구해서 적절한 거리 계산
    const designMeshes = [];
    rlhfSystem.scenes[side].traverse(object => {
        if (object.userData.isDesignMesh) {
            designMeshes.push(object);
        }
    });
    
    let distance = 10; // 기본 거리
    let center = new THREE.Vector3(0, 0, 0);
    
    if (designMeshes.length > 0) {
        const box = new THREE.Box3();
        designMeshes.forEach(mesh => {
            box.expandByObject(mesh);
        });
        
        const size = new THREE.Vector3();
        box.getSize(size);
        distance = Math.max(size.x, size.y, size.z) * 2;
        box.getCenter(center);
    }
    
    // 뷰포트별 카메라 위치 설정
    switch (viewType.toLowerCase()) {
        case 'top':
            // Top 뷰: 위에서 아래로 내려다보기
            camera.position.set(center.x, center.y + distance, center.z);
            camera.lookAt(center);
            break;
            
        case 'front':
            // Front 뷰: 남쪽(앞)에서 북쪽(뒤)으로 보기 (Z축 양의 방향에서)
            camera.position.set(center.x, center.y, center.z + distance);
            camera.lookAt(center);
            break;
            
        case 'perspective':
            // Perspective 뷰: 3D 관찰 각도
            camera.position.set(
                center.x + distance * 0.7, 
                center.y + distance * 0.5, 
                center.z + distance * 0.7
            );
            camera.lookAt(center);
            break;
            
        default:
            console.warn(`알 수 없는 뷰 타입: ${viewType}`);
            return;
    }
    
    // 컨트롤 타겟 업데이트
    if (controls && controls.target) {
        controls.target.copy(center);
        controls.update();
    }
    
    console.log(`${side} 뷰어를 ${viewType} 뷰로 설정함`);
}

function toggleViewerMode(side) {
    if (rlhfSystem) {
        rlhfSystem.toggleViewerMode(side);
    }
}

// 키보드 단축키
document.addEventListener('keydown', (event) => {
    if (event.key === 'f' || event.key === 'F') {
        // F 키로 양쪽 뷰어 모드 동시 전환
        // 현재 A 뷰어의 모드를 확인해서 반대로 설정
        if (rlhfSystem) {
            const currentModeA = rlhfSystem.viewerMode.a;
            const newMode = currentModeA === 'light' ? 'dark' : 'light';
            
            // 양쪽 뷰어를 같은 모드로 설정
            rlhfSystem.updateViewerBackground('a', newMode);
            rlhfSystem.updateViewerBackground('b', newMode);
            
            console.log(`F키: 양쪽 뷰어를 ${newMode} 모드로 전환`);
        }
    }
});

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    rlhfSystem = new RLHFSystem();
});

// 모달 관련 함수들
function updateModalTarget() {
    const slider = document.getElementById('modal-target-slider');
    const display = document.getElementById('modal-target-display');
    const info = document.getElementById('modal-target-info');
    const time = document.getElementById('estimated-time');
    
    const value = parseInt(slider.value);
    display.textContent = value;
    
    if (rlhfSystem && rlhfSystem.designStats) {
        const percentage = Math.round((value / rlhfSystem.designStats.max_comparisons) * 100);
        info.textContent = `전체 비교의 ${percentage}%`;
    }
    
    const estimatedMinutes = Math.ceil(value / 4);
    time.textContent = `${estimatedMinutes}분`;
}

function setModalTarget(type) {
    if (!rlhfSystem || !rlhfSystem.designStats) return;
    
    const maxComparisons = Math.min(rlhfSystem.designStats.max_comparisons, 500);
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
    updateModalTarget();
}

function startSession() {
    const slider = document.getElementById('modal-target-slider');
    const targetValue = parseInt(slider.value);
    
    // 목표값 설정
    if (rlhfSystem) {
        rlhfSystem.sessionStats.target_comparisons = targetValue;
        rlhfSystem.updateStats();
    }
    
    // 모달 닫기 (여러 방법으로 시도)
    const modalElement = document.getElementById('targetSetupModal');
    
    try {
        // Bootstrap 5 방식
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            } else {
                // 인스턴스가 없으면 새로 생성해서 닫기
                const newModal = new bootstrap.Modal(modalElement);
                newModal.hide();
            }
        } else {
            // 수동으로 모달 닫기
            modalElement.style.display = 'none';
            modalElement.classList.remove('show');
            document.body.classList.remove('modal-open');
            
            // 배경 오버레이 제거
            const backdrop = document.querySelector('.modal-backdrop');
            if (backdrop) {
                backdrop.remove();
            }
        }
    } catch (error) {
        console.error('모달 닫기 오류:', error);
        // 강제로 숨기기
        modalElement.style.display = 'none';
        modalElement.classList.remove('show');
        document.body.classList.remove('modal-open');
        
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
    }
    
    // 첫 번째 비교 로드
    if (rlhfSystem) {
        rlhfSystem.loadNextComparison();
    }
    
    console.log(`세션 시작: 목표 ${targetValue}회`);
}
