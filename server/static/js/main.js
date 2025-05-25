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

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    rlhfSystem = new RLHFSystem();
});/**
 * RLHF 인간 피드백 시스템 - 메인 애플리케이션
 */

class RLHFSystem {
    constructor() {
        this.currentDesigns = { a: null, b: null };
        this.scenes = { a: null, b: null };
        this.cameras = { a: null, b: null };
        this.renderers = { a: null, b: null };
        this.controls = { a: null, b: null };
        this.sessionStats = { 
            total_comparisons: 0, 
            target_comparisons: 100 
        };
        
        this.init();
    }
    
    async init() {
        console.log('RLHF 시스템 초기화 중...');
        
        try {
            this.initViewers();
            await this.loadNextComparison();
            console.log('초기화 완료');
        } catch (error) {
            console.error('초기화 오류:', error);
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
        this.scenes[side].background = new THREE.Color(0x2c3e50);
        
        // Camera 생성
        this.cameras[side] = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.cameras[side].position.set(3, 3, 3);
        this.cameras[side].lookAt(0, 0, 0);
        
        // Renderer 생성
        this.renderers[side] = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderers[side].setSize(width, height);
        this.renderers[side].setPixelRatio(window.devicePixelRatio);
        
        // Controls 생성
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls[side] = new THREE.OrbitControls(
                this.cameras[side], 
                this.renderers[side].domElement
            );
            this.controls[side].enableDamping = true;
            this.controls[side].dampingFactor = 0.05;
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
    
    setupLights(side) {
        const scene = this.scenes[side];
        
        // 주변광
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        // 주 방향성 조명
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        // 보조 조명
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 3, -5);
        scene.add(fillLight);
        
        // 그리드 헬퍼
        const gridHelper = new THREE.GridHelper(10, 10, 0x555555, 0x333333);
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
        
        // 현재 비교 쌍 업데이트
        document.getElementById('current-pair').textContent = 
            `${designA.id.split('_')[0]} vs ${designB.id.split('_')[0]}`;
    }
    
    async loadMesh(side, design) {
        try {
            // 기존 메시 제거
            this.clearMesh(side);
            
            // 실제 메시 데이터가 있으면 API에서 로드 시도
            if (design.id && !design.id.startsWith('fallback_')) {
                const response = await fetch(`/api/mesh/${design.id}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'success' && data.mesh) {
                        this.createMeshFromData(side, data.mesh);
                        return;
                    }
                }
            }
            
            // 대체: 기본 큐브 생성
            this.createDefaultMesh(side);
            
        } catch (error) {
            console.error(`메시 로드 오류 (${side}):`, error);
            this.createDefaultMesh(side);
        }
    }
    
    createMeshFromData(side, meshData) {
        try {
            if (!meshData.meshes || !Array.isArray(meshData.meshes)) {
                this.createDefaultMesh(side);
                return;
            }
            
            const allMeshes = [];
            
            meshData.meshes.forEach(meshInfo => {
                if (!meshInfo.vertices || !meshInfo.faces) return;
                
                const geometry = new THREE.BufferGeometry();
                
                // 정점 설정 (좌표계 변환 적용)
                const vertices = [];
                for (let i = 0; i < meshInfo.vertices.length; i++) {
                    const vertex = meshInfo.vertices[i];
                    // Grasshopper 좌표계 → Three.js 좌표계 변환
                    // X → X, Y → Z, Z → -Y (일반적인 변환)
                    vertices.push(vertex[0], vertex[2], -vertex[1]);
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
                
                // 재질 생성
                const material = new THREE.MeshPhongMaterial({
                    color: side === 'a' ? 0x3498db : 0xe74c3c,
                    specular: 0x111111,
                    shininess: 30,
                    side: THREE.DoubleSide
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.isDesignMesh = true;
                this.scenes[side].add(mesh);
                allMeshes.push(mesh);
            });
            
            // 메시들을 중앙 정렬하고 카메라 위치 조정
            if (allMeshes.length > 0) {
                this.centerAndFitMeshes(side, allMeshes);
            }
            
        } catch (error) {
            console.error('메시 데이터 처리 오류:', error);
            this.createDefaultMesh(side);
        }
    }
    
    centerAndFitMeshes(side, meshes) {
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
        
        // 카메라 위치 자동 조정
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        
        if (maxDim > 0) {
            const camera = this.cameras[side];
            const fov = camera.fov * (Math.PI / 180);
            let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 1.2;
            
            // 카메라 위치 설정 (건축물 보기에 적합한 각도)
            camera.position.set(cameraDistance, cameraDistance * 0.7, cameraDistance);
            camera.lookAt(0, 0, 0);
            
            if (this.controls[side] && this.controls[side].target) {
                this.controls[side].target.set(0, 0, 0);
                this.controls[side].update();
            }
        }
    }
    
    createDefaultMesh(side) {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshPhongMaterial({
            color: side === 'a' ? 0x3498db : 0xe74c3c
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.isDesignMesh = true;
        this.scenes[side].add(mesh);
    }
    
    clearMesh(side) {
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
    
    updateMetrics(side, design) {
        const state = design.state || [0, 0, 0, 0];
        const reward = design.reward || 0;
        
        const metrics = {
            bcr: this.formatMetric(state[0] * 100, '%', [0, 70]),
            far: this.formatMetric(state[1] * 100, '%', [200, 500]),
            sunlight: this.formatMetric(state[2], 'kWh', [80000, 100000]),
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
                    this.updateStats();
                    
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
        document.getElementById('session-progress').textContent = `${Math.round(progress)}%`;
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
    alert('두 디자인 중 더 선호하는 디자인을 선택하세요.\n건축 지표를 참고하여 판단해주세요.');
}

// 뷰어 컨트롤 함수들
function resetView(side) {
    if (rlhfSystem && rlhfSystem.cameras[side] && rlhfSystem.controls[side]) {
        const camera = rlhfSystem.cameras[side];
        const controls = rlhfSystem.controls[side];
        
        // 건축물 보기에 적합한 기본 위치로 리셋
        camera.position.set(5, 3.5, 5);
        camera.lookAt(0, 0, 0);
        
        if (controls.target) {
            controls.target.set(0, 0, 0);
            controls.update();
        }
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
        rlhfSystem.showNotification(`디자인 ${side.toUpperCase()} 스크린샷이 저장되었습니다.`, 'success');
    }
}