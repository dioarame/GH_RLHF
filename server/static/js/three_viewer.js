/**
 * Grasshopper RLHF - Three.js 3D 뷰어
 * 
 * 이 모듈은 Grasshopper에서 내보낸 메쉬 데이터를 
 * Three.js를 사용하여 렌더링합니다.
 */

class ThreeViewer {
    constructor(containerId, options = {}) {
        // 기본 옵션 설정
        this.options = Object.assign({
            showGrid: true,
            showAxes: true,
            backgroundColor: 0x2c3e50,
            wireframe: false,
            cameraDistance: 5,
            lightIntensity: 0.8
        }, options);
        
        // DOM 컨테이너
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`컨테이너 요소를 찾을 수 없습니다: ${containerId}`);
            return;
        }

        // 기본 상태
        this.isInitialized = false;
        this.isWireframe = this.options.wireframe;
        this.currentDesignId = null;
        this.meshes = [];
        
        // 이벤트 핸들러 바인딩
        this.handleResize = this.handleResize.bind(this);
        
        // Three.js 초기화
        this.init();
    }

    /**
     * Three.js 시스템 초기화
     */
    init() {
        try {
            // Scene 생성
            this.setupScene();
            
            // 카메라 설정
            this.setupCamera();
            
            // 렌더러 설정
            this.setupRenderer();
            
            // 컨트롤 추가
            this.setupControls();
            
            // 조명 설정
            this.setupLights();
            
            // Helper 추가
            if (this.options.showGrid || this.options.showAxes) {
                this.addHelpers();
            }
            
            // 애니메이션 루프 시작
            this.animate();
            
            // 창 크기 변경 이벤트 리스너
            window.addEventListener('resize', this.handleResize);
            
            // 초기화 성공
            this.isInitialized = true;
            console.log('Three.js 뷰어 초기화 완료');
        } catch (error) {
            console.error('Three.js 뷰어 초기화 실패:', error);
        }
    }
    
    /**
     * Scene 설정
     */
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
    }
    
    /**
     * 카메라 설정
     */
    setupCamera() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        
        const distance = this.options.cameraDistance;
        this.camera.position.set(distance, distance, distance);
        this.camera.lookAt(0, 0, 0);
    }
    
    /**
     * 렌더러 설정
     */
    setupRenderer() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    /**
     * 컨트롤 설정
     */
    setupControls() {
        // OrbitControls가 없는 경우 대체 구현
        if (typeof THREE.OrbitControls === 'undefined') {
            console.log('OrbitControls not available, using simple controls');
            // 간단한 컨트롤 구현
            this.controls = {
                update: function() {},
                target: new THREE.Vector3(0, 0, 0),
                dispose: function() {},
                reset: function() {}
            };
        } else {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.target.set(0, 0, 0);
        }
    }

    /**
     * 조명 설정
     */
    setupLights() {
        // 주변광
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
        
        // 주 방향성 조명
        const mainLight = new THREE.DirectionalLight(0xffffff, this.options.lightIntensity);
        mainLight.position.set(5, 10, 7);
        mainLight.castShadow = true;
        
        // 그림자 품질 설정
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        mainLight.shadow.camera.near = 0.5;
        mainLight.shadow.camera.far = 50;
        mainLight.shadow.camera.left = -10;
        mainLight.shadow.camera.right = 10;
        mainLight.shadow.camera.top = 10;
        mainLight.shadow.camera.bottom = -10;
        
        this.scene.add(mainLight);
        
        // 채움 조명
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
    }

    /**
     * Helper 추가 (그리드, 축)
     */
    addHelpers() {
        if (this.options.showGrid) {
            // 그리드 헬퍼
            const gridHelper = new THREE.GridHelper(20, 20, 0x555555, 0x333333);
            this.scene.add(gridHelper);
        }
        
        if (this.options.showAxes) {
            // 축 헬퍼
            const axesHelper = new THREE.AxesHelper(2);
            this.scene.add(axesHelper);
        }
    }

    /**
     * 애니메이션 루프
     */
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // 컨트롤 업데이트
        if (this.controls) {
            this.controls.update();
        }
        
        // 씬 렌더링
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    /**
     * 창 크기 변경 핸들러
     */
    handleResize() {
        if (!this.isInitialized) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        // 카메라 비율 업데이트
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        // 렌더러 크기 업데이트
        this.renderer.setSize(width, height);
    }

    /**
     * 뷰 리셋
     */
    resetView() {
        if (!this.isInitialized) return;
        
        // 카메라 위치 재설정
        const distance = this.options.cameraDistance;
        this.camera.position.set(distance, distance, distance);
        this.camera.lookAt(0, 0, 0);
        
        // 컨트롤 리셋
        this.controls.reset();
    }

    /**
     * 와이어프레임 모드 전환
     */
    toggleWireframe() {
        if (!this.isInitialized || this.meshes.length === 0) return false;
        
        this.isWireframe = !this.isWireframe;
        
        // 모든 메쉬의 재질 업데이트
        this.meshes.forEach(mesh => {
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(material => {
                        material.wireframe = this.isWireframe;
                    });
                } else {
                    mesh.material.wireframe = this.isWireframe;
                }
            }
        });
        
        return this.isWireframe;
    }

    /**
     * 스크린샷 생성
     */
    takeScreenshot() {
        if (!this.isInitialized) return null;
        
        // 현재 렌더러에서 이미지 데이터 URL 생성
        return this.renderer.domElement.toDataURL('image/png');
    }

    /**
     * 모든 메쉬 제거
     */
    clearMeshes() {
        if (!this.isInitialized) return;
        
        // 기존 메쉬 제거
        this.meshes.forEach(mesh => {
            this.scene.remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(material => material.dispose());
                } else {
                    mesh.material.dispose();
                }
            }
        });
        
        this.meshes = [];
    }

    /**
     * JSON 형식의 메쉬 데이터 로드
     */
    loadJsonMesh(jsonData) {
        if (!this.isInitialized) return 0;
        
        // 기존 메쉬 제거
        this.clearMeshes();
        
        try {
            // 데이터 유효성 검사
            if (!jsonData || !jsonData.meshes || !Array.isArray(jsonData.meshes)) {
                console.error('유효하지 않은 JSON 메쉬 데이터');
                return 0;
            }
            
            // 각 메쉬 처리
            jsonData.meshes.forEach(meshData => {
                if (!meshData.vertices || !meshData.faces) return;
                
                // 지오메트리 생성
                const geometry = new THREE.BufferGeometry();
                
                // 정점 설정
                const vertices = [];
                for (let i = 0; i < meshData.vertices.length; i++) {
                    const vertex = meshData.vertices[i];
                    vertices.push(vertex[0], vertex[1], vertex[2]);
                }
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                
                // 인덱스 설정
                const indices = [];
                for (let i = 0; i < meshData.faces.length; i++) {
                    const face = meshData.faces[i];
                    if (face.length === 3) {
                        indices.push(face[0], face[1], face[2]);
                    } else if (face.length === 4) {
                        // 쿼드를 두 개의 삼각형으로 변환
                        indices.push(face[0], face[1], face[2]);
                        indices.push(face[0], face[2], face[3]);
                    }
                }
                
                if (indices.length > 0) {
                    geometry.setIndex(indices);
                }
                
                // 법선 설정 (제공된 경우)
                if (meshData.normals && meshData.normals.length > 0) {
                    const normals = [];
                    for (let i = 0; i < meshData.normals.length; i++) {
                        const normal = meshData.normals[i];
                        normals.push(normal[0], normal[1], normal[2]);
                    }
                    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
                } else {
                    // 법선 자동 계산
                    geometry.computeVertexNormals();
                }
                
                // 재질 생성
                const material = new THREE.MeshPhongMaterial({
                    color: 0x3498db,
                    specular: 0x111111,
                    shininess: 30,
                    flatShading: false,
                    side: THREE.DoubleSide,
                    wireframe: this.isWireframe
                });
                
                // 메쉬 생성 및 추가
                const mesh = new THREE.Mesh(geometry, material);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                this.scene.add(mesh);
                this.meshes.push(mesh);
            });
            
            // 중앙 정렬 및 카메라 조정
            this.centerAndFitMeshes();
            
            return this.meshes.length;
            
        } catch (error) {
            console.error('JSON 메쉬 로드 중 오류:', error);
            return 0;
        }
    }
    
    /**
     * 메쉬 중앙 정렬 및 카메라 조정
     */
    centerAndFitMeshes() {
        if (this.meshes.length === 0) return;
        
        // 모든 메쉬를 포함하는 경계 상자 계산
        const meshGroup = new THREE.Group();
        this.meshes.forEach(mesh => meshGroup.add(mesh.clone()));
        
        const box = new THREE.Box3().setFromObject(meshGroup);
        const center = new THREE.Vector3();
        box.getCenter(center);
        
        // 메쉬를 중앙으로 이동
        this.meshes.forEach(mesh => {
            mesh.position.sub(center);
        });
        
        // 카메라 위치 조정
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2));
        
        // 안전 마진 추가
        cameraDistance *= 1.5;
        
        // 카메라 위치 설정
        this.camera.position.set(cameraDistance, cameraDistance, cameraDistance);
        this.camera.lookAt(0, 0, 0);
        
        // 컨트롤 타겟 설정
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    /**
     * 디자인 ID로 메쉬 로드
     */
    async loadDesignMesh(designId) {
        if (!this.isInitialized || !designId) return false;
        
        this.currentDesignId = designId;
        
        try {
            // 로딩 표시
            this.showLoading(true);
            
            // 메쉬 데이터 요청
            const response = await fetch(`/api/mesh/${designId}`);
            const data = await response.json();
            
            if (data.status === 'error') {
                console.error('메쉬 데이터 로드 오류:', data.message);
                this.showLoading(false);
                return false;
            }
            
            // 메쉬 로드
            const meshData = data.mesh;
            
            if (meshData.format === 'json') {
                this.loadJsonMesh(meshData);
            } else {
                console.error('지원되지 않는 메쉬 형식:', meshData.format);
                this.showLoading(false);
                return false;
            }
            
            // 로딩 완료
            this.showLoading(false);
            return true;
            
        } catch (error) {
            console.error('디자인 메쉬 로드 중 오류:', error);
            this.showLoading(false);
            return false;
        }
    }

    /**
     * 로딩 표시 제어
     */
    showLoading(isLoading) {
        // 기존 로딩 요소 제거
        const existingLoader = this.container.querySelector('.viewer-loading');
        if (existingLoader) {
            existingLoader.remove();
        }
        
        // 로딩 표시가 필요한 경우
        if (isLoading) {
            const loader = document.createElement('div');
            loader.className = 'viewer-loading';
            
            const spinner = document.createElement('div');
            spinner.className = 'spinner-border text-light mb-2';
            spinner.setAttribute('role', 'status');
            
            const span = document.createElement('span');
            span.className = 'visually-hidden';
            span.textContent = '로딩 중...';
            
            const text = document.createElement('p');
            text.className = 'mb-0';
            text.textContent = '3D 모델 로딩 중...';
            
            spinner.appendChild(span);
            loader.appendChild(spinner);
            loader.appendChild(text);
            
            this.container.appendChild(loader);
        }
    }

    /**
     * 정리 함수
     */
    destroy() {
        // 이벤트 리스너 제거
        window.removeEventListener('resize', this.handleResize);
        
        // Three.js 리소스 정리
        this.clearMeshes();
        
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
        
        if (this.controls) {
            this.controls.dispose();
        }
        
        // 상태 초기화
        this.isInitialized = false;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
    }
}

// 전역 뷰어 인스턴스
let threeViewer;

// DOM이 로드된 후 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 뷰어 컨테이너가 있는지 확인
    const viewerContainer = document.getElementById('viewer-container');
    if (viewerContainer) {
        // 뷰어 옵션 설정
        const viewerOptions = {
            showGrid: true,
            showAxes: true,
            backgroundColor: 0x2c3e50,
            wireframe: false,
            cameraDistance: 5
        };
        
        // 뷰어 생성
        threeViewer = new ThreeViewer('viewer-container', viewerOptions);
        
        // 뷰 리셋 버튼 이벤트 리스너
        const resetButton = document.getElementById('btn-reset-view');
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                threeViewer.resetView();
            });
        }
        
        // 와이어프레임 버튼 이벤트 리스너
        const wireframeButton = document.getElementById('btn-wireframe');
        if (wireframeButton) {
            wireframeButton.addEventListener('click', () => {
                const isWireframe = threeViewer.toggleWireframe();
                wireframeButton.classList.toggle('active', isWireframe);
            });
        }
        
        // 스크린샷 버튼 이벤트 리스너
        const screenshotButton = document.getElementById('btn-screenshot');
        if (screenshotButton) {
            screenshotButton.addEventListener('click', () => {
                const dataUrl = threeViewer.takeScreenshot();
                if (dataUrl) {
                    // 스크린샷 표시
                    const container = document.createElement('div');
                    container.className = 'screenshot-container';
                    
                    const img = document.createElement('img');
                    img.src = dataUrl;
                    img.className = 'screenshot-image';
                    
                    const controls = document.createElement('div');
                    controls.className = 'screenshot-controls';
                    
                    const closeButton = document.createElement('button');
                    closeButton.className = 'btn btn-light me-2';
                    closeButton.innerHTML = '<i class="fa-solid fa-times"></i>';
                    closeButton.addEventListener('click', () => {
                        document.body.removeChild(container);
                    });
                    
                    const downloadButton = document.createElement('button');
                    downloadButton.className = 'btn btn-primary';
                    downloadButton.innerHTML = '<i class="fa-solid fa-download me-1"></i> 다운로드';
                    downloadButton.addEventListener('click', () => {
                        const link = document.createElement('a');
                        link.href = dataUrl;
                        link.download = `design-${threeViewer.currentDesignId || 'screenshot'}.png`;
                        link.click();
                    });
                    
                    controls.appendChild(closeButton);
                    controls.appendChild(downloadButton);
                    
                    container.appendChild(img);
                    container.appendChild(controls);
                    
                    document.body.appendChild(container);
                }
            });
        }
    }
});

// 전역으로 내보내기
window.threeViewer = threeViewer;