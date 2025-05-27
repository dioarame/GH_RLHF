// static/js/viewer/ViewerBase.js

import { VIEWER_CONFIG } from '../core/constants.js';

export class ViewerBase {
    constructor(containerId, side) {
        this.containerId = containerId;
        this.side = side;
        this.container = document.getElementById(containerId);
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.clock = new THREE.Clock();
        
        this.viewerMode = 'light';
        
        if (!this.container) {
            console.error(`컨테이너 ${containerId}를 찾을 수 없습니다`);
            return;
        }
        
        this.init();
    }
    
    init() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        // Scene 생성
        this.scene = new THREE.Scene();
        this.updateBackground('light');
        
        // Camera 생성
        this.camera = new THREE.PerspectiveCamera(
            VIEWER_CONFIG.camera.fov,
            width / height,
            VIEWER_CONFIG.camera.near,
            VIEWER_CONFIG.camera.far
        );
        this.camera.position.set(
            VIEWER_CONFIG.camera.initialPosition.x,
            VIEWER_CONFIG.camera.initialPosition.y,
            VIEWER_CONFIG.camera.initialPosition.z
        );
        this.camera.lookAt(
            VIEWER_CONFIG.camera.lookAt.x,
            VIEWER_CONFIG.camera.lookAt.y,
            VIEWER_CONFIG.camera.lookAt.z
        );
        
        // Renderer 생성
        this.createRenderer(width, height);
        
        // Controls 생성
        this.createControls();
        
        // 조명 설정
        this.setupLights();
        
        // DOM에 추가
        this.attachToDOM();
        
        // 애니메이션 시작
        this.animate();
    }
    
    createRenderer(width, height) {
        this.renderer = new THREE.WebGLRenderer({
            antialias: VIEWER_CONFIG.renderer.antialias,
            alpha: VIEWER_CONFIG.renderer.alpha,
            powerPreference: VIEWER_CONFIG.renderer.powerPreference,
            precision: VIEWER_CONFIG.renderer.precision
        });
        
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // 그림자 설정
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = VIEWER_CONFIG.renderer.shadowMapType;
        this.renderer.shadowMap.autoUpdate = true;
        
        // 톤 매핑 및 색상 관리
        this.renderer.toneMapping = VIEWER_CONFIG.renderer.toneMapping;
        this.renderer.toneMappingExposure = VIEWER_CONFIG.renderer.toneMappingExposure;
        this.renderer.outputEncoding = VIEWER_CONFIG.renderer.outputEncoding;
        this.renderer.physicallyCorrectLights = VIEWER_CONFIG.renderer.physicallyCorrectLights;
    }
    
    createControls() {
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            
            Object.assign(this.controls, VIEWER_CONFIG.controls);
        } else {
            console.warn('OrbitControls를 사용할 수 없습니다');
            this.controls = {
                update: () => {},
                dispose: () => {}
            };
        }
    }
    
    attachToDOM() {
        const loadingOverlay = this.container.querySelector('.loading-overlay');
        if (loadingOverlay) {
            this.container.insertBefore(this.renderer.domElement, loadingOverlay);
        } else {
            this.container.appendChild(this.renderer.domElement);
        }
    }
    
    setupLights() {
        // LightingManager가 처리하므로 여기서는 기본 조명만
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
    }
    
    addGrid() {
        const gridHelper = new THREE.GridHelper(100, 100, 0x606060, 0x404040);
        gridHelper.material.opacity = 0.2;
        gridHelper.material.transparent = true;
        gridHelper.isGridHelper = true;
        this.scene.add(gridHelper);
    }
    
    updateBackground(mode) {
        this.viewerMode = mode;
        
        if (!this.scene) return;
        
        if (mode === 'light') {
            this.scene.background = new THREE.Color(0x87CEEB);
            this.scene.fog = new THREE.Fog(0x87CEEB, 100, 500);
        } else {
            this.scene.background = new THREE.Color(0x1a1a1a);
            this.scene.fog = null;
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    handleResize() {
        if (!this.container || !this.camera || !this.renderer) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        if (this.controls) this.controls.dispose();
        if (this.renderer) this.renderer.dispose();
        // 추가 정리 작업
    }
}