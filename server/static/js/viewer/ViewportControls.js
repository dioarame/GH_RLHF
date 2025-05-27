// static/js/viewer/ViewportControls.js

export class ViewportControls {
    constructor(scene, camera, controls) {
        this.scene = scene;
        this.camera = camera;
        this.controls = controls;
    }
    
    setViewport(viewType) {
        const { distance, center } = this.calculateSceneBounds();
        
        switch (viewType.toLowerCase()) {
            case 'top':
                this.setTopView(center, distance);
                break;
            case 'front':
                this.setFrontView(center, distance);
                break;
            case 'perspective':
                this.setPerspectiveView(center, distance);
                break;
            default:
                console.warn(`알 수 없는 뷰 타입: ${viewType}`);
        }
    }
    
    calculateSceneBounds() {
        const designMeshes = [];
        this.scene.traverse(object => {
            if (object.userData.isDesignMesh) {
                designMeshes.push(object);
            }
        });
        
        let distance = 10;
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
        
        return { distance, center };
    }
    
    setTopView(center, distance) {
        this.camera.position.set(center.x, center.y + distance, center.z);
        this.camera.lookAt(center);
        this.updateControls(center);
    }
    
    setFrontView(center, distance) {
        this.camera.position.set(center.x, center.y, center.z + distance);
        this.camera.lookAt(center);
        this.updateControls(center);
    }
    
    setPerspectiveView(center, distance) {
        // calculateSceneBounds를 다시 호출하지 않고 전달받은 값 사용
        if (!center && !distance) {
            const bounds = this.calculateSceneBounds();
            center = bounds.center;
            distance = bounds.distance;
        }
        
        // 디자인 메시가 있을 때만 줌 조정
        const designMeshes = [];
        this.scene.traverse(object => {
            if (object.userData.isDesignMesh) {
                designMeshes.push(object);
            }
        });
        
        if (designMeshes.length > 0) {
            // 디자인 메시만의 경계 상자 계산
            const box = new THREE.Box3();
            designMeshes.forEach(mesh => {
                box.expandByObject(mesh);
            });
            
            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            
            if (maxDim > 0) {
                const fov = this.camera.fov * (Math.PI / 180);
                // 적절한 거리 계산 (너무 멀지 않게)
                let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 0.7;
                
                // 건축물에 적합한 뷰 각도
                const buildingHeight = size.y; // Y축이 높이
                const cameraHeight = Math.max(buildingHeight * 0.5, cameraDistance * 0.3);
                
                // 중심점 계산
                const designCenter = new THREE.Vector3();
                box.getCenter(designCenter);
                
                // 카메라 위치 설정
                this.camera.position.set(
                    designCenter.x + cameraDistance * 0.7,
                    designCenter.y + cameraHeight,
                    designCenter.z + cameraDistance * 0.7
                );
                
                // 건물 중심을 바라보도록
                this.camera.lookAt(designCenter);
                
                // 컨트롤 타겟 업데이트
                if (this.controls && this.controls.target) {
                    this.controls.target.copy(designCenter);
                    this.controls.update();
                }
                
                console.log(`Perspective 뷰 설정 완료:`, {
                    center: designCenter,
                    distance: cameraDistance,
                    size: { x: size.x.toFixed(2), y: size.y.toFixed(2), z: size.z.toFixed(2) }
                });
            }
        } else {
            // 디자인 메시가 없으면 기본 뷰
            this.camera.position.set(10, 10, 10);
            this.camera.lookAt(0, 0, 0);
            this.updateControls(new THREE.Vector3(0, 0, 0));
        }
    }
    
    updateControls(center) {
        if (this.controls && this.controls.target) {
            this.controls.target.copy(center);
            this.controls.update();
        }
    }
    
    resetView() {
        this.zoomToFitAll();
    }
    
    zoomToFitAll() {
        const designMeshes = [];
        this.scene.traverse(object => {
            if (object.userData.isDesignMesh) {
                designMeshes.push(object);
            }
        });
        
        if (designMeshes.length === 0) return;
        
        const box = new THREE.Box3();
        designMeshes.forEach(mesh => {
            box.expandByObject(mesh);
        });
        
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        
        if (maxDim > 0) {
            const fov = this.camera.fov * (Math.PI / 180);
            const cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 2.5;
            
            const buildingHeight = size.z;
            const cameraHeight = Math.max(buildingHeight * 0.4, cameraDistance * 0.3);
            
            this.camera.position.set(
                cameraDistance * 0.7,
                cameraHeight,
                cameraDistance * 0.7
            );
            
            const lookAtHeight = buildingHeight * 0.3;
            this.camera.lookAt(0, lookAtHeight, 0);
            
            if (this.controls && this.controls.target) {
                this.controls.target.set(0, lookAtHeight, 0);
                this.controls.update();
            }
        }
    }
    
    toggleWireframe() {
        this.scene.traverse(object => {
            if (object.userData.isDesignMesh && object.material) {
                object.material.wireframe = !object.material.wireframe;
            }
        });
    }
}