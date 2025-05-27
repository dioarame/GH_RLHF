// static/js/viewer/MeshLoader.js

import { MATERIALS } from '../core/constants.js';
import { APIClient } from '../utils/APIClient.js';

export class MeshLoader {
    constructor(scene) {
        this.scene = scene;
    }
    
    async loadDesignMesh(designId, meshData) {
        try {
            // 기존 디자인 메시 제거
            this.clearDesignMeshes();
            
            if (!meshData) {
                // API에서 메시 데이터 로드
                const response = await APIClient.getMeshData(designId);
                if (response.status === 'success' && response.mesh) {
                    meshData = response.mesh;
                }
            }
            
            if (meshData && meshData.meshes) {
                this.createMeshFromData(meshData);
            } else {
                this.createDefaultMesh();
            }
        } catch (error) {
            console.error('메시 로드 오류:', error);
            this.createDefaultMesh();
        }
    }
    
    createMeshFromData(meshData) {
        if (!meshData.meshes || !Array.isArray(meshData.meshes)) {
            this.createDefaultMesh();
            return;
        }
        
        const meshGroup = new THREE.Group();
        
        meshData.meshes.forEach((meshInfo, meshIndex) => {
            if (!meshInfo.vertices || !meshInfo.faces) return;
            
            const geometry = new THREE.BufferGeometry();
            
            // 정점 설정 (Grasshopper → Three.js 좌표계 변환)
            const vertices = [];
            for (let i = 0; i < meshInfo.vertices.length; i++) {
                const vertex = meshInfo.vertices[i];
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
            
            // 면의 방향 분석
            const isHorizontal = this.analyzeOrientation(geometry);
            
            // 재질 선택
            const material = this.selectMaterial(meshIndex, isHorizontal);
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData.isDesignMesh = true;
            mesh.userData.isHorizontal = isHorizontal;
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            
            // 환경 메시와 동일하게 스케일 적용
            mesh.scale.set(1, 1, -1);
            
            // Z-fighting 방지
            if (isHorizontal) {
                mesh.position.y += meshIndex * 0.001;
            }
            
            meshGroup.add(mesh);
        });
        
        meshGroup.userData.isDesignMesh = true;
        this.scene.add(meshGroup);
    }
    
    analyzeOrientation(geometry) {
        const normals = geometry.getAttribute('normal').array;
        let horizontalScore = 0;
        let verticalScore = 0;
        
        for (let i = 0; i < normals.length; i += 3) {
            const nx = Math.abs(normals[i]);
            const ny = Math.abs(normals[i + 1]);
            const nz = Math.abs(normals[i + 2]);
            
            if (ny > 0.8) {
                horizontalScore++;
            } else {
                verticalScore++;
            }
        }
        
        return horizontalScore > verticalScore;
    }
    
    selectMaterial(meshIndex, isHorizontal) {
        if (isHorizontal) {
            // 수평면: 콘크리트
            return new THREE.MeshStandardMaterial({
                ...MATERIALS.building.concrete,
                side: THREE.DoubleSide
            });
        } else {
            // 수직면: 다양한 재질
            const materials = [
                MATERIALS.building.glass,
                MATERIALS.building.metal,
                MATERIALS.building.wood,
                { color: 0xff6b35, roughness: 0.4, metalness: 0.2 },
                { color: 0xff69b4, roughness: 0.4, metalness: 0.2 },
                { color: 0xffd700, roughness: 0.4, metalness: 0.2 }
            ];
            
            const selectedMaterial = materials[meshIndex % materials.length];
            return new THREE.MeshStandardMaterial({
                ...selectedMaterial,
                side: THREE.DoubleSide
            });
        }
    }
    
    createDefaultMesh() {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshStandardMaterial({
            color: 0xe0e0e0,
            roughness: 0.7,
            metalness: 0.1,
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.userData.isDesignMesh = true;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(0, 0.5, 0);
        
        this.scene.add(mesh);
    }
    
    clearDesignMeshes() {
        const meshesToRemove = [];
        this.scene.traverse(object => {
            if (object.userData.isDesignMesh) {
                meshesToRemove.push(object);
            }
        });
        
        meshesToRemove.forEach(mesh => {
            this.scene.remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            }
        });
    }
    
    async loadEnvironmentMeshes(environmentData) {
        if (environmentData.contour) {
            this.createEnvironmentMesh(environmentData.contour, MATERIALS.environment.contour, 'contour');
        }
        
        if (environmentData.surface) {
            this.createEnvironmentMesh(environmentData.surface, MATERIALS.environment.surface, 'surface');
        }
    }
    
    createEnvironmentMesh(meshData, materialConfig, type) {
        if (!meshData.meshes || !Array.isArray(meshData.meshes)) return;
        
        meshData.meshes.forEach(meshInfo => {
            if (!meshInfo.vertices || !meshInfo.faces) return;
            
            const geometry = new THREE.BufferGeometry();
            
            // 정점 설정
            const vertices = [];
            for (let i = 0; i < meshInfo.vertices.length; i++) {
                const vertex = meshInfo.vertices[i];
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
            
            const material = new THREE.MeshStandardMaterial({
                ...materialConfig,
                side: THREE.DoubleSide,
                emissive: 0x000000,
                envMapIntensity: 0.2
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData.isEnvironmentMesh = true;
            mesh.userData.envType = type;
            mesh.receiveShadow = true;
            mesh.scale.set(1, 1, -1);
            
            this.scene.add(mesh);
        });
    }
}