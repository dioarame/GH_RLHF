// static/js/lighting/LightingManager.js

import { LIGHT_PRESETS } from '../core/constants.js';

export class LightingManager {
    constructor(scene) {
        this.scene = scene;
        this.lights = {};
        
        this.settings = {
            ambient: {
                intensity: 0.6,
                color: '#ffffff'
            },
            sun: {
                intensity: 0.8,
                color: '#ffffff',
                x: 50,
                y: 100,
                z: 50,
                shadowMapSize: 2048
            },
            sky: {
                intensity: 0.4,
                color: '#87CEEB'
            },
            hemisphere: {
                intensity: 0.5,
                skyColor: '#87CEEB',
                groundColor: '#8B6914'
            },
            showHelpers: false
        };
        
        this.setupLights();
    }
    
    setupLights() {
        // 기존 조명 제거
        this.removeAllLights();
        
        // 1. 주변광
        this.lights.ambient = new THREE.AmbientLight(
            this.settings.ambient.color,
            this.settings.ambient.intensity
        );
        this.scene.add(this.lights.ambient);
        
        // 2. 태양광
        this.lights.sun = new THREE.DirectionalLight(
            this.settings.sun.color,
            this.settings.sun.intensity
        );
        this.lights.sun.position.set(
            this.settings.sun.x,
            this.settings.sun.y,
            this.settings.sun.z
        );
        this.lights.sun.castShadow = true;
        this.configureSunShadow();
        this.scene.add(this.lights.sun);
        
        // 3. 하늘빛
        this.lights.sky = new THREE.DirectionalLight(
            this.settings.sky.color,
            this.settings.sky.intensity
        );
        this.lights.sky.position.set(-50, 50, -30);
        this.scene.add(this.lights.sky);
        
        // 4. 반구광
        this.lights.hemisphere = new THREE.HemisphereLight(
            this.settings.hemisphere.skyColor,
            this.settings.hemisphere.groundColor,
            this.settings.hemisphere.intensity
        );
        this.lights.hemisphere.position.set(0, 50, 0);
        this.scene.add(this.lights.hemisphere);
        
        // 5. 헬퍼들
        if (this.settings.showHelpers) {
            this.addHelpers();
        }
    }
    
    configureSunShadow() {
        const shadow = this.lights.sun.shadow;
        shadow.mapSize.width = this.settings.sun.shadowMapSize;
        shadow.mapSize.height = this.settings.sun.shadowMapSize;
        shadow.camera.near = 0.1;
        shadow.camera.far = 500;
        shadow.camera.left = -100;
        shadow.camera.right = 100;
        shadow.camera.top = 100;
        shadow.camera.bottom = -100;
        shadow.radius = 2;
        shadow.blurSamples = 10;
        shadow.normalBias = 0.05;
    }
    
    removeAllLights() {
        const lights = [];
        this.scene.traverse(child => {
            if (child.isLight) {
                lights.push(child);
            }
        });
        lights.forEach(light => {
            this.scene.remove(light);
            if (light.dispose) light.dispose();
        });
    }
    
    addHelpers() {
        // 축 헬퍼
        const axesHelper = new THREE.AxesHelper(5);
        axesHelper.isAxesHelper = true;
        this.scene.add(axesHelper);
        
        // 태양광 헬퍼
        if (this.lights.sun) {
            this.lights.sunHelper = new THREE.DirectionalLightHelper(this.lights.sun, 5);
            this.lights.sunHelper.isLightHelper = true;
            this.scene.add(this.lights.sunHelper);
        }
    }
    
    updateLighting() {
        if (this.lights.ambient) {
            this.lights.ambient.intensity = this.settings.ambient.intensity;
            this.lights.ambient.color.set(this.settings.ambient.color);
        }
        
        if (this.lights.sun) {
            this.lights.sun.intensity = this.settings.sun.intensity;
            this.lights.sun.color.set(this.settings.sun.color);
            this.lights.sun.position.set(
                this.settings.sun.x,
                this.settings.sun.y,
                this.settings.sun.z
            );
        }
        
        if (this.lights.sky) {
            this.lights.sky.intensity = this.settings.sky.intensity;
            this.lights.sky.color.set(this.settings.sky.color);
        }
        
        if (this.lights.hemisphere) {
            this.lights.hemisphere.intensity = this.settings.hemisphere.intensity;
            this.lights.hemisphere.color.set(this.settings.hemisphere.skyColor);
            this.lights.hemisphere.groundColor.set(this.settings.hemisphere.groundColor);
        }
        
        if (this.lights.sunHelper) {
            this.lights.sunHelper.update();
        }
    }
    
    applyPreset(presetName) {
        const preset = LIGHT_PRESETS[presetName];
        if (!preset) return;
        
        Object.assign(this.settings.ambient, preset.ambient);
        Object.assign(this.settings.sun, preset.sun);
        Object.assign(this.settings.sky, preset.sky);
        Object.assign(this.settings.hemisphere, preset.hemisphere);
        
        this.updateLighting();
        
        return preset.renderer; // 렌더러 설정 반환
    }
    
    toggleHelpers() {
        this.settings.showHelpers = !this.settings.showHelpers;
        
        this.scene.traverse(object => {
            if (object.isAxesHelper || object.isLightHelper) {
                object.visible = this.settings.showHelpers;
            }
        });
    }
}