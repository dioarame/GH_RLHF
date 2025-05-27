// static/js/core/constants.js

export const VIEWER_CONFIG = {
    camera: {
        fov: 45,
        near: 0.1,
        far: 1000,
        initialPosition: { x: 5, y: 3, z: 5 },
        lookAt: { x: 0, y: 1, z: 0 }
    },
    renderer: {
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
        precision: "highp",
        shadowMapType: THREE.PCFSoftShadowMap,
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.8,
        outputEncoding: THREE.sRGBEncoding,
        physicallyCorrectLights: true
    },
    controls: {
        enableDamping: true,
        dampingFactor: 0.05,
        rotateSpeed: 0.5,
        zoomSpeed: 0.5,
        panSpeed: 0.3,
        minDistance: 0.1,
        maxDistance: 1000
    }
};

export const LIGHT_PRESETS = {
    bright: {
        ambient: { intensity: 0.6, color: '#ffffff' },
        sun: { intensity: 1.0, color: '#ffffff', x: 50, y: 100, z: 50 },
        sky: { intensity: 0.4, color: '#87CEEB' },
        hemisphere: { intensity: 0.5, skyColor: '#87CEEB', groundColor: '#8B6914' },
        renderer: { toneMappingExposure: 1.2 }
    },
    vibrant: {
        ambient: { intensity: 1.0, color: '#ffffff' },
        sun: { intensity: 1.5, color: '#fffaf0', x: 100, y: 150, z: 80 },
        sky: { intensity: 0.8, color: '#87ceeb' },
        hemisphere: { intensity: 0.9, skyColor: '#87ceeb', groundColor: '#ffd4a3' },
        renderer: { toneMappingExposure: 2.0 }
    },
    soft: {
        ambient: { intensity: 0.7, color: '#fff5e6' },
        sun: { intensity: 0.6, color: '#ffe4b5', x: 30, y: 80, z: 40 },
        sky: { intensity: 0.3, color: '#b0c4de' },
        hemisphere: { intensity: 0.4, skyColor: '#b0c4de', groundColor: '#8b7355' },
        renderer: { toneMappingExposure: 1.0 }
    },
    cloudy: {
        ambient: { intensity: 0.8, color: '#e0e0e0' },
        sun: { intensity: 0.3, color: '#d3d3d3', x: 20, y: 60, z: 30 },
        sky: { intensity: 0.5, color: '#a9a9a9' },
        hemisphere: { intensity: 0.6, skyColor: '#d3d3d3', groundColor: '#696969' },
        renderer: { toneMappingExposure: 0.9 }
    },
    evening: {
        ambient: { intensity: 0.3, color: '#ff6b35' },
        sun: { intensity: 0.7, color: '#ff8c00', x: 80, y: 30, z: 60 },
        sky: { intensity: 0.2, color: '#ff7f50' },
        hemisphere: { intensity: 0.3, skyColor: '#ff7f50', groundColor: '#8b4513' },
        renderer: { toneMappingExposure: 1.1 }
    }
};

export const MATERIALS = {
    environment: {
        contour: {
            color: 0xe0e0e0,
            roughness: 0.9,
            metalness: 0.0,
            opacity: 1.0
        },
        surface: {
            color: 0xf8f8f8,
            roughness: 0.95,
            metalness: 0.0,
            opacity: 0.8,
            transparent: true
        }
    },
    building: {
        concrete: {
            color: 0xc0c0c0,
            roughness: 0.9,
            metalness: 0.0
        },
        glass: {
            color: 0x88ccff,
            roughness: 0.1,
            metalness: 0.0,
            transparent: true,
            opacity: 0.7
        },
        metal: {
            color: 0xe0e0e0,
            roughness: 0.3,
            metalness: 0.8
        },
        wood: {
            color: 0xd4a373,
            roughness: 0.7,
            metalness: 0.0
        }
    }
};