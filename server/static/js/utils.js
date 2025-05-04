/**
 * Grasshopper RLHF - 공통 유틸리티 함수
 * 
 * 여러 JavaScript 모듈에서 재사용할 수 있는 공통 함수 및 유틸리티
 */

const AppUtils = {
    /**
     * 토스트 알림 표시
     * 
     * @param {string} title - 알림 제목
     * @param {string} message - 알림 메시지
     * @param {string} type - 알림 유형 (success, info, warning, danger)
     * @param {number} duration - 표시 시간(ms)
     */
    showToast: function(title, message, type = 'info', duration = 5000) {
        // 토스트 컨테이너 확인 또는 생성
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        // 토스트 ID 생성
        const toastId = `toast-${Date.now()}`;
        
        // 토스트 HTML 생성
        const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}</strong>: ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
        `;
        
        // 토스트 추가
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        // 토스트 객체 생성 및 표시
        const toast = new bootstrap.Toast(document.getElementById(toastId), { delay: duration });
        toast.show();
        
        // 표시 후 삭제
        const toastEl = document.getElementById(toastId);
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    },
    
    /**
     * API 오류 처리
     * 
     * @param {Error} error - 오류 객체
     * @param {string} fallbackMessage - 기본 오류 메시지
     */
    handleApiError: function(error, fallbackMessage = '오류가 발생했습니다') {
        console.error('API 오류:', error);
        this.showToast('오류', fallbackMessage, 'danger');
        return null;
    },
    
    /**
     * 타임스탬프를 사람이 읽을 수 있는 날짜 형식으로 변환
     * 
     * @param {number} timestamp - 밀리초 단위 타임스탬프
     * @returns {string} - 형식화된 날짜 문자열
     */
    formatDate: function(timestamp) {
        if (!timestamp) return 'N/A';
        
        try {
            const date = new Date(timestamp);
            return date.toLocaleString();
        } catch (e) {
            console.error('날짜 형식 변환 오류:', e);
            return 'Invalid Date';
        }
    },
    
    /**
     * 값에 소수점 자릿수 지정
     * 
     * @param {number} value - 대상 값
     * @param {number} decimals - 소수점 자릿수
     * @returns {string} - 형식화된 문자열
     */
    formatNumber: function(value, decimals = 2) {
        if (value === undefined || value === null) return 'N/A';
        
        try {
            return parseFloat(value).toFixed(decimals);
        } catch (e) {
            console.error('숫자 형식 변환 오류:', e);
            return 'Error';
        }
    },
    
    /**
     * 배열 값을 문자열로 변환
     * 
     * @param {Array} array - 배열
     * @param {number} decimals - 숫자 값의 소수점 자릿수
     * @returns {string} - 쉼표로 구분된 문자열
     */
    formatArray: function(array, decimals = 2) {
        if (!array || !Array.isArray(array)) return 'N/A';
        
        try {
            return array.map(item => {
                if (typeof item === 'number') {
                    return this.formatNumber(item, decimals);
                }
                return item;
            }).join(', ');
        } catch (e) {
            console.error('배열 형식 변환 오류:', e);
            return 'Error';
        }
    },
    
    /**
     * 문자열 잘라내기 (너무 긴 경우)
     * 
     * @param {string} text - 원본 문자열
     * @param {number} maxLength - 최대 길이
     * @returns {string} - 잘라낸 문자열
     */
    truncateText: function(text, maxLength = 100) {
        if (!text) return '';
        
        if (text.length <= maxLength) {
            return text;
        }
        
        return text.substring(0, maxLength) + '...';
    },
    
    /**
     * 간단한 디바운스 함수
     * 
     * @param {Function} func - 대상 함수
     * @param {number} wait - 대기 시간(ms)
     * @returns {Function} - 디바운스된 함수
     */
    debounce: function(func, wait = 300) {
        let timeout;
        
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * 객체의 심층 복사본 생성
     * 
     * @param {Object} obj - 복사할 객체
     * @returns {Object} - 복사된 객체
     */
    deepClone: function(obj) {
        if (obj === null || typeof obj !== 'object') {
            return obj;
        }
        
        // 날짜 객체 처리
        if (obj instanceof Date) {
            return new Date(obj.getTime());
        }
        
        // 배열 처리
        if (Array.isArray(obj)) {
            return obj.map(item => this.deepClone(item));
        }
        
        // 일반 객체 처리
        const clonedObj = {};
        for (const key in obj) {
            if (Object.prototype.hasOwnProperty.call(obj, key)) {
                clonedObj[key] = this.deepClone(obj[key]);
            }
        }
        
        return clonedObj;
    },
    
    /**
     * 값의 범위 제한
     * 
     * @param {number} value - 대상 값
     * @param {number} min - 최소값
     * @param {number} max - 최대값
     * @returns {number} - 제한된 값
     */
    clamp: function(value, min, max) {
        return Math.min(Math.max(value, min), max);
    },
    
    /**
     * 안전한 JSON 파싱
     * 
     * @param {string} jsonString - JSON 문자열
     * @param {*} defaultValue - 기본값
     * @returns {*} - 파싱된 객체 또는 기본값
     */
    safeJsonParse: function(jsonString, defaultValue = null) {
        try {
            return JSON.parse(jsonString);
        } catch (e) {
            console.error('JSON 파싱 오류:', e);
            return defaultValue;
        }
    },
    
    /**
     * 쿼리 매개변수 객체를 URL 문자열로 변환
     * 
     * @param {Object} params - 매개변수 객체
     * @returns {string} - URL 쿼리 문자열
     */
    objectToQueryString: function(params) {
        if (!params || typeof params !== 'object') return '';
        
        return Object.entries(params)
            .filter(([_, value]) => value !== undefined && value !== null)
            .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
            .join('&');
    }
};

// 공통 API 클라이언트
class ApiClient {
    /**
     * 공통 API 호출 함수
     * 
     * @param {string} url - API 엔드포인트
     * @param {Object} options - fetch 옵션
     * @returns {Promise} - 응답 데이터 또는 오류
     */
    static async request(url, options = {}) {
        try {
            // 기본 옵션 설정
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            // 옵션 병합
            const fetchOptions = { ...defaultOptions, ...options };
            
            // JSON 요청 본문 처리
            if (fetchOptions.body && typeof fetchOptions.body === 'object') {
                fetchOptions.body = JSON.stringify(fetchOptions.body);
            }
            
            // API 호출
            const response = await fetch(url, fetchOptions);
            
            // JSON 응답 파싱
            const data = await response.json();
            
            // 오류 응답 처리
            if (!response.ok || data.status === 'error') {
                const errorMessage = data.message || `서버 오류 (${response.status})`;
                throw new Error(errorMessage);
            }
            
            return data;
            
        } catch (error) {
            // 오류 처리
            AppUtils.handleApiError(error, `API 요청 실패: ${url}`);
            throw error;
        }
    }
    
    /**
     * GET 요청
     * 
     * @param {string} url - API 엔드포인트
     * @param {Object} params - 쿼리 매개변수
     * @returns {Promise} - 응답 데이터
     */
    static async get(url, params = {}) {
        // 쿼리 매개변수 추가
        const queryString = AppUtils.objectToQueryString(params);
        const fullUrl = queryString ? `${url}?${queryString}` : url;
        
        return this.request(fullUrl, { method: 'GET' });
    }
    
    /**
     * POST 요청
     * 
     * @param {string} url - API 엔드포인트
     * @param {Object} data - 요청 본문 데이터
     * @returns {Promise} - 응답 데이터
     */
    static async post(url, data = {}) {
        return this.request(url, {
            method: 'POST',
            body: data
        });
    }
    
    /**
     * PUT 요청
     * 
     * @param {string} url - API 엔드포인트
     * @param {Object} data - 요청 본문 데이터
     * @returns {Promise} - 응답 데이터
     */
    static async put(url, data = {}) {
        return this.request(url, {
            method: 'PUT',
            body: data
        });
    }
    
    /**
     * DELETE 요청
     * 
     * @param {string} url - API 엔드포인트
     * @returns {Promise} - 응답 데이터
     */
    static async delete(url) {
        return this.request(url, { method: 'DELETE' });
    }
}

// 전역 객체로 등록
window.AppUtils = AppUtils;
window.ApiClient = ApiClient;