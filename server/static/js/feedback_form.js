/**
 * Grasshopper RLHF - 피드백 폼 처리
 * 
 * 이 모듈은 사용자 피드백 입력과 제출을 관리합니다.
 */

class FeedbackForm {
    constructor() {
        // 폼 요소
        this.form = document.getElementById('feedback-form');
        this.submitButton = document.getElementById('btn-submit-feedback');
        
        // 평가 슬라이더
        this.sliders = {
            aesthetic: document.getElementById('rating-aesthetic'),
            functionality: document.getElementById('rating-functionality'),
            innovation: document.getElementById('rating-innovation'),
            feasibility: document.getElementById('rating-feasibility'),
            overall: document.getElementById('rating-overall')
        };
        
        // 평가 값 표시 요소
        this.valueLabels = {
            aesthetic: document.getElementById('rating-aesthetic-value'),
            functionality: document.getElementById('rating-functionality-value'),
            innovation: document.getElementById('rating-innovation-value'),
            feasibility: document.getElementById('rating-feasibility-value'),
            overall: document.getElementById('rating-overall-value')
        };
        
        // 코멘트 입력
        this.commentInput = document.getElementById('feedback-comment');
        
        // 모달
        this.feedbackModal = new bootstrap.Modal(document.getElementById('feedbackModal'));
        this.nextDesignButton = document.getElementById('btn-next-design');
        
        // 현재 디자인 ID
        this.currentDesignId = null;
        
        // 디자인 정보 표시 요소
        this.designInfoContainer = document.getElementById('design-info');
        
        // 초기화
        this.init();
    }

    /**
     * 초기화
     */
    init() {
        // 슬라이더 이벤트 리스너 등록
        for (const key in this.sliders) {
            if (this.sliders[key] && this.valueLabels[key]) {
                this.sliders[key].addEventListener('input', () => {
                    this.updateValueLabel(key);
                });
            }
        }
        
        // 폼 제출 이벤트 리스너
        if (this.form) {
            this.form.addEventListener('submit', (event) => {
                event.preventDefault();
                this.submitFeedback();
            });
        }
        
        // 다음 디자인 버튼 이벤트 리스너
        if (this.nextDesignButton) {
            this.nextDesignButton.addEventListener('click', () => {
                // 모달 닫기
                this.feedbackModal.hide();
                
                // 이벤트 발생
                const event = new CustomEvent('nextDesign', {
                    detail: { currentDesignId: this.currentDesignId }
                });
                document.dispatchEvent(event);
            });
        }
        
        console.log('피드백 폼 초기화 완료');
    }

    /**
     * 평가 값 표시 업데이트
     */
    updateValueLabel(key) {
        if (this.sliders[key] && this.valueLabels[key]) {
            const value = parseFloat(this.sliders[key].value);
            this.valueLabels[key].textContent = value.toFixed(1);
        }
    }

    /**
     * 피드백 제출
     */
    async submitFeedback() {
        if (!this.currentDesignId) {
            AppUtils.showToast('오류', '피드백을 제출할 디자인이 선택되지 않았습니다.', 'danger');
            return;
        }
        
        try {
            // 평가 데이터 수집
            const ratings = {};
            for (const key in this.sliders) {
                if (this.sliders[key]) {
                    ratings[key] = parseFloat(this.sliders[key].value);
                }
            }
            
            // 코멘트 가져오기
            const comment = this.commentInput ? this.commentInput.value.trim() : '';
            
            // 제출 데이터 구성
            const feedbackData = {
                design_id: this.currentDesignId,
                ratings: ratings,
                comment: comment
            };
            
            // 제출 버튼 비활성화
            if (this.submitButton) {
                this.submitButton.disabled = true;
                this.submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span> 제출 중...';
            }
            
            // API 호출
            const response = await ApiClient.post('/api/feedback', feedbackData);
            
            // 제출 버튼 원래대로 복원
            if (this.submitButton) {
                this.submitButton.disabled = false;
                this.submitButton.innerHTML = '<i class="fa-solid fa-paper-plane me-1"></i> 피드백 제출';
            }
            
            // 성공 메시지
            AppUtils.showToast('성공', '피드백이 성공적으로 제출되었습니다.', 'success');
            
            // 모달 표시
            this.feedbackModal.show();
            
            // 폼 리셋
            this.resetForm();
            
        } catch (error) {
            // 제출 버튼 원래대로 복원
            if (this.submitButton) {
                this.submitButton.disabled = false;
                this.submitButton.innerHTML = '<i class="fa-solid fa-paper-plane me-1"></i> 피드백 제출';
            }
            
            // 오류 처리는 ApiClient에서 자동으로 수행
        }
    }

    /**
     * 폼 리셋
     */
    resetForm() {
        // 슬라이더를 기본값으로 리셋
        for (const key in this.sliders) {
            if (this.sliders[key]) {
                this.sliders[key].value = 5;
                this.updateValueLabel(key);
            }
        }
        
        // 코멘트 입력 리셋
        if (this.commentInput) {
            this.commentInput.value = '';
        }
    }

    /**
     * 디자인 로드 처리
     */
    loadDesign(designData) {
        if (!designData) return;
        
        // 현재 디자인 ID 설정
        this.currentDesignId = designData.id;
        
        // 제출 버튼 활성화
        if (this.submitButton) {
            this.submitButton.disabled = false;
        }
        
        // 디자인 정보 표시
        this.displayDesignInfo(designData);
        
        // 기존 피드백 확인
        this.checkExistingFeedback(designData.id);
    }

    /**
     * 디자인 정보 표시
     */
    displayDesignInfo(designData) {
        if (!this.designInfoContainer || !designData) return;
        
        // 기본 정보 형식화
        const timestamp = designData.timestamp ? new Date(designData.timestamp) : new Date();
        const dateString = AppUtils.formatDate(timestamp);
        
        // 상태 값 가져오기
        const state = designData.state || [];
        
        // 액션 값 가져오기
        const action = designData.action || [];
        
        // HTML 구성
        let html = '<table class="design-info-table table-sm">';
        
        // ID 및 날짜
        html += `<tr><td>디자인 ID:</td><td>${designData.id}</td></tr>`;
        html += `<tr><td>생성 시간:</td><td>${dateString}</td></tr>`;
        
        // 보상 값
        html += `<tr><td>보상 값:</td><td><span class="badge bg-info">${AppUtils.formatNumber(designData.reward, 4)}</span></td></tr>`;
        
        // 상태 및 액션 값
        html += `<tr><td colspan="2" class="pt-2"><strong>상태 값:</strong></td></tr>`;
        state.forEach((value, index) => {
            html += `<tr><td>상태 ${index + 1}:</td><td>${AppUtils.formatNumber(value, 4)}</td></tr>`;
        });
        
        html += `<tr><td colspan="2" class="pt-2"><strong>액션 값:</strong></td></tr>`;
        action.forEach((value, index) => {
            html += `<tr><td>액션 ${index + 1}:</td><td>${AppUtils.formatNumber(value, 4)}</td></tr>`;
        });
        
        html += '</table>';
        
        // 컨테이너에 삽입
        this.designInfoContainer.innerHTML = html;
    }

    /**
     * 기존 피드백 확인
     */
    async checkExistingFeedback(designId) {
        if (!designId) return;
        
        try {
            // 피드백 데이터 요청
            const data = await ApiClient.get(`/api/feedback/${designId}`);
            
            if (data.feedback && data.feedback.length > 0) {
                // 가장 최근 피드백 사용
                const latestFeedback = data.feedback[data.feedback.length - 1];
                
                // 평가 값 적용
                if (latestFeedback.ratings) {
                    for (const key in latestFeedback.ratings) {
                        if (this.sliders[key]) {
                            this.sliders[key].value = latestFeedback.ratings[key];
                            this.updateValueLabel(key);
                        }
                    }
                }
                
                // 코멘트 적용
                if (latestFeedback.comment && this.commentInput) {
                    this.commentInput.value = latestFeedback.comment;
                }
                
                // 알림 표시
                AppUtils.showToast('기존 피드백', '이 디자인에 대한 기존 피드백이 로드되었습니다.', 'info');
            }
        } catch (error) {
            // 오류 처리는 ApiClient에서 자동으로 수행
        }
    }
}

// DOM이 로드된 후 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 전역 피드백 폼 인스턴스 생성
    window.feedbackForm = new FeedbackForm();
});