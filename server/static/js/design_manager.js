/**
 * Grasshopper RLHF - 디자인 관리자
 * 
 * 이 모듈은 디자인 목록 로드 및 관리를 담당합니다.
 */

class DesignManager {
    constructor() {
        // DOM 요소
        this.designList = document.getElementById('design-list');
        this.refreshButton = document.getElementById('btn-refresh');
        this.zmqStatusEl = document.getElementById('zmq-status');
        
        // 상태 변수
        this.designs = [];
        this.selectedDesignId = null;
        this.currentSort = { field: 'timestamp', order: 'desc' };
        
        // 초기화
        this.init();
    }

    /**
     * 초기화
     */
    init() {
        // 초기 데이터 로드
        this.loadDesigns();
        
        // 이벤트 리스너 등록
        if (this.refreshButton) {
            this.refreshButton.addEventListener('click', () => {
                this.loadDesigns();
            });
        }
        
        // 정렬 옵션 이벤트 리스너
        document.querySelectorAll('.sort-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                const sortField = e.target.dataset.sort;
                const sortOrder = e.target.dataset.order;
                
                if (sortField && sortOrder) {
                    this.sortDesigns(sortField, sortOrder);
                }
            });
        });
        
        // 다음 디자인 이벤트 리스너
        document.addEventListener('nextDesign', (e) => {
            if (e.detail && e.detail.currentDesignId) {
                this.selectNextDesign(e.detail.currentDesignId);
            }
        });
        
        // ZMQ 상태 확인
        this.checkZmqStatus();
        
        // 정기적인 ZMQ 상태 확인 (30초마다)
        setInterval(() => this.checkZmqStatus(), 30000);
        
        console.log('디자인 관리자 초기화 완료');
    }

    /**
     * 디자인 목록 로드
     */
    async loadDesigns() {
        if (!this.designList) return;
        
        try {
            // 로딩 상태 표시
            this.designList.innerHTML = `
                <div class="list-group-item text-center py-5">
                    <div class="spinner-border text-primary mb-2" role="status">
                        <span class="visually-hidden">로딩 중...</span>
                    </div>
                    <p class="mb-0">디자인 목록을 불러오는 중...</p>
                </div>
            `;
            
            // 디자인 데이터 요청
            const data = await ApiClient.get('/api/designs');
            
            // 디자인 목록 저장
            this.designs = data.designs || [];
            
            // 데이터 없음 처리
            if (this.designs.length === 0) {
                this.designList.innerHTML = `
                    <div class="list-group-item text-center py-5">
                        <i class="fa-solid fa-info-circle mb-2 fs-4 text-info"></i>
                        <p class="mb-0">디자인 데이터가 없습니다.</p>
                        <p class="small text-muted mt-2">디자인 데이터를 생성하거나 RLHF 기준 데이터를 가져오세요.</p>
                    </div>
                `;
                return;
            }
            
            // 디자인 정렬
            this.sortDesigns(this.currentSort.field, this.currentSort.order, false);
            
        } catch (error) {
            // 오류 처리는 ApiClient에서 자동으로 수행
            this.designList.innerHTML = `
                <div class="list-group-item text-center py-5">
                    <div class="alert alert-danger mb-0">
                        <i class="fa-solid fa-exclamation-circle me-1"></i>
                        오류: 디자인 목록을 불러올 수 없습니다.
                    </div>
                </div>
            `;
        }
    }

    /**
     * 디자인 정렬
     */
    sortDesigns(field, order, updateList = true) {
        if (!this.designs || this.designs.length === 0) return;
        
        // 정렬 상태 업데이트
        this.currentSort = { field, order };
        
        // 정렬 함수
        const sortFn = (a, b) => {
            let aValue = a[field];
            let bValue = b[field];
            
            // 타임스탬프는 숫자로 변환
            if (field === 'timestamp') {
                aValue = parseInt(aValue) || 0;
                bValue = parseInt(bValue) || 0;
            }
            
            // 오름차순/내림차순 결정
            if (order === 'asc') {
                return aValue < bValue ? -1 : (aValue > bValue ? 1 : 0);
            } else {
                return aValue > bValue ? -1 : (aValue < bValue ? 1 : 0);
            }
        };
        
        // 디자인 배열 정렬
        this.designs.sort(sortFn);
        
        // 목록 업데이트 필요한 경우
        if (updateList) {
            this.renderDesignList();
        }
    }

    /**
     * 디자인 목록 렌더링
     */
    renderDesignList() {
        if (!this.designList || !this.designs || this.designs.length === 0) return;
        
        // HTML 초기화
        this.designList.innerHTML = '';
        
        // 각 디자인에 대한 HTML 생성
        this.designs.forEach(design => {
            // 디자인 ID
            const designId = design.id;
            
            // 보상 값 형식화
            const reward = AppUtils.formatNumber(design.reward, 4);
            
            // 타임스탬프 형식화
            const dateString = AppUtils.formatDate(design.timestamp);
            
            // 액션 값 가져오기
            const actionString = AppUtils.formatArray(design.action, 2);
            
            // 디자인 아이템 생성
            const itemEl = document.createElement('div');
            itemEl.className = `list-group-item design-item${designId === this.selectedDesignId ? ' active' : ''}`;
            itemEl.dataset.designId = designId;
            
            // HTML 내용
            itemEl.innerHTML = `
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <div class="design-item-title">${designId.split('_')[0]}</div>
                        <div class="small">액션: ${actionString}</div>
                        <div class="design-item-date">${dateString}</div>
                    </div>
                    <div class="design-item-reward">${reward}</div>
                </div>
            `;
            
            // 클릭 이벤트 리스너
            itemEl.addEventListener('click', () => {
                this.selectDesign(designId);
            });
            
            // 목록에 추가
            this.designList.appendChild(itemEl);
        });
    }

    /**
     * 디자인 선택
     */
    selectDesign(designId) {
        if (!designId) return;
        
        // 이전 선택 항목 비활성화
        if (this.selectedDesignId) {
            const prevItem = this.designList.querySelector(`.design-item[data-design-id="${this.selectedDesignId}"]`);
            if (prevItem) {
                prevItem.classList.remove('active');
            }
        }
        
        // 새 선택 항목 활성화
        const item = this.designList.querySelector(`.design-item[data-design-id="${designId}"]`);
        if (item) {
            item.classList.add('active');
            
            // 스크롤 위치 조정
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        // 선택 디자인 ID 업데이트
        this.selectedDesignId = designId;
        
        // 선택한 디자인 데이터 찾기
        const designData = this.designs.find(d => d.id === designId);
        if (!designData) return;
        
        // 뷰어 제목 업데이트
        const viewerTitle = document.getElementById('viewer-title');
        if (viewerTitle) {
            viewerTitle.textContent = `디자인 ${designId.split('_')[0]}`;
        }
        
        // 3D 메쉬 로드
        if (window.threeViewer) {
            window.threeViewer.loadDesignMesh(designId);
        }
        
        // 피드백 폼 업데이트
        if (window.feedbackForm) {
            window.feedbackForm.loadDesign(designData);
        }
    }

    /**
     * 다음 디자인 선택
     */
    selectNextDesign(currentDesignId) {
        if (!this.designs || this.designs.length <= 1) return;
        
        // 현재 디자인의 인덱스 찾기
        const currentIndex = this.designs.findIndex(d => d.id === currentDesignId);
        if (currentIndex === -1) return;
        
        // 다음 인덱스 계산 (마지막이면 처음으로)
        const nextIndex = (currentIndex + 1) % this.designs.length;
        
        // 다음 디자인 선택
        const nextDesignId = this.designs[nextIndex].id;
        this.selectDesign(nextDesignId);
    }

    /**
     * ZMQ 상태 확인
     */
    async checkZmqStatus() {
        if (!this.zmqStatusEl) return;
        
        try {
            // ZMQ 상태 요청
            const data = await ApiClient.get('/api/zmq/ping');
            
            if (data.response && data.response.status === 'success') {
                // 연결 성공
                this.zmqStatusEl.innerHTML = 'ZMQ 연결됨';
                this.zmqStatusEl.classList.remove('text-danger', 'text-warning');
                this.zmqStatusEl.classList.add('text-success');
            } else {
                // 연결 실패
                this.zmqStatusEl.innerHTML = 'ZMQ 연결 실패';
                this.zmqStatusEl.classList.remove('text-success', 'text-warning');
                this.zmqStatusEl.classList.add('text-danger');
            }
        } catch (error) {
            // 예외 발생
            this.zmqStatusEl.innerHTML = 'ZMQ 상태 확인 오류';
            this.zmqStatusEl.classList.remove('text-success', 'text-danger');
            this.zmqStatusEl.classList.add('text-warning');
        }
    }

    /**
     * 기준 데이터 로드
     */
    async loadReferenceData() {
        try {
            // 기준 데이터 요청
            const data = await ApiClient.get('/api/reference-data');
            
            const referenceData = data.reference_data;
            if (!referenceData) return false;
            
            // 디자인 데이터 변환 및 추가
            const designsToAdd = [];
            
            // 상위 디자인 추가
            if (referenceData.top_designs && Array.isArray(referenceData.top_designs)) {
                referenceData.top_designs.forEach(design => {
                    designsToAdd.push({
                        id: `top_${design.msg_id || Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
                        timestamp: design.timestamp || Date.now(),
                        state: design.state || [],
                        action: design.action || [],
                        reward: design.reward || 0,
                        metadata: {
                            type: 'top',
                            reference: true
                        }
                    });
                });
            }
            
            // 다양한 디자인 추가
            if (referenceData.diverse_designs && Array.isArray(referenceData.diverse_designs)) {
                referenceData.diverse_designs.forEach(design => {
                    designsToAdd.push({
                        id: `diverse_${design.msg_id || Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
                        timestamp: design.timestamp || Date.now(),
                        state: design.state || [],
                        action: design.action || [],
                        reward: design.reward || 0,
                        metadata: {
                            type: 'diverse',
                            reference: true,
                            cluster: design.cluster
                        }
                    });
                });
            }
            
            // 서버에 디자인 추가 요청
            for (const design of designsToAdd) {
                await ApiClient.post('/api/designs', design);
            }
            
            // 디자인 목록 새로고침
            await this.loadDesigns();
            
            // 성공 메시지
            AppUtils.showToast('기준 데이터', `${designsToAdd.length}개의 디자인이 로드되었습니다.`, 'success');
            
            return true;
            
        } catch (error) {
            // 오류 처리는 ApiClient에서 자동으로 수행
            return false;
        }
    }
}

// DOM이 로드된 후 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 전역 디자인 관리자 인스턴스 생성
    window.designManager = new DesignManager();
});