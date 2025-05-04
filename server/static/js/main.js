/**
 * Grasshopper RLHF - 메인 애플리케이션 로직
 * 
 * 이 모듈은 사용자 인터페이스 초기화 및 화면 전환을 처리합니다.
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Grasshopper RLHF 인간 피드백 수집 시스템 초기화 중...');
    
    // 네비게이션 링크
    const navLinks = {
        designs: document.getElementById('nav-designs'),
        feedback: document.getElementById('nav-feedback'),
        analysis: document.getElementById('nav-analysis')
    };
    
    // 활성 화면 설정
    function setActiveScreen(screenId) {
        // 네비게이션 링크 업데이트
        for (const key in navLinks) {
            if (navLinks[key]) {
                navLinks[key].classList.toggle('active', key === screenId);
            }
        }
        
        // 화면별 특수 처리
        switch (screenId) {
            case 'designs':
                // 디자인 화면 활성화 시 처리
                break;
                
            case 'feedback':
                // 피드백 화면 활성화 시 처리
                break;
                
            case 'analysis':
                // 분석 화면 활성화 시 처리
                break;
        }
    }
    
    // 네비게이션 이벤트 리스너 등록
    for (const key in navLinks) {
        if (navLinks[key]) {
            navLinks[key].addEventListener('click', (e) => {
                e.preventDefault();
                setActiveScreen(key);
            });
        }
    }
    
    // 키보드 단축키 처리
    document.addEventListener('keydown', (e) => {
        // Ctrl+R: 새로고침
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            if (window.designManager) {
                window.designManager.loadDesigns();
            }
        }
        
        // 화살표 키: 디자인 탐색
        if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
            e.preventDefault();
            if (window.designManager && window.designManager.designs.length > 0) {
                const currentIndex = window.designManager.designs.findIndex(d => d.id === window.designManager.selectedDesignId);
                if (currentIndex !== -1) {
                    let newIndex;
                    
                    if (e.key === 'ArrowLeft') {
                        // 이전 디자인
                        newIndex = (currentIndex - 1 + window.designManager.designs.length) % window.designManager.designs.length;
                    } else {
                        // 다음 디자인
                        newIndex = (currentIndex + 1) % window.designManager.designs.length;
                    }
                    
                    window.designManager.selectDesign(window.designManager.designs[newIndex].id);
                }
            }
        }
        
        // Escape: 모달 닫기
        if (e.key === 'Escape') {
            // 활성화된 모달 닫기
            const modals = document.querySelectorAll('.modal.show');
            if (modals.length > 0) {
                e.preventDefault();
                modals.forEach(modal => {
                    const modalInstance = bootstrap.Modal.getInstance(modal);
                    if (modalInstance) {
                        modalInstance.hide();
                    }
                });
            }
        }
    });
    
    // 서버 상태 확인 및 기본 화면 설정
    setActiveScreen('designs');
    
    console.log('Grasshopper RLHF 인간 피드백 수집 시스템 초기화 완료');
});