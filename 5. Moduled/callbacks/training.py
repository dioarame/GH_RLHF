# callbacks/training.py
from stable_baselines3.common.callbacks import BaseCallback
import time

class FPSLimiter(BaseCallback):
    """학습 속도를 제한하는 콜백 클래스"""
    
    def __init__(self, limit_fps):
        """
        FPS 제한 콜백 초기화
        
        Args:
            limit_fps: 초당 최대 스텝 수
        """
        super().__init__(verbose=0)
        self.limit_fps = min(limit_fps, 2.0)  # 최대 2 FPS로 제한
        self.min_interval = 1.0 / self.limit_fps if self.limit_fps > 0 else 0
        self.last_time = time.time()
        self.start_time = time.time()
        print(f"  속도 제한: {self.limit_fps} steps/sec (최소 간격: {self.min_interval:.3f}초)")

    def _on_training_start(self) -> None:
        """학습 시작 시 타이머 초기화"""
        self.start_time = time.time()
        self.last_time = time.time()

    def _on_step(self) -> bool:
        """매 스텝 마다 속도 제한 적용"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        # 더 자주 로그 출력
        if self.num_timesteps > 0 and self.num_timesteps % 10 == 0:
            total_elapsed = current_time - self.start_time
            steps_per_sec = self.num_timesteps / total_elapsed if total_elapsed > 0 else 0
            print(f"🔄 스텝 {self.num_timesteps}/{self.locals.get('total_timesteps', '?')}: {steps_per_sec:.2f} steps/sec")

        # 속도 제한 강화
        if self.min_interval > 0:
            wait_time = self.min_interval - elapsed
            if wait_time > 0: 
                # 대기 시간 출력 (선택사항)
                if wait_time > 0.5:  # 0.5초 이상 대기할 때만 출력
                    print(f"⏱️ 속도 제한: {wait_time:.2f}초 대기 중...")
                time.sleep(wait_time)
                
        self.last_time = time.time()
        return True