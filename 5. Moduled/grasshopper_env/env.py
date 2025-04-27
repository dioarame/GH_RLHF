# grasshopper_env/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import Dict, Optional, Any, Tuple, List

from .utils import ComputeClient
from .communication import ZMQCommunicator

class SimpleGrasshopperEnv(gym.Env):
    """Grasshopper 환경의 Gym 인터페이스"""
    
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(
        self,
        compute_url: str,
        gh_definition_path: str,
        state_output_param_name: str,
        reward_output_param_name: str,
        slider_info_param_name: str = "SliderInfo",
        max_episode_steps: int = 100,
        action_push_port: int = 5556,
        use_push_mode: bool = True,
    ):
        super().__init__()

        # 환경 설정
        self.state_name = state_output_param_name
        self.reward_name = reward_output_param_name
        self.slider_info_name = slider_info_param_name
        self._max_steps = max_episode_steps
        self._step_counter = 0
        self.retry_count = 3
        self.retry_delay = 0.5
        
        # 기본 Action/Observation Space 설정 (초기값)
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)
        self.slider_roundings = [0.01]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # 클라이언트 초기화
        self.compute_client = ComputeClient(compute_url, gh_definition_path)
        self.communicator = ZMQCommunicator(action_push_port, use_push_mode)
        
        # 환경 정보 초기화 (슬라이더 범위 등)
        self._initialize_env_info()

    def _initialize_env_info(self):
        """환경 정보 초기화 (슬라이더 범위, 상태 차원 등)"""
        print("\n[초기화] 슬라이더 정보 가져오고 Action/Observation Space 정의 중...")
        slider_infos = []
        obs_dimension = 1
        initial_obs = None

        for attempt in range(self.retry_count):
            try:
                print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - Rhino.Compute 요청 중...")
                response = self.compute_client.call_compute()
                
                if response:
                    # 슬라이더 정보 파싱
                    slider_info_raw = self.compute_client.get_param_data(response, self.slider_info_name)
                    if slider_info_raw:
                        print(f"[초기화] '{self.slider_info_name}' 파라미터 데이터 발견.")
                        slider_infos = self.compute_client.parse_slider_info(slider_info_raw)
                        if slider_infos:
                            print(f"[초기화] {len(slider_infos)}개 슬라이더 정보 파싱 성공!")
                        else:
                            print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.slider_info_name}' 데이터 파싱 실패.")
                    else:
                        print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.slider_info_name}' 파라미터 또는 데이터 없음.")

                    # 상태 정보 파싱
                    state_raw = self.compute_client.get_param_data(response, self.state_name)
                    if state_raw is not None:
                        print(f"[초기화] '{self.state_name}' 파라미터 데이터 발견: '{state_raw}'")
                        try:
                            parsed_state = [float(x.strip()) for x in state_raw.split(',') if x.strip()]
                            if parsed_state:
                                obs_dimension = len(parsed_state)
                                initial_obs = np.array(parsed_state, dtype=np.float32)
                                print(f"[초기화] 상태 파싱 성공. Observation 차원: {obs_dimension}")
                            else:
                                print(f"[초기화] 상태 데이터 파싱 실패 (빈 결과). 기본 1차원 사용.")
                                initial_obs = np.array([0.0], dtype=np.float32)
                        except ValueError:
                            print(f"[초기화] 상태 데이터 '{state_raw}'를 float 리스트로 변환 실패. 기본 1차원 사용.")
                            initial_obs = np.array([0.0], dtype=np.float32)
                    else:
                        print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.state_name}' 파라미터 또는 데이터 없음. 기본 1차원 사용.")
                        initial_obs = np.array([0.0], dtype=np.float32)

                    if slider_infos and initial_obs is not None:
                        break  # 성공적으로 정보를 가져온 경우 반복 종료
                else:
                    print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - Compute 응답 없음.")

            except Exception as e:
                print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - 오류 발생: {e}")

            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)

        # Action Space 설정
        if slider_infos:
            mins = np.array([info[0] for info in slider_infos], dtype=np.float32)
            maxs = np.array([info[1] for info in slider_infos], dtype=np.float32)
            self.slider_roundings = [info[2] if info[2] is not None else 0.01 for info in slider_infos]
            self.action_space = spaces.Box(low=mins, high=maxs, dtype=np.float32)
            print(f"[초기화] {len(slider_infos)}차원 Action Space 설정 완료:")
            print(f"  - 최소값: {self.action_space.low}")
            print(f"  - 최대값: {self.action_space.high}")
            print(f"  - Rounding 값: {self.slider_roundings}")
        else:
            print(f"[초기화] 슬라이더 정보를 가져올 수 없어 기본 1차원 Action Space 사용.")
            print(f"  - Action Space: {self.action_space}")
            print(f"  - Rounding 값: {self.slider_roundings}")

        # Observation Space 설정
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dimension,), dtype=np.float32)
        print(f"[초기화] Observation Space 설정 완료 (Shape: {self.observation_space.shape})")
        if initial_obs is not None:
            print(f"  - 감지된 초기 상태 값: {initial_obs}")
        
    def reset(self, *, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        self._step_counter = 0

        print("[Reset] 환경 초기화 중...")
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}

        # 초기 액션 샘플링 및 전송
        init_action = self.action_space.sample()
        print(f"[Reset] 초기 액션 샘플링: {init_action}")
        self.communicator.send_action(
            init_action, 
            self.action_space.low, 
            self.action_space.high, 
            self.slider_roundings
        )
        
        # Grasshopper가 액션을 처리하고 상태를 계산할 시간 부여
        time.sleep(0.2)

        # 초기 상태 가져오기
        for attempt in range(self.retry_count):
            print(f"[Reset] 시도 {attempt+1}/{self.retry_count} - 초기 상태 가져오기...")
            compute_result = self.compute_client.call_compute()

            if compute_result is not None:
                state, _ = self.compute_client.parse_state_reward(
                    compute_result, 
                    self.state_name, 
                    self.reward_name,
                    self.observation_space.shape
                )
                print(f"[Reset] 성공. 초기 상태: {state}")
                return state, info
            else:
                print(f"[Reset Error] 시도 {attempt+1}/{self.retry_count} - Compute 호출 실패.")
                if attempt < self.retry_count - 1:
                    print(f"[Reset] 재시도 전 액션 재전송: {init_action}")
                    self.communicator.send_action(
                        init_action, 
                        self.action_space.low, 
                        self.action_space.high, 
                        self.slider_roundings
                    )
                    time.sleep(self.retry_delay)

        print("[Reset Error] 최대 시도 횟수 초과. 기본 상태 반환.")
        info["error"] = "Reset failed after retries"
        return state, info

    def step(self, action):
        """환경 진행"""
        # 액션 전송
        send_success = self.communicator.send_action(
            action, 
            self.action_space.low, 
            self.action_space.high, 
            self.slider_roundings
        )
        
        if not send_success:
            print("[Step Warning] 액션 전송 실패. 계속 진행합니다.")

        # Grasshopper가 액션 처리 후 상태/보상 계산할 시간 부여
        time.sleep(0.1)

        # 기본값 설정
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = -2e3
        terminated = False
        truncated = False
        info = {}

        # 상태 및 보상 가져오기
        compute_result = self.compute_client.call_compute()

        if compute_result is not None:
            state, reward = self.compute_client.parse_state_reward(
                compute_result, 
                self.state_name, 
                self.reward_name,
                self.observation_space.shape
            )
            # 100번째 스텝마다 또는 처음 스텝에서 로그 출력
            if self._step_counter % 100 == 0 or self._step_counter == 0:
                print(f"[Step {self._step_counter}] 상태: {state}, 보상: {reward}")
        else:
            print("[Step Error] Compute 호출 실패. 기본값 사용.")
            info["error"] = "Compute call failed in step"

        # 스텝 카운터 증가 및 truncation 체크
        self._step_counter += 1
        truncated = self._step_counter >= self._max_steps
        if truncated: info["TimeLimit.truncated"] = True

        return state, reward, terminated, truncated, info

    def render(self):
        """환경 렌더링 (사용하지 않음)"""
        pass

    def close(self):
        """환경 리소스 정리"""
        print("Closing environment...")
        
        # Compute 클라이언트 정리
        if hasattr(self, 'compute_client'):
            self.compute_client.close()
            
        # ZMQ 통신 종료
        if hasattr(self, 'communicator'):
            self.communicator.close()
            
        print("Environment closed.")