#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RLHF 통합 건축 설계 최적화 시스템

기존 rl_architecture_optimizer.py를 베이스로 하여 RLHF 기능을 통합한 버전
- 인간 피드백 기반 보상 모델 통합
- 3가지 가중치 조합 지원 (0.3, 0.5, 0.7)
- 연속 학습 (각 가중치당 3라운드)
- Closed Brep 처리 로직 유지
- 자동화된 분석 및 보고서 생성
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
import datetime
import signal
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import zmq
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

# === 경로 설정 ===
BASE_DIR = Path(r"C:\Users\valen\Desktop\Dev\6. RLHF")
MODULES_DIR = BASE_DIR / "python_modules"
DATA_DIR = BASE_DIR / "data"
ZMQ_LOGS_DIR = DATA_DIR / "zmq_logs"

# 모듈 경로 추가
sys.path.insert(0, str(MODULES_DIR))

# === 파일 경로 설정 ===
DEFAULT_PATHS = {
    'reward_model': BASE_DIR / "python_modules" / "improved_reward_models" / "improved_reward_model_symmetry_20250530_165846.pt",
    'initial_ppo_model': BASE_DIR / "data" / "models" / "ppo_architecture_20250523_162526" / "final_model.zip",
    'base_output_dir': BASE_DIR / "rlhf_experiments"
}

# === 글로벌 변수 ===
STATE_QUEUE = queue.Queue()
STOP_EVENT = threading.Event()
LAST_STATE = None
DEBUG = False

# === 로깅 함수들 (기존 RL과 동일) ===
def log_info(message):
    print(message)

def log_warning(message):
    print(f"\033[93m⚠️ {message}\033[0m")

def log_error(message):
    print(f"\033[91m❌ {message}\033[0m")

def log_success(message):
    print(f"\033[92m✅ {message}\033[0m")

def log_debug(message):
    if DEBUG:
        print(f"\033[94m🔍 {message}\033[0m")

# === 신호 핸들러 ===
def signal_handler(sig, frame):
    log_info("\n🛑 사용자에 의해 학습이 중단되었습니다.")
    STOP_EVENT.set()

# === 모듈 임포트 ===
try:
    from reward_adapter import create_reward_function
    REWARD_ADAPTER_AVAILABLE = True
except ImportError as e:
    REWARD_ADAPTER_AVAILABLE = False
    log_warning(f"Reward adapter import failed: {e}")

# 인간 피드백 보상 모델 관련
try:
    from architectural_reward_model import ProbabilisticArchitecturalModel_V2
    HUMAN_REWARD_MODEL_AVAILABLE = True
    
    # 🔍 진단 코드 추가
    print(f"🔍 임포트 결과:")
    print(f"   ProbabilisticArchitecturalModel_V2 = {ProbabilisticArchitecturalModel_V2}")
    print(f"   type = {type(ProbabilisticArchitecturalModel_V2)}")
    print(f"   callable = {callable(ProbabilisticArchitecturalModel_V2)}")
    
except ImportError as e:
    HUMAN_REWARD_MODEL_AVAILABLE = False
    print(f"❌ Human reward model import failed: {e}")

# === 기존 imports 다음에 추가 ===
def normalize_rewards_for_rlhf(env_reward, human_reward):
    """가중치별 비교 실험을 위한 균형 잡힌 정규화"""
    ENV_SCALE = 2.661
    HUMAN_SCALE = 0.357
    
    env_normalized = np.tanh(env_reward / ENV_SCALE)
    human_normalized = np.tanh(human_reward / HUMAN_SCALE)
    
    return env_normalized, human_normalized

# === RLHF 보상 모델 로더 ===
class HumanRewardModelLoader:
    """인간 피드백 보상 모델 로더"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scalers = {}  # 개선된 모델용
        self.scaler_mean = None  # 기존 모델용
        self.scaler_scale = None  # 기존 모델용
        self.device = 'cpu'
        self.is_improved_model = False  # 모델 타입 구분
        
    def load_model(self):
        """보상 모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                log_error(f"Reward model file not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            log_info(f"Loading reward model: {os.path.basename(self.model_path)}")
            
            # 개선된 모델인지 확인
            if 'scalers' in checkpoint:
                # 개선된 모델 로드
                self.is_improved_model = True
                log_info("Loading improved reward model with multiple scalers")
                
                # 스케일러 정보 복원
                from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                
                scaler_info = checkpoint['scalers']
                for feature, info in scaler_info.items():
                    if info['type'] == 'StandardScaler':
                        scaler = StandardScaler()
                        scaler.mean_ = np.array(info['mean'])
                        scaler.scale_ = np.array(info['scale'])
                    elif info['type'] == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                        scaler.min_ = np.array(info['min'])
                        scaler.scale_ = np.array(info['scale'])
                    elif info['type'] == 'RobustScaler':
                        scaler = RobustScaler()
                        scaler.center_ = np.array(info['center'])
                        scaler.scale_ = np.array(info['scale'])
                    
                    self.scalers[feature] = scaler
                
                # 모델 생성
                from architectural_reward_model import SimplifiedRewardModel
                config = checkpoint.get('config', {})
                self.model = SimplifiedRewardModel(
                    state_dim=4,
                    hidden_dim=config.get('hidden_dim', 96)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # 학습 정보 출력
                if 'best_model_info' in checkpoint and checkpoint['best_model_info']:
                    best_info = checkpoint['best_model_info']
                    log_info(f"✅ Best model info:")
                    log_info(f"   Validation accuracy: {best_info['val_accuracy']:.3f}")
                    log_info(f"   Reward mean: {best_info['reward_stats']['mean']:.4f}")
                    log_info(f"   Negative ratio: {best_info['reward_stats']['negative_ratio']:.3f}")
                
            else:
                # 기존 모델 로드
                self.is_improved_model = False
                model_type = checkpoint.get('model_type', 'unknown')
                timestamp = checkpoint.get('timestamp', 'unknown')
                log_info(f"Model type: {model_type}, Timestamp: {timestamp}")
                
                if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                    self.scaler_mean = np.array(checkpoint['scaler_mean'])
                    self.scaler_scale = np.array(checkpoint['scaler_scale'])
                    log_success("Scaler information loaded")
                else:
                    log_warning("No scaler information found in checkpoint")
                
                if HUMAN_REWARD_MODEL_AVAILABLE:
                    state_dim = checkpoint.get('state_dim', 4)
                    self.model = ProbabilisticArchitecturalModel_V2(state_dim=state_dim)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                else:
                    log_error("ProbabilisticArchitecturalModel_V2 not available")
                    return False
            
            log_success("Human reward model loaded successfully")
            return True
                
        except Exception as e:
            log_error(f"Failed to load reward model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_reward(self, state_vector):
        """보상 예측"""
        if self.model is None:
            return 0.0
        
        try:
            if len(state_vector) != 4:
                log_warning(f"Invalid state vector length: {len(state_vector)}")
                return 0.0
            
            # 제약 조건 체크 제거 - 환경 보상이 이미 처리하므로 이중 처벌 방지
            bcr, far, winter_sunlight, sv_ratio = state_vector
            
            # 주석 처리 또는 제거
            # if bcr > 0.7:  # BCR 70% 초과
            #     return -30.0
            # if far < 2.0:  # FAR 200% 미만
            #     return -30.0
            # if far > 5.0:  # FAR 500% 초과
            #     return -30.0
            
            # 선택적: 극단값에 대한 소프트 패널티 (필요시)
            soft_penalty = 0.0
            if far > 5.0:
                # 초과 정도에 비례한 부드러운 패널티
                excess_ratio = min((far - 5.0) / 2.0, 1.0)  # 최대 1.0
                soft_penalty = -2.0 * excess_ratio  # 최대 -2.0
            elif far < 2.0:
                # 미달 정도에 비례한 부드러운 패널티
                deficit_ratio = min((2.0 - far) / 2.0, 1.0)
                soft_penalty = -2.0 * deficit_ratio
            
            if bcr > 0.7:
                excess_ratio = min((bcr - 0.7) / 0.3, 1.0)
                soft_penalty += -1.0 * excess_ratio  # 추가 최대 -1.0
            
            # 정상 범위인 경우만 모델 예측 진행
            # 스케일링 적용
            if self.is_improved_model and self.scalers:  # 개선된 모델
                state_scaled = np.zeros_like(state_vector)
                feature_names = ['BCR', 'FAR', 'WinterTime', 'SVR']
                for i, feature in enumerate(feature_names):
                    if feature in self.scalers:
                        state_scaled[i] = self.scalers[feature].transform(
                            np.array(state_vector[i]).reshape(-1, 1)
                        )[0, 0]
                    else:
                        state_scaled[i] = state_vector[i]
            elif self.scaler_mean is not None and self.scaler_scale is not None:  # 기존 모델
                state_scaled = (np.array(state_vector) - self.scaler_mean) / self.scaler_scale
            else:
                state_scaled = np.array(state_vector)
            
            # 모델 예측
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0)
                raw_reward = self.model(state_tensor).item()
                
                # 스케일링 적용
                if self.is_improved_model:  # 개선된 모델
                    # 법적 제약 명시적 처리
                    if bcr > 0.7 or far > 5.0 or far < 2.0:
                        # 위반 시 무조건 음수
                        base_penalty = -5.0
                        
                        # 모델 예측도 고려하되 제한적으로
                        model_contribution = raw_reward * 0.5
                        
                        final_reward = base_penalty + np.clip(model_contribution, -5, 2)
                    else:
                        # 정상 범위에서만 모델 신뢰
                        final_reward = raw_reward * 2.5
                        final_reward = np.clip(final_reward, -10.0, 8.0)
                else:  # 기존 모델
                    final_reward = 8.0 * torch.tanh(torch.tensor(raw_reward / 2.0)).item()
                    final_reward = np.clip(final_reward, -10.0, 10.0)
            
            return final_reward
            
        except Exception as e:
            log_warning(f"Human reward prediction error: {e}")
            return 0.0

# === RLHF 상태 수신기 (기존 StateReceiver 구조 유지) ===
class RLHFStateReceiver:
    """RLHF를 위한 향상된 상태 수신기"""
    
    def __init__(self, port=5557, save_dir=None, human_reward_model=None):
        self.port = port
        self.save_dir = save_dir or ZMQ_LOGS_DIR
        self.human_reward_model = human_reward_model
                
        # 환경 보상 함수를 초기화 시 한 번만 생성
        self.env_reward_function = None
        if REWARD_ADAPTER_AVAILABLE:
            self.env_reward_function = create_reward_function(
                reward_type="optimized",
                bcr_limit=70.0,
                far_min=200.0,
                far_max=500.0,
                use_seasonal=True,
                debug=DEBUG
            )
        
        # 기존과 동일한 구조
        self.context = None
        self.socket = None
        self.file = None
        self.csv_file = None
        self.csv_writer = None
        self.running = False
        self.thread = None
        self.stop_event = STOP_EVENT
        self.message_count = 0
        self.data_message_count = 0
        self.health_check_count = 0
        self.start_time = None
        
        # 통계
        self.closed_brep_count = 0
        self.invalid_geometry_count = 0
        
        # 디렉토리 구조 개선
        self.metrics_dir = os.path.join(self.save_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 타임스탬프를 파일명이 아닌 폴더명에만 사용
        self.log_file_path = os.path.join(self.metrics_dir, "state_reward_log.json")
        self.metrics_file_path = os.path.join(self.metrics_dir, "architecture_metrics.csv")
    
    def initialize(self):
        """초기화"""
        try:
            # ZMQ 초기화
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.set_hwm(1000)
            bind_address = f"tcp://*:{self.port}"
            self.socket.bind(bind_address)
            log_success(f"ZMQ PULL 소켓이 {bind_address}에 바인딩되었습니다.")
            
            # 로그 파일 초기화
            self.file = open(self.log_file_path, 'w', encoding='utf-8')
            self.file.write('[\n')
            
            # CSV 메트릭 파일 초기화
            self.csv_file = open(self.metrics_file_path, 'w', encoding='utf-8')
            header = "timestamp,step,is_closed_brep,excluded_from_training,bcr,far,winter_sunlight,sv_ratio,env_reward,human_reward,action1,action2,action3,action4"
            self.csv_file.write(header + "\n")
            self.csv_file.flush()
            
            log_info(f"📊 메트릭 CSV 파일 생성됨: {self.metrics_file_path}")
            
            self.running = True
            self.start_time = time.time()
            return True
            
        except Exception as e:
            log_error(f"상태 수신기 초기화 오류: {e}")
            self.cleanup()
            return False
    
    def start(self):
        """상태 수신 시작"""
        log_info(f"🔄 상태 수신기가 포트 {self.port}에서 실행 중입니다.")
        log_info(f"📝 데이터는 {self.log_file_path}에 저장됩니다.")
        log_info("\n👂 상태 수신 대기 중...")
        
        self.receive_loop()
    
    def receive_loop(self):
        """메시지 수신 루프"""
        while not self.stop_event.is_set():
            try:
                try:
                    message = self.socket.recv_string(flags=zmq.NOBLOCK)
                    self.process_message(message)
                except zmq.Again:
                    time.sleep(0.1)
                    
                    # 타임아웃 체크
                    elapsed = time.time() - self.start_time
                    if self.data_message_count == 0 and elapsed > 60:
                        log_warning(f"\n⏱️ 60초 동안 상태/보상 데이터가 없어 자동 종료합니다\n")
                        self.stop_event.set()
                        break
                        
            except Exception as e:
                log_error(f"메시지 수신 중 오류: {e}")
                time.sleep(0.5)
        
        self.cleanup()
    
    def process_message(self, message):
        """메시지 처리 - 기존 RL과 동일한 로직"""
        try:
            data = json.loads(message)
            self.message_count += 1
            
            log_debug(f"수신된 메시지: {message[:100]}...")
            
            # health_check 메시지 확인
            is_health_check = data.get("type") == "health_check"
            
            if is_health_check:
                self.health_check_count += 1
                return
            
            # 실제 데이터 메시지 처리
            self.data_message_count += 1
            
            # 상태 데이터가 있는지 확인
            if 'state' in data:
                # JSON 파일에 데이터 기록
                if self.data_message_count > 1:
                    self.file.write(',\n')
                self.file.write(json.dumps(data, ensure_ascii=False))
                
                # CSV 파일에 메트릭 기록
                if self.csv_file:
                    timestamp = data.get('timestamp', int(time.time() * 1000))
                    
                    # 상태 처리 - 기존 RL과 동일
                    state = data.get('state', [0, 0, 0, 0])
                    actions = data.get('action', [0, 0, 0, 0])
                    
                    # Closed Brep 확인
                    is_closed_brep = False
                    bcr = 0
                    far = 0
                    winter_sunlight = 0
                    sv_ratio = 0
                    
                    # 상태 형식 분석 및 처리
                    if isinstance(state, list):
                        # 숫자 값과 문자열 분리
                        numeric_values = [item for item in state if isinstance(item, (int, float))]
                        string_values = [item for item in state if isinstance(item, str)]
                        
                        log_debug(f"상태 숫자값: {numeric_values}")
                        if string_values:
                            log_debug(f"상태 문자열: {string_values}")
                        
                        # Closed Brep 문자열 확인
                        is_closed_brep = any("Closed Brep" in s for s in string_values) and not any(s == "0" for s in string_values)
                        
                        # 숫자 값 처리
                        if len(numeric_values) >= 4:
                            bcr = numeric_values[0]
                            far = numeric_values[1]
                            winter_sunlight = numeric_values[2]
                            sv_ratio = numeric_values[3]
                            is_closed_brep = True
                        elif len(numeric_values) == 3:
                            bcr = numeric_values[0]
                            far = numeric_values[1]
                            winter_sunlight = numeric_values[2]
                            sv_ratio = 1.0
                        
                        # 첫 번째 요소가 "0"인 경우 특별 처리
                        if len(state) > 0 and isinstance(state[0], str) and state[0] == "0":
                            is_closed_brep = False
                            bcr = far = winter_sunlight = sv_ratio = 0
                    
                    # 보상 계산
                    try:
                        if is_closed_brep:
                            # 환경 보상 계산
                            if self.env_reward_function:  # 이미 생성된 인스턴스 사용
                                state_4d = [bcr, far, winter_sunlight, sv_ratio]
                                env_reward, env_info = self.env_reward_function.calculate_reward(state_4d)
                            else:
                                env_reward = 1.0
                                env_info = {}
                            
                            # 인간 피드백 보상 계산
                            human_reward = 0.0
                            if self.human_reward_model:
                                human_reward = self.human_reward_model.predict_reward([bcr, far, winter_sunlight, sv_ratio])
                            
                            self.closed_brep_count += 1
                        else:
                            # 비정상 상태
                            env_reward = -30.0
                            human_reward = -30.0 # 환경 보상과 동일하게 설정
                            env_info = {"error": "Invalid geometry (Not a Closed Brep)"}
                            self.invalid_geometry_count += 1
                        
                        # 데이터에 보상 정보 추가
                        data['env_reward'] = env_reward
                        data['human_reward'] = human_reward
                        data['env_reward_info'] = env_info
                        data['is_closed_brep'] = is_closed_brep
                        
                    except Exception as e:
                        log_error(f"보상 계산 중 오류: {e}")
                        env_reward = -10.0
                        human_reward = 0.0
                        data['env_reward'] = env_reward
                        data['human_reward'] = human_reward
                        data['reward_error'] = str(e)
                    
                    # CSV 헤더 작성 (첫 번째 메시지일 때)
                    # if self.data_message_count == 1:
                    #     header = "timestamp,step,is_closed_brep,excluded_from_training,bcr,far,winter_sunlight,sv_ratio,env_reward,human_reward"
                    #     for i in range(len(actions[:4])):
                    #         header += f",action{i+1}"
                    #     self.csv_file.write(header + "\n")
                    
                    # CSV 라인 작성
                    excluded = 1 if not is_closed_brep else 0
                    csv_line = f"{timestamp},{self.data_message_count},{int(is_closed_brep)},{excluded},{bcr},{far},{winter_sunlight},{sv_ratio},{env_reward},{human_reward}"
                    
                    for action in actions[:4]:
                        csv_line += f",{action}"
                    
                    self.csv_file.write(csv_line + "\n")
                    self.csv_file.flush()
                    
                    # 상태를 큐에 추가
                    formatted_state = [bcr, far, winter_sunlight, sv_ratio]
                    
                    log_debug(f"큐에 상태 추가: state={formatted_state}, reward={env_reward}, is_closed_brep={is_closed_brep}")
                    
                    STATE_QUEUE.put((formatted_state, env_reward, data))
                    global LAST_STATE
                    LAST_STATE = (formatted_state, env_reward, data)
            else:
                log_warning(f"'state' 키가 없는 메시지: {message[:50]}...")
                        
        except json.JSONDecodeError:
            log_error(f"잘못된 JSON 형식: {message[:100]}...")
        except Exception as e:
            log_error(f"메시지 처리 중 오류: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        log_info("\n🧹 상태 수신기 리소스 정리 중...")
        self.running = False
        
        if self.file:
            try:
                self.file.write('\n]')
                self.file.close()
                log_info("📁 로그 파일이 닫혔습니다.")
            except:
                pass
            self.file = None
        
        if self.csv_file:
            try:
                self.csv_file.close()
            except:
                pass
            self.csv_file = None
        
        if self.socket:
            try:
                self.socket.close()
                log_info("🔌 ZMQ 소켓이 닫혔습니다.")
            except:
                pass
            self.socket = None
        
        if self.context:
            try:
                self.context.term()
                log_info("🔄 ZMQ 컨텍스트가 종료되었습니다.")
            except:
                pass
            self.context = None
        
        # 통계 출력
        elapsed = time.time() - self.start_time if self.start_time else 0
        log_info(f"\n📊 총 수신: {self.message_count}개 메시지 (실제 데이터: {self.data_message_count}개, health_check: {self.health_check_count}개)")
        log_info(f"⏱️ 실행 시간: {elapsed:.1f}초")
        rate = self.data_message_count / elapsed if elapsed > 0 else 0
        log_info(f"⚡ 평균 수신 속도: {rate:.1f}개/초")
        log_info(f"✅ Closed Brep 수: {self.closed_brep_count}")
        log_info(f"❌ Invalid geometry 수: {self.invalid_geometry_count}")

# === RLHF 통합 환경 (기존 ArchitectureOptimizationEnv 구조 유지) ===
class RLHFArchitectureOptimizationEnv(gym.Env):
    """RLHF가 통합된 건축 설계 최적화 환경"""
    
    def __init__(self, 
                 action_port=5556, 
                 state_port=5557,
                 reward_weights={'env': 0.7, 'human': 0.3},
                 human_reward_model_path=None,
                 bcr_limit=70.0,
                 far_min_limit=200.0,
                 far_max_limit=500.0,
                 slider_mins=None,
                 slider_maxs=None,
                 wait_time=5.0,
                 initial_wait=6.0):
        
        super(RLHFArchitectureOptimizationEnv, self).__init__()
        
        # 포트 설정
        self.action_port = action_port
        self.state_port = state_port
        
        # 보상 가중치
        self.reward_weights = reward_weights
        
        # 건축 제한
        self.bcr_limit = bcr_limit
        self.far_min_limit = far_min_limit
        self.far_max_limit = far_max_limit
        
        # 시간 설정
        self.wait_time = wait_time
        self.initial_wait = initial_wait
        
        # 슬라이더 범위
        self.slider_mins = np.array([10.0, 50.0, 0.0, 0.0]) if slider_mins is None else np.array(slider_mins)
        self.slider_maxs = np.array([25.0, 100.0, 100.0, 100.0]) if slider_maxs is None else np.array(slider_maxs)
        log_info(f"📏 슬라이더 실제 범위: 최소값={self.slider_mins}, 최대값={self.slider_maxs}")
        
        # 액션/상태 공간
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 10.0, 200000.0, 6.0]),
            dtype=np.float32
        )
        
        # 환경 보상 함수
        if REWARD_ADAPTER_AVAILABLE:
            self.env_reward_function = create_reward_function(
                reward_type="optimized",
                bcr_limit=bcr_limit,
                far_min=far_min_limit,
                far_max=far_max_limit,
                use_seasonal=True,
                debug=DEBUG
            )
        else:
            self.env_reward_function = None
        
        # 인간 피드백 보상 모델
        self.human_reward_model = None
        if human_reward_model_path:
            self.human_reward_model = HumanRewardModelLoader(human_reward_model_path)
            if not self.human_reward_model.load_model():
                self.human_reward_model = None
        
        # ZMQ 초기화
        self.context = None
        self.action_socket = None
        self._initialize_zmq()
        
        # 에피소드 추적
        self.episode_steps = 0
        self.total_steps = 0
        self.current_state = None
        self.current_env_reward = 0.0
        self.current_human_reward = 0.0
        self.current_info = {}
        
        # 통계 추적
        self.reward_stats = {
            'env_rewards': [],
            'human_rewards': [],
            'combined_rewards': [],
            'closed_brep_count': 0,
            'invalid_geometry_count': 0
        }
        
        log_info(f"🏗️ RLHF 건축 최적화 환경이 초기화되었습니다.")
        log_info(f"   - BCR 제한: {self.bcr_limit}%")
        log_info(f"   - FAR 허용 범위: {self.far_min_limit}% ~ {self.far_max_limit}%")
        log_info(f"   - 보상 가중치: 환경={self.reward_weights['env']}, 인간={self.reward_weights['human']}")
        log_info(f"   - 액션 공간: {self.action_space}")
        log_info(f"   - 상태 공간: {self.observation_space}")
    
    def _initialize_zmq(self):
        """ZMQ 통신 초기화"""
        try:
            self.context = zmq.Context()
            self.action_socket = self.context.socket(zmq.PUSH)
            self.action_socket.set_hwm(1000)
            self.action_socket.setsockopt(zmq.LINGER, 500)
            bind_address = f"tcp://*:{self.action_port}"
            self.action_socket.bind(bind_address)
            log_success(f"액션 전송 ZMQ PUSH 소켓이 {bind_address}에 바인딩되었습니다.")
            return True
        except Exception as e:
            log_error(f"ZMQ 초기화 오류: {e}")
            return False
    
    def _normalize_actions(self, actions):
        """액션 정규화"""
        actions_0_1 = (actions + 1.0) / 2.0
        real_actions = self.slider_mins + actions_0_1 * (self.slider_maxs - self.slider_mins)
        return real_actions
    
    def _send_action(self, action_values):
        """ZMQ를 통해 액션 전송"""
        try:
            action_list = action_values.tolist()
            action_json = json.dumps(action_list)
            self.action_socket.send_string(action_json)
            log_debug(f"액션 전송됨: {action_json}")
            return True
        except Exception as e:
            log_error(f"액션 전송 오류: {e}")
            return False
    
    def _wait_for_state(self, timeout=20.0):
        """상태 대기"""
        start_time = time.time()
        log_info("👂 새 상태 데이터 수신 대기 중...")
        log_info(f"⏱️ 새 상태 이벤트 대기 시작 (타임아웃: {int(timeout)}초)")
        
        while not STOP_EVENT.is_set():
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                log_info(f"⏱️ 새 상태 대기 타임아웃 ({elapsed:.1f}초), 최신 사용 가능 상태 반환")
                return False
            
            if int(elapsed) % 3 == 0 and elapsed % 3 < 0.1:
                log_info(f"⏱️ 상태 대기 중... ({elapsed:.1f}/{timeout}초)")
            
            try:
                state, reward, info = STATE_QUEUE.get(block=False)
                
                if len(state) == 4:
                    self.current_state = np.array(state, dtype=np.float32)
                else:
                    log_error(f"❌ 상태 형식 오류: {state}")
                    continue
                
                # RLHF: info에서 분리된 보상 정보 추출
                self.current_env_reward = info.get('env_reward', reward)
                self.current_human_reward = info.get('human_reward', 0.0)
                self.current_info = info
                return True
                
            except queue.Empty:
                time.sleep(0.1)
                continue
        
        return False

    def _get_last_state(self):
        """마지막으로 수신된 상태를 반환"""
        global LAST_STATE
        if LAST_STATE is not None:
            state, reward, info = LAST_STATE
            if len(state) == 4:
                env_reward = info.get('env_reward', reward)
                human_reward = info.get('human_reward', 0.0)
                return np.array(state, dtype=np.float32), env_reward, human_reward, info
        
        return np.zeros(4, dtype=np.float32), 0.0, 0.0, {}
    
    def _clear_state_queue(self):
        """상태 큐 비우기"""
        log_debug("이전 상태 큐 비우는 중...")
        count = 0
        while True:
            try:
                STATE_QUEUE.get(block=False)
                count += 1
            except queue.Empty:
                break
        
        if count > 0:
            log_debug(f"{count}개의 이전 상태 항목을 제거했습니다.")
    
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        self.episode_steps = 0
        
        # 환경 보상 함수 이전 상태 초기화
        if self.env_reward_function and hasattr(self.env_reward_function, 'reset_prev_state'):
            self.env_reward_function.reset_prev_state()
        
        # 초기 액션 생성
        initial_action = np.zeros(4, dtype=np.float32)
        
        real_action = self._normalize_actions(initial_action)
        log_info(f"🔄 환경 초기화: 정규화된 초기 액션={initial_action}, 실제 값={real_action}")
        
        self._send_action(real_action)
        
        log_info(f"⏱️ 초기화 중 {self.initial_wait}초 대기...")
        time.sleep(self.initial_wait)
        
        log_info("👂 초기 상태 데이터 수신 대기 중...")
        if self._wait_for_state(timeout=30.0):
            initial_state = self.current_state
            initial_info = self.current_info
        else:
            log_warning("초기 상태를 받지 못했습니다. 기본값 사용.")
            initial_state, _, _, initial_info = self._get_last_state()
        
        log_info(f"초기 상태: {initial_state}")
        
        return initial_state, initial_info
    
    def step(self, action):
        """환경 스텝 실행"""
        log_debug(f"원본 액션: {action}")
        
        # 액션 범위 체크
        if np.any(np.abs(action) > 1.0):
            log_warning(f"액션 범위를 벗어남: {action}")
            action = np.clip(action, -1.0, 1.0)
        
        log_info(f"\n🎮 스텝 {self.total_steps}, 정규화된 액션: {action}")
        
        # 실제 슬라이더 값으로 변환
        real_action = self._normalize_actions(action)
        log_info(f"📊 실제 슬라이더 값: {real_action}")
        
        # 이전 상태 큐 비우기
        self._clear_state_queue()
        
        # 액션 전송
        self._send_action(real_action)
        
        # 처리 대기
        log_info(f"⏱️ Grasshopper 처리를 위해 {self.wait_time}초 대기 중...")
        time.sleep(self.wait_time)
        
        # 재시도 로직 (Closed Brep 확인)
        max_retries = 5
        retries = 0
        valid_state_received = False
        
        while retries < max_retries and not valid_state_received:
            if self._wait_for_state():
                state = self.current_state
                env_reward = self.current_env_reward
                human_reward = self.current_human_reward
                info = self.current_info
                
                # Closed Brep 확인
                is_valid_state = info.get('is_closed_brep', True)
                
                if is_valid_state:
                    valid_state_received = True
                    self.reward_stats['closed_brep_count'] += 1
                else:
                    self.reward_stats['invalid_geometry_count'] += 1
                    retries += 1
                    if retries < max_retries:
                        # 재시도 전략
                        if retries == 1:
                            noise = np.random.normal(0, 0.1, size=action.shape)
                            new_action = np.clip(action + noise, -1.0, 1.0)
                        elif retries == 2:
                            new_action = np.clip(action * 0.95, -1.0, 1.0)
                        elif retries == 3:
                            new_action = np.clip(action * 1.05, -1.0, 1.0)
                        else:
                            noise = np.random.normal(0, 0.3, size=action.shape)
                            new_action = np.clip(action + noise, -1.0, 1.0)
                        
                        log_info(f"🔄 유효하지 않은 상태로 인한 재시도 {retries}/{max_retries}, 수정된 액션: {new_action}")
                        
                        real_action = self._normalize_actions(new_action)
                        self._send_action(real_action)
                        time.sleep(self.wait_time)
            else:
                log_warning("Grasshopper에서 상태를 받지 못했습니다. 이전 상태 사용.")
                state, env_reward, human_reward, info = self._get_last_state()
                valid_state_received = True
        
        # 모든 재시도 후에도 유효한 상태를 받지 못한 경우
        if not valid_state_received:
            log_warning(f"⚠️ {max_retries}번의 시도 후에도 유효한 상태를 받지 못했습니다.")
            truncated = True
            env_reward = -30.0
            human_reward = -10.0
            state, _, _, info = self._get_last_state()
            info['excluded_from_training'] = True
            info['reason'] = "Invalid geometry (Not a Closed Brep) after max retries"
            terminated = False
        else:
            truncated = False
            terminated = False
        
        # === RLHF 최종 보상 계산 ===
        print(f"🚨🚨🚨 DEBUG: 정규화 적용 전 - env: {env_reward:.4f}, human: {human_reward:.4f}")
        env_norm, human_norm = normalize_rewards_for_rlhf(env_reward, human_reward)
        print(f"🚨🚨🚨 DEBUG: 정규화 적용 후 - env_norm: {env_norm:.4f}, human_norm: {human_norm:.4f}")
        combined_reward = (
            self.reward_weights['env'] * env_norm +
            self.reward_weights['human'] * human_norm
        ) * 5.0
        print(f"🚨🚨🚨 DEBUG: 최종 결합 보상: {combined_reward:.4f}")
        
        # 통계 업데이트
        self.reward_stats['env_rewards'].append(env_reward)
        self.reward_stats['human_rewards'].append(human_reward)
        self.reward_stats['combined_rewards'].append(combined_reward)
        
        # 상태와 보상 로깅
        bcr = state[0] * 100.0 if len(state) > 0 else 0.0
        far = state[1] * 100.0 if len(state) > 1 else 0.0
        
        if len(state) >= 4:
            winter_sunlight = state[2]
            sv_ratio = state[3]
            log_info(f"📊 BCR: {bcr:.1f}%, FAR: {far:.1f}%, 겨울 일사량: {winter_sunlight:.2f}, 표면적 체적비: {sv_ratio:.4f}")
        
        is_closed_brep = info.get('is_closed_brep', True)
        log_info(f"💰 보상: 환경={env_reward:.4f}, 인간={human_reward:.4f}, 결합={combined_reward:.4f} (Closed Brep: {'Yes' if is_closed_brep else 'No'})")
        
        # 에피소드 종료 조건
        if not truncated:
            truncated = self.episode_steps >= 50
        
        # 정보 업데이트
        info.update({
            'episode_steps': self.episode_steps,
            'total_steps': self.total_steps,
            'actual_action': real_action.tolist(),
            'is_closed_brep': is_closed_brep,
            'env_reward': env_reward,
            'human_reward': human_reward,
            'combined_reward': combined_reward,
            'reward_weights': self.reward_weights
        })
        
        self.episode_steps += 1
        self.total_steps += 1
        
        return state, combined_reward, terminated, truncated, info
    
    def get_reward_statistics(self):
        """보상 통계 반환"""
        if not self.reward_stats['combined_rewards']:
            return {}
        
        return {
            'env_reward_mean': np.mean(self.reward_stats['env_rewards']),
            'human_reward_mean': np.mean(self.reward_stats['human_rewards']),
            'combined_reward_mean': np.mean(self.reward_stats['combined_rewards']),
            'env_reward_std': np.std(self.reward_stats['env_rewards']),
            'human_reward_std': np.std(self.reward_stats['human_rewards']),
            'combined_reward_std': np.std(self.reward_stats['combined_rewards']),
            'total_steps': len(self.reward_stats['combined_rewards']),
            'closed_brep_count': self.reward_stats['closed_brep_count'],
            'invalid_geometry_count': self.reward_stats['invalid_geometry_count'],
            'closed_brep_ratio': self.reward_stats['closed_brep_count'] / (self.reward_stats['closed_brep_count'] + self.reward_stats['invalid_geometry_count']) if (self.reward_stats['closed_brep_count'] + self.reward_stats['invalid_geometry_count']) > 0 else 0
        }
    
    def close(self):
        """환경 리소스 정리"""
        log_info("🧹 환경 리소스 정리 중...")
        
        if self.action_socket:
            try:
                self.action_socket.close()
                log_info("🔌 액션 ZMQ 소켓이 닫혔습니다.")
            except:
                pass
            self.action_socket = None
        
        if self.context:
            try:
                self.context.term()
                log_info("🔄 액션 ZMQ 컨텍스트가 종료되었습니다.")
            except:
                pass
            self.context = None
        
        super().close()

# === PPO 학습 함수 (기존 RL과 동일) ===
def train_rlhf_ppo(env, 
                    total_timesteps=10000, 
                    learning_rate=0.0003, 
                    save_dir=None, 
                    log_dir=None,
                    initial_model_path=None):
    """RLHF PPO 학습"""
    
    # 저장 디렉토리 생성
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    if log_dir is None:
        log_dir = os.path.join(DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 기존 코드 제거:
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_name = f"rlhf_ppo_architecture_{timestamp}"
    # model_path = os.path.join(save_dir, model_name)
    
    # 새로운 코드 (모델 경로 단순화):
    model_path = save_dir  # 바로 save_dir 사용
    
    # 체크포인트 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_path,
        name_prefix="checkpoint"
    )
    
    # 모델 설정
    if initial_model_path and os.path.exists(initial_model_path):
        log_info(f"🔄 기존 모델 로드 중: {initial_model_path}")
        model = PPO.load(initial_model_path, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                activation_fn=torch.nn.Tanh
            ),
            verbose=1
        )
    
    # 총 스텝 수에 대한 정보 출력
    log_info(f"🔄 총 {total_timesteps}개의 타임스텝 동안 학습합니다...")
    
    try:
        # 학습 시작
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # 최종 모델 저장
        final_model_path = os.path.join(model_path, "final_model")
        model.save(final_model_path)
        log_success(f"🎉 학습 완료! 모델이 {final_model_path}에 저장되었습니다.")
        
        return model, final_model_path
    except KeyboardInterrupt:
        log_info("\n⏹️ 학습이 중단되었습니다.")
        # 중단된 모델 저장
        interrupted_model_path = os.path.join(model_path, "interrupted_model")
        model.save(interrupted_model_path)
        log_info(f"🛑 중단된 모델이 {interrupted_model_path}에 저장되었습니다.")
        return model, interrupted_model_path
    except Exception as e:
        log_error(f"학습 중 오류 발생: {e}")
        return None, None

# === 메인 실행 함수 ===
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="RLHF 통합 건축 설계 최적화")
    
    # 모델 경로
    parser.add_argument('--reward-model-path', type=str, 
                       default=str(DEFAULT_PATHS['reward_model']),
                       help='인간 피드백 보상 모델 경로')
    parser.add_argument('--initial-model', type=str, default=None,
                       help='연속 학습을 위한 초기 PPO 모델 경로')
    
    # 보상 가중치
    parser.add_argument('--env-weight', type=float, default=0.7,
                       help='환경 보상 가중치')
    parser.add_argument('--human-weight', type=float, default=0.3,
                       help='인간 피드백 보상 가중치')
    
    # 학습 설정
    parser.add_argument('--timesteps', type=int, default=3000,
                       help='총 학습 타임스텝')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='학습률')
    
    # 포트 설정
    parser.add_argument('--action-port', type=int, default=5556,
                       help='ZMQ 액션 포트 (Grasshopper로)')
    parser.add_argument('--state-port', type=int, default=5557,
                       help='ZMQ 상태 포트 (Grasshopper에서)')
    
    # 출력 설정
    parser.add_argument('--output-dir', type=str, 
                       default=str(DEFAULT_PATHS['base_output_dir']),
                       help='결과 저장 디렉토리')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 로깅 활성화')
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    global DEBUG
    DEBUG = args.debug
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 보상 가중치 설정
    reward_weights = {'env': args.env_weight, 'human': args.human_weight}
    
    # 타임스탬프 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 헤더 출력
    print("\n" + "="*80)
    print(f"🏗️  RLHF 통합 건축 설계 최적화")
    print("="*80)
    print(f"🕒 세션 ID: {timestamp}")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"💻 학습 디바이스: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"⚖️  보상 가중치: 환경={args.env_weight}, 인간={args.human_weight}")
    print(f"📊 학습 타임스텝: {args.timesteps:,}")
    print(f"🧠 인간 보상 모델: {os.path.basename(args.reward_model_path)}")
    if args.initial_model:
        print(f"📦 초기 모델: {os.path.basename(args.initial_model)}")
    print("")
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("[1/4] StateReceiver 초기화 중...")
        
        # StateReceiver 초기화 및 시작
        human_reward_model = HumanRewardModelLoader(args.reward_model_path)
        if not human_reward_model.load_model():
            human_reward_model = None
            log_warning("인간 보상 모델 로드 실패, 더미 보상 사용")
        
        state_receiver = RLHFStateReceiver(
            port=args.state_port,
            save_dir=output_dir / 'zmq_logs',
            human_reward_model=human_reward_model
        )
        
        if not state_receiver.initialize():
            log_error("StateReceiver 초기화 실패")
            return
        
        # StateReceiver 스레드 시작
        receiver_thread = threading.Thread(target=state_receiver.start)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        print("[2/4] RLHF 환경 초기화 중...")
        
        # RLHF 환경 생성
        env = RLHFArchitectureOptimizationEnv(
            action_port=args.action_port,
            state_port=args.state_port,
            reward_weights=reward_weights,
            human_reward_model_path=args.reward_model_path,
            wait_time=5.0,
            initial_wait=6.0
        )
        
        print("[3/4] PPO 학습 시작...")
        
        # PPO 학습 실행
        model, model_path = train_rlhf_ppo(
            env=env,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            save_dir=output_dir / 'models',
            log_dir=output_dir / 'logs',
            initial_model_path=args.initial_model
        )
        
        if model and not STOP_EVENT.is_set():
            print("[4/4] 분석 보고서 생성 중...")
            
            # 통계 수집
            env_stats = env.get_reward_statistics()
            
            # 결과 보고서 생성
            results = {
                'session_info': {
                    'timestamp': timestamp,
                    'reward_weights': reward_weights,
                    'total_timesteps': args.timesteps,
                    'reward_model': args.reward_model_path,
                    'initial_model': args.initial_model,
                    'final_model': model_path
                },
                'training_statistics': env_stats,
                'final_performance': {
                    'env_reward_mean': env_stats.get('env_reward_mean', 0),
                    'human_reward_mean': env_stats.get('human_reward_mean', 0),
                    'combined_reward_mean': env_stats.get('combined_reward_mean', 0),
                    'closed_brep_ratio': env_stats.get('closed_brep_ratio', 0),
                    'total_steps': env_stats.get('total_steps', 0)
                }
            }
            
            # 결과 저장
            results_path = output_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            log_success(f"📊 학습 결과가 {results_path}에 저장되었습니다.")
        
        print(f"\n✅ 학습이 성공적으로 완료되었습니다!")
        print(f"📁 결과가 저장된 위치: {output_dir}")
        
    except Exception as e:
        log_error(f"학습 실패: {e}")
        raise
        
    finally:
        # 리소스 정리
        print("\n🧹 리소스 정리 중...")
        
        # 환경 정리
        try:
            env.close()
        except:
            pass
        
        # StateReceiver 정리
        if 'state_receiver' in locals():
            try:
                STOP_EVENT.set()
                if 'receiver_thread' in locals() and receiver_thread.is_alive():
                    receiver_thread.join(timeout=5.0)
                state_receiver.cleanup()
            except:
                pass
        
        print("💯 프로그램 종료")

if __name__ == "__main__":
    main()