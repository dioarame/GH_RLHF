#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 강화학습 시스템
ZMQ를 통해 Grasshopper와 통신하여 건축 설계 파라미터를 최적화합니다.
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
from typing import Dict, List, Tuple, Union, Optional, Any
from reward_adapter import create_reward_function

# 초기 설정 및 글로벌 변수
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ZMQ_LOGS_DIR = os.path.join(DATA_DIR, "zmq_logs")
STATE_QUEUE = queue.Queue()
STOP_EVENT = threading.Event()
LAST_STATE = None
DEBUG = False

# 로깅 함수들
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

# 신호 핸들러
def signal_handler(sig, frame):
    log_info("\n🛑 사용자에 의해 학습이 중단되었습니다.")
    STOP_EVENT.set()

# ZMQ 상태 수신기 클래스
class StateReceiver:
    def __init__(self, port=5557, save_dir=ZMQ_LOGS_DIR, reward_function=None):
        self.port = port
        self.save_dir = save_dir
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
        
        # 외부에서 전달받은 보상 함수 사용
        self.reward_function = reward_function
        
        # 보상 함수가 없으면 기본 함수 생성
        if self.reward_function is None:
            from reward_adapter import create_reward_function
            self.reward_function = create_reward_function(
                reward_type="optimized",  # 또는 "original"
                bcr_limit=70.0,
                far_min=200.0,
                far_max=500.0,
                use_seasonal=True,
                debug=DEBUG
            )
        
        # 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 타임스탬프 생성
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.save_dir, f"state_reward_log_{self.timestamp}.json")
        self.metrics_file_path = os.path.join(self.save_dir, f"architecture_metrics_{self.timestamp}.csv")
    
    def initialize(self):
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
            self.file.write('[\n')  # JSON 배열 시작
            
            # CSV 메트릭 파일 초기화
            self.csv_file = open(self.metrics_file_path, 'w', encoding='utf-8')
            
            # 헤더는 process_message에서 첫 메시지 처리 시 작성하도록 변경
            # header = "timestamp,step,bcr,far,sunlight,reward,action1,action2,action3,action4\n"
            # self.csv_file.write(header)
            
            log_info(f"📊 메트릭 CSV 파일 생성됨: {self.metrics_file_path}")
            
            self.running = True
            self.start_time = time.time()
            return True
        except Exception as e:
            log_error(f"상태 수신기 초기화 오류: {e}")
            self.cleanup()
            return False
    
    def start(self):
        log_info(f"🔄 상태 수신기가 포트 {self.port}에서 실행 중입니다.")
        log_info(f"📝 데이터는 {self.log_file_path}에 저장됩니다.")
        log_info("\n👂 상태 수신 대기 중...")
        
        # 메인 스레드에서 직접 실행
        self.receive_loop()
    
    def receive_loop(self):
        if not self.running:
            return
        
        while not self.stop_event.is_set():
            try:
                # 비차단 모드로 메시지 수신 (짧은 대기 시간)
                try:
                    message = self.socket.recv_string(flags=zmq.NOBLOCK)
                    self.process_message(message)
                except zmq.Again:
                    # 메시지가 없으면 대기
                    time.sleep(0.1)
                    
                    # 일정 시간 동안 데이터가 없으면 자동 종료
                    elapsed = time.time() - self.start_time
                    if self.data_message_count == 0 and elapsed > 60:
                        log_warning(f"\n⏱️ 60초 동안 상태/보상 데이터가 없어 자동 종료합니다\n")
                        self.stop_event.set()
                        break
                    
            except Exception as e:
                log_error(f"메시지 수신 중 오류: {e}")
                # 잠시 대기 후 계속
                time.sleep(0.5)
        
        # 수신 루프 종료 후 정리
        self.cleanup()
    
    # StateReceiver 클래스의 process_message 메서드 부분 수정

# StateReceiver 클래스 내에서 process_message 메서드를 올바르게 정의
# 파일: StateReceiver 클래스 수정

class StateReceiver:
    def __init__(self, port=5557, save_dir=ZMQ_LOGS_DIR, reward_function=None):
        self.port = port
        self.save_dir = save_dir
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
        
        # 외부에서 전달받은 보상 함수 사용
        self.reward_function = reward_function
        
        # 보상 함수가 없으면 기본 함수 생성
        if self.reward_function is None:
            from reward_adapter import create_reward_function
            self.reward_function = create_reward_function(
                reward_type="optimized",  # 또는 "original"
                bcr_limit=70.0,
                far_min=200.0,
                far_max=500.0,
                use_seasonal=True,
                debug=DEBUG
            )
        
        # 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 타임스탬프 생성
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.save_dir, f"state_reward_log_{self.timestamp}.json")
        self.metrics_file_path = os.path.join(self.save_dir, f"architecture_metrics_{self.timestamp}.csv")
    
    def initialize(self):
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
            self.file.write('[\n')  # JSON 배열 시작
            
            # CSV 메트릭 파일 초기화
            self.csv_file = open(self.metrics_file_path, 'w', encoding='utf-8')
            
            # 헤더는 process_message에서 첫 메시지 처리 시 작성하도록 변경
            
            log_info(f"📊 메트릭 CSV 파일 생성됨: {self.metrics_file_path}")
            
            self.running = True
            self.start_time = time.time()
            return True
        except Exception as e:
            log_error(f"상태 수신기 초기화 오류: {e}")
            self.cleanup()
            return False
    
    def start(self):
        log_info(f"🔄 상태 수신기가 포트 {self.port}에서 실행 중입니다.")
        log_info(f"📝 데이터는 {self.log_file_path}에 저장됩니다.")
        log_info("\n👂 상태 수신 대기 중...")
        
        # 메인 스레드에서 직접 실행
        self.receive_loop()
    
    def receive_loop(self):
        if not self.running:
            return
        
        while not self.stop_event.is_set():
            try:
                # 비차단 모드로 메시지 수신 (짧은 대기 시간)
                try:
                    message = self.socket.recv_string(flags=zmq.NOBLOCK)
                    self.process_message(message)
                except zmq.Again:
                    # 메시지가 없으면 대기
                    time.sleep(0.1)
                    
                    # 일정 시간 동안 데이터가 없으면 자동 종료
                    elapsed = time.time() - self.start_time
                    if self.data_message_count == 0 and elapsed > 60:
                        log_warning(f"\n⏱️ 60초 동안 상태/보상 데이터가 없어 자동 종료합니다\n")
                        self.stop_event.set()
                        break
                    
            except Exception as e:
                log_error(f"메시지 수신 중 오류: {e}")
                # 잠시 대기 후 계속
                time.sleep(0.5)
        
        # 수신 루프 종료 후 정리
        self.cleanup()
    
    # 여기서 process_message 메서드가 적절하게 들여쓰기 되어야 함
    # StateReceiver 클래스의 process_message 메서드 수정
    def process_message(self, message):
        try:
            data = json.loads(message)
            self.message_count += 1
            
            # 디버그 모드에서 모든 메시지 출력
            if DEBUG:
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
                    
                    # 상태 처리 개선 - Closed Brep 확인
                    state = data.get('state', [0, 0, 0, 0])
                    actions = data.get('action', [0, 0, 0, 0])
                    
                    # 새로운 방식 - Closed Brep 확인
                    is_closed_brep = False
                    bcr = 0
                    far = 0
                    winter_sunlight = 0
                    sv_ratio = 0
                    
                    # 상태 형식 분석 및 처리 (개선된 버전)
                    if isinstance(state, list):
                        # 숫자 값과 문자열 분리
                        numeric_values = [item for item in state if isinstance(item, (int, float))]
                        string_values = [item for item in state if isinstance(item, str)]
                        
                        if DEBUG:
                            log_debug(f"상태 숫자값: {numeric_values}")
                            if string_values:
                                log_debug(f"상태 문자열: {string_values}")
                        
                        # Closed Brep 문자열 확인
                        is_closed_brep = any("Closed Brep" in s for s in string_values) and not any(s == "0" for s in string_values)
                        
                        # 숫자 값 처리 (Grasshopper 형식에 맞게)
                        if len(numeric_values) >= 4:
                            bcr = numeric_values[0]
                            far = numeric_values[1]
                            winter_sunlight = numeric_values[2]
                            sv_ratio = numeric_values[3]
                            is_closed_brep = True  # 충분한 숫자 값이 있으면 정상으로 간주
                        elif len(numeric_values) == 3:
                            bcr = numeric_values[0]
                            far = numeric_values[1]
                            winter_sunlight = numeric_values[2]
                            sv_ratio = 1.0  # 기본값
                        
                        # 첫 번째 요소가 "0"인 경우 특별 처리
                        if len(state) > 0 and isinstance(state[0], str) and state[0] == "0":
                            is_closed_brep = False
                            bcr = far = winter_sunlight = sv_ratio = 0
                    
                    # 정확한 보상 계산 - 환경과 공유된 보상 함수 사용
                    try:
                        # 상태가 정상인 경우(Closed Brep)에만 실제 보상 계산
                        if is_closed_brep:
                            # 4차원 상태 벡터 구성 - 올바른 순서로
                            state_4d = [bcr, far, winter_sunlight, sv_ratio]
                            # 보상 함수 호출
                            reward_value, reward_info = self.reward_function.calculate_reward(state_4d)
                        else:
                            # 비정상 상태(Closed Brep이 아님)는 큰 패널티
                            reward_value = -30.0
                            reward_info = {"error": "Invalid geometry (Not a Closed Brep)"}
                            
                        # 여기서 계산된 보상을, 이후 상태 업데이트에 사용할 수 있도록 데이터에 추가
                        data['calculated_reward'] = reward_value
                        data['reward_info'] = reward_info
                        data['is_closed_brep'] = is_closed_brep
                        
                    except Exception as e:
                        log_error(f"보상 계산 중 오류: {e}")
                        # 오류 발생 시 간단한 대체 계산 사용
                        reward_value = -10.0 if not is_closed_brep else 0
                        data['calculated_reward'] = reward_value
                        data['reward_info'] = {"error": str(e)}
                        data['is_closed_brep'] = is_closed_brep
                    
                    # CSV 헤더가 없을 경우 추가
                    if self.data_message_count == 1:
                        header = "timestamp,step,is_closed_brep,excluded_from_training,bcr,far,winter_sunlight,sv_ratio,reward"
                        
                        # 액션 헤더 추가
                        for i in range(len(actions[:4])):
                            header += f",action{i+1}"
                        
                        self.csv_file.write(header + "\n")

                    # CSV 라인 작성 (수정: 표면적 체적비 추가)
                    excluded = 1 if not is_closed_brep else 0  # 유효하지 않은 상태는 학습에서 제외
                    csv_line = f"{timestamp},{self.data_message_count},{int(is_closed_brep)},{excluded},{bcr},{far},{winter_sunlight},{sv_ratio},{reward_value}"
                    
                    # 액션 값 추가
                    for action in actions[:4]:
                        csv_line += f",{action}"
                    
                    self.csv_file.write(csv_line + "\n")
                    self.csv_file.flush()
                    
                    # 상태와 보상을 큐에 추가 - 올바른 상태 벡터
                    formatted_state = [bcr, far, winter_sunlight, sv_ratio]
                    
                    if DEBUG:
                        log_debug(f"큐에 상태 추가: state={formatted_state}, reward={reward_value}, is_closed_brep={is_closed_brep}")
                    
                    STATE_QUEUE.put((formatted_state, reward_value, data))
                    global LAST_STATE
                    LAST_STATE = (formatted_state, reward_value, data)
            else:
                log_warning(f"'state' 키가 없는 메시지: {message[:50]}...")
                        
        except json.JSONDecodeError:
            log_error(f"잘못된 JSON 형식: {message[:100]}...")
        except Exception as e:
            log_error(f"메시지 처리 중 오류: {e}")
    
    def cleanup(self):
        log_info("\n🧹 상태 수신기 리소스 정리 중...")
        self.running = False
        
        # 파일 정리
        if self.file:
            try:
                self.file.write('\n]')  # JSON 배열 종료
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
        
        # ZMQ 소켓 정리
        if self.socket:
            try:
                self.socket.close()
                log_info("🔌 ZMQ 소켓이 닫혔습니다.")
            except:
                pass
            self.socket = None
        
        # ZMQ 컨텍스트 정리
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

# 건축 최적화 환경 클래스
class ArchitectureOptimizationEnv(gym.Env):
    """건축 설계 최적화를 위한 강화학습 환경"""
    
    # __init__ 메서드 중복을 제거하고 하나만 유지
    def __init__(self, action_port=5556, state_port=5557, 
        bcr_limit=70.0, far_min_limit=200.0, far_max_limit=500.0,
        slider_mins=None, slider_maxs=None, 
        use_seasonal_reward=False,
        reward_type="optimized",
        wait_time=5.0, initial_wait=6.0):
        super(ArchitectureOptimizationEnv, self).__init__()
        
        # ZMQ 설정
        self.action_port = action_port
        self.state_port = state_port
        self.context = None
        self.action_socket = None
        
        # 건축 제한 설정
        self.bcr_limit = bcr_limit
        self.far_min_limit = far_min_limit
        self.far_max_limit = far_max_limit
        
        # 시간 설정
        self.wait_time = wait_time
        self.initial_wait = initial_wait
        
        # 슬라이더 범위 설정 (기본값)
        self.slider_mins = np.array([10.0, 50.0, 0.0, 0.0]) if slider_mins is None else np.array(slider_mins)
        self.slider_maxs = np.array([25.0, 100.0, 100.0, 100.0]) if slider_maxs is None else np.array(slider_maxs)
        log_info(f"📏 슬라이더 실제 범위: 최소값={self.slider_mins}, 최대값={self.slider_maxs}")
        
        # 계절별 보상 함수 사용 여부 저장
        self.use_seasonal_reward = use_seasonal_reward
        
        # reward_type 저장
        self.reward_type = reward_type
        
        # 보상 함수 초기화 - 항상 동일한 방식으로 초기화하고, 
        # use_seasonal_reward에 따라 가중치만 조정
        try:
            from reward_adapter import create_reward_function
            self.reward_function = create_reward_function(
                reward_type=self.reward_type,  # 전달받은 reward_type 사용
                bcr_limit=bcr_limit,
                far_min=far_min_limit,
                far_max=far_max_limit,
                use_seasonal=use_seasonal_reward,
                debug=DEBUG
            )
            print(f"{self.reward_type} 보상 함수를 사용합니다.")
        except ImportError as e:
            print(f"reward_adapter를 임포트할 수 없습니다: {e}")
            # 기본 보상 함수 생성 (모듈 없이 인라인으로 정의)
            self.reward_function = self._create_default_reward_function(
                bcr_limit=bcr_limit,
                far_min=far_min_limit, 
                far_max=far_max_limit,
                use_seasonal=use_seasonal_reward
            )
        
        # 액션 공간: 4개의 정규화된 슬라이더 값 (-1.0 ~ 1.0)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(4,),
            dtype=np.float32
        )
        
        # 상태 공간: 계절별/일반 보상에 따라 다르게 정의
        if self.use_seasonal_reward:
            # [BCR, FAR, WinterTime, SV_Ratio]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0]),
                high=np.array([1.0, 10.0, 200000.0, 6.0]),  # SV_Ratio 최대 6.0
                dtype=np.float32
            )
        else:
            # [BCR, FAR, 일사량]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),
                high=np.array([1.0, 10.0, 1.0]),  # 정규화된 값 범위
                dtype=np.float32
            )
        
        # 에피소드 추적
        self.episode_steps = 0
        self.total_steps = 0
        self.current_state = None  # 상태 공간 차원에 맞게 초기화
        self.current_reward = 0.0
        self.current_info = {}
        
        # ZMQ 초기화
        self._initialize_zmq()
        
        log_info(f"🏗️ 건축 최적화 환경이 초기화되었습니다.")
        log_info(f"   - BCR 제한: {self.bcr_limit}%")
        log_info(f"   - FAR 허용 범위: {self.far_min_limit}% ~ {self.far_max_limit}%")
        log_info(f"   - 보상 함수 유형: {'계절별' if self.use_seasonal_reward else '일반'} ({self.reward_type})")
        log_info(f"   - 액션 공간: {self.action_space}")
        log_info(f"   - 상태 공간: {self.observation_space}")
    
    def _initialize_zmq(self):
        """ZMQ 통신 초기화"""
        try:
            self.context = zmq.Context()
            
            # PUSH 소켓 초기화 (액션 전송용)
            self.action_socket = self.context.socket(zmq.PUSH)
            self.action_socket.set_hwm(1000)  # High Water Mark 설정
            self.action_socket.setsockopt(zmq.LINGER, 500)  # Linger 설정 (500ms)
            bind_address = f"tcp://*:{self.action_port}"
            self.action_socket.bind(bind_address)
            log_success(f"액션 전송 ZMQ PUSH 소켓이 {bind_address}에 바인딩되었습니다.")
            
            return True
        except Exception as e:
            log_error(f"ZMQ 초기화 오류: {e}")
            return False

    def _troubleshoot_connection(self):
            """연결 문제 해결을 위한 진단 기능"""
            log_info("\n🔍 ZMQ 연결 문제 진단 중...")
            
            # 1. 포트 가용성 확인
            import socket
            try:
                # 바인딩된 포트이므로 여기서는 사용 중이어야 함
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(('localhost', self.action_port))
                if result == 0:
                    log_info(f"✓ 포트 {self.action_port}가 사용 중입니다 (정상).")
                else:
                    log_warning(f"⚠️ 포트 {self.action_port}가 사용 중이지 않습니다. ZMQ 바인딩이 실패했을 수 있습니다.")
                s.close()
            except:
                log_warning(f"⚠️ 포트 {self.action_port} 확인 중 오류가 발생했습니다.")
            
            # 2. ZeroMQ 버전 확인
            log_info(f"✓ ZeroMQ 버전: {zmq.zmq_version()}")
            log_info(f"✓ PyZMQ 버전: {zmq.__version__}")
            
            # 3. Grasshopper 연결 테스트
            try:
                test_data = {"type": "diagnostic", "timestamp": int(time.time() * 1000)}
                json_data = json.dumps(test_data)
                self.action_socket.send_string(json_data)
                log_info(f"✓ 진단 메시지 전송됨: {json_data}")
            except Exception as e:
                log_error(f"⚠️ 진단 메시지 전송 실패: {e}")
            
            # 4. 메시지 형식 확인
            test_actions = np.array([0.0, 0.0, 0.0, 0.0])
            real_actions = self._normalize_actions(test_actions)
            log_info(f"✓ 테스트 액션 변환: 입력=[-1.0, -1.0, -1.0, -1.0] → 출력={self._normalize_actions(np.array([-1.0, -1.0, -1.0, -1.0]))}")
            log_info(f"✓ 테스트 액션 변환: 입력=[0.0, 0.0, 0.0, 0.0] → 출력={real_actions}")
            log_info(f"✓ 테스트 액션 변환: 입력=[1.0, 1.0, 1.0, 1.0] → 출력={self._normalize_actions(np.array([1.0, 1.0, 1.0, 1.0]))}")
            
            # 5. 연결 제안
            log_info("\n🔧 문제 해결 제안:")
            log_info("1. Grasshopper의 ZmqListener 컴포넌트가 올바른 포트(5556)를 사용하고 있는지 확인하세요.")
            log_info("2. ZmqListener와 ZmqStateSender 컴포넌트의 'Run' 파라미터가 모두 True로 설정되어 있는지 확인하세요.")
            log_info("3. Rhino와 Grasshopper를 재시작한 후 다시 시도하세요.")
            log_info("4. 필요한 경우 방화벽 설정을 확인하고 포트 5556과 5557이 허용되어 있는지 확인하세요.")
            log_info("")

    def _normalize_actions(self, actions):
        """액션을 정규화합니다 (-1.0 ~ 1.0 -> 실제 슬라이더 범위)"""
        # 정규화된 값을 0.0 ~ 1.0 범위로 변환
        actions_0_1 = (actions + 1.0) / 2.0
        
        # 0.0 ~ 1.0 값을 실제 슬라이더 범위로 변환
        real_actions = self.slider_mins + actions_0_1 * (self.slider_maxs - self.slider_mins)
        
        return real_actions
    
    def _send_action(self, action_values):
        """ZMQ를 통해 Grasshopper로 액션을 전송합니다"""
        try:
            # 액션 값을 리스트로 변환
            action_list = action_values.tolist()
            
            # 액션 JSON 형식으로 직렬화
            action_json = json.dumps(action_list)
            
            # 액션 전송
            self.action_socket.send_string(action_json)
            log_debug(f"액션 전송됨: {action_json}")
            return True
        except Exception as e:
            log_error(f"액션 전송 오류: {e}")
            return False
    
    def _wait_for_state(self, timeout=20.0):
        """ZMQ를 통해 Grasshopper에서 상태를 수신합니다"""
        start_time = time.time()
        log_info("👂 새 상태 데이터 수신 대기 중...")
        log_info(f"⏱️ 새 상태 이벤트 대기 시작 (타임아웃: {int(timeout)}초)")
        
        while not STOP_EVENT.is_set():
            elapsed = time.time() - start_time
            
            # 타임아웃 확인
            if elapsed > timeout:
                log_info(f"⏱️ 새 상태 대기 타임아웃 ({elapsed:.1f}초), 최신 사용 가능 상태 반환")
                return False
            
            # 주기적인 대기 메시지
            if int(elapsed) % 3 == 0 and elapsed % 3 < 0.1:  # 3초마다만 출력
                log_info(f"⏱️ 상태 대기 중... ({elapsed:.1f}/{timeout}초)")
            
            # 큐에서 상태 확인
            try:
                state, reward, info = STATE_QUEUE.get(block=False)
                
                # 상태 형식 확인 및 처리
                if self.use_seasonal_reward:
                    # 계절별 보상을 사용하는 경우, 4개 요소 필요 [BCR, FAR, Winter, SV_Ratio]
                    if len(state) == 4:
                        self.current_state = np.array(state, dtype=np.float32)
                    elif len(state) == 3:
                        # SV_Ratio가 없는 경우, 기본값 1.0 사용
                        bcr, far, winter = state
                        self.current_state = np.array([bcr, far, winter, 1.0], dtype=np.float32)
                        log_warning("⚠️ 표면적 체적비가 없습니다. 기본값 1.0 사용.")
                    else:
                        log_error(f"❌ 상태 형식 오류: {state}")
                        continue
                else:
                    # 일반 보상을 사용하는 경우, 3개 요소면 충분 [BCR, FAR, 일조량]
                    if len(state) >= 3:
                        self.current_state = np.array(state[:3], dtype=np.float32)
                    else:
                        log_error(f"❌ 상태 형식 오류: {state}")
                        continue
                
                self.current_reward = reward
                self.current_info = info
                return True
            except queue.Empty:
                # 큐가 비어있으면 잠시 대기
                time.sleep(0.1)
                continue
        
        # 종료 이벤트가 설정된 경우
        return False

    def _get_last_state(self):
        """마지막으로 수신된 상태를 반환합니다"""
        global LAST_STATE
        if LAST_STATE is not None:
            state, reward, info = LAST_STATE
            
            # 계절별 보상 사용 여부에 따라 처리
            if self.use_seasonal_reward:
                if len(state) == 4:
                    return np.array(state, dtype=np.float32), reward, info
                elif len(state) == 3:
                    # 3개 요소만 있는 경우, 임시로 4번째 요소 추가
                    bcr, far, sunlight = state
                    return np.array([bcr, far, sunlight, sunlight], dtype=np.float32), reward, info
            else:
                # 일반 보상을 사용하는 경우, 처음 3개 요소만 사용
                if len(state) >= 3:
                    return np.array(state[:3], dtype=np.float32), reward, info
                
        # 기본 상태 반환 (계절별 보상 사용 여부에 따라)
        if self.use_seasonal_reward:
            return np.zeros(4, dtype=np.float32), 0.0, {}
        else:
            return np.zeros(3, dtype=np.float32), 0.0, {}
    
    # 보상 계산 시 상세 로그 추가
    def _calculate_reward(self, state):
        """개선된 보상 함수를 사용하여 보상 계산"""
        # 상태 형식 맞추기
        if not self.use_seasonal_reward and len(state) == 3:
            # 일반 보상 사용 시 3개 요소만 있으면, 동일한 일조량으로 여름/겨울 설정
            bcr, far, sunlight = state
            state_4d = [bcr, far, sunlight, sunlight]
        else:
            state_4d = state
        
        # 계절별 보상 함수 사용 (이전 상태는 함수 내부에서 관리)
        reward, info = self.reward_function.calculate_reward(state_4d)
        
        # 디버그 로깅 (확장된 부분)
        if DEBUG:
            log_debug(f"===== 보상 상세 계산 =====")
            log_debug(f"입력 상태(4D): {state_4d}")
            log_debug(f"계산된 보상: {reward}")
            log_debug(f"BCR 점수: {info['bcr_score']:.4f}, 가중치 적용: {info['weighted_bcr_reward']:.2f}")
            log_debug(f"FAR 점수: {info['far_score']:.4f}, 가중치 적용: {info['weighted_far_reward']:.2f}")
            log_debug(f"겨울 점수: {info['winter_score']:.4f}, 가중치 적용: {info['weighted_winter_reward']:.2f}")
            log_debug(f"표면적 체적비 점수: {info['sv_ratio_score']:.4f}, 가중치 적용: {info['weighted_sv_ratio_reward']:.2f}")
            log_debug(f"기본 보상(패널티 전): {info['base_reward_before_penalty']:.2f}")
            
            if 'legality_penalty' in info and info['legality_penalty'] > 0:
                log_debug(f"법적 위반 패널티: {info['legality_penalty']:.2f}")
                log_debug(f"BCR 위반: {info['bcr_violated']}, FAR 최소: {info['far_min_violated']}, FAR 최대: {info['far_max_violated']}")
            
            if 'improvement_bonus' in info and info['improvement_bonus'] != 0:
                log_debug(f"개선 보너스: {info['improvement_bonus']:.2f}")
            
            log_debug(f"최종 보상: {reward:.2f}")
        
        return reward, info
    
    def _clear_state_queue(self):
        """STATE_QUEUE의 모든 항목을 비워 이전 상태를 제거합니다"""
        log_debug("이전 상태 큐 비우는 중...")
        count = 0
        
        # 큐의 모든 항목 제거
        global STATE_QUEUE
        while True:
            try:
                STATE_QUEUE.get(block=False)
                count += 1
            except queue.Empty:
                break
        
        if count > 0:
            log_debug(f"{count}개의 이전 상태 항목을 제거했습니다.")
    
    def reset(self, seed=None, options=None):
        """환경을 초기화하고 초기 상태를 반환합니다"""
        super().reset(seed=seed)
        self.episode_steps = 0
        
        # 계절별 보상 함수를 사용하는 경우, 이전 상태 초기화
        if self.use_seasonal_reward:
            self.reward_function.reset_prev_state()
        
        # 초기 액션 생성 (모든 슬라이더를 중간값으로 설정)
        initial_action = np.zeros(4, dtype=np.float32)
        
        # 액션을 실제 슬라이더 값로 변환
        real_action = self._normalize_actions(initial_action)
        log_info(f"🔄 환경 초기화: 정규화된 초기 액션={initial_action}, 실제 값={real_action}")
        
        # 액션 전송
        self._send_action(real_action)
        
        # Grasshopper가 처리할 시간을 줌
        log_info(f"⏱️ 초기화 중 {self.initial_wait}초 대기...")
        time.sleep(self.initial_wait)
        
        # 초기 상태 수신
        log_info("👂 초기 상태 데이터 수신 대기 중...")
        if self._wait_for_state(timeout=30.0):
            initial_state = self.current_state
            initial_info = self.current_info
        else:
            log_warning("초기 상태를 받지 못했습니다. 기본값 사용.")
            initial_state, _, initial_info = self._get_last_state()
        
        log_info(f"초기 상태: {initial_state}")
        
        return initial_state, initial_info

    def step(self, action):
    # 액션 로깅 및 처리 (기존 코드 유지)
        log_debug(f"원본 액션: {action}")
        
        # 액션 범위 체크
        if np.any(np.abs(action) > 1.0):
            log_warning(f"액션 범위를 벗어남: {action}")
            action = np.clip(action, -1.0, 1.0)
        
        # 정규화된 액션 출력
        log_info(f"\n🎮 스텝 {self.total_steps}, 정규화된 액션: {action}")
        
        # 실제 슬라이더 값으로 변환
        real_action = self._normalize_actions(action)
        log_info(f"📊 실제 슬라이더 값: {real_action}")
        
        # 현재 큐의 모든 항목 비우기 - 이전 상태 제거
        self._clear_state_queue()
        
        # ZMQ를 통해 Grasshopper로 액션 전송
        self._send_action(real_action)
        
        # Grasshopper가 처리할 시간을 충분히 줌
        log_info(f"⏱️ Grasshopper 처리를 위해 {self.wait_time}초 대기 중...")
        time.sleep(self.wait_time)
        
        # 최대 재시도 횟수 설정
        max_retries = 5  # 더 많은 재시도 기회
        retries = 0
        valid_state_received = False
        
        while retries < max_retries and not valid_state_received:
            # ZMQ를 통해 Grasshopper에서 상태 수신
            if self._wait_for_state():
                state = self.current_state
                reward = self.current_reward
                info = self.current_info
                
                # 상태가 유효한지 확인 (Closed Brep 여부)
                is_valid_state = True
                if 'reward_info' in info:
                    reward_info = info['reward_info']
                    if isinstance(reward_info, dict) and 'error' in reward_info and 'Not a Closed Brep' in reward_info['error']:
                        is_valid_state = False
                        log_warning(f"⚠️ 유효하지 않은 형태 (시도 {retries+1}/{max_retries}): Closed Brep이 아님")
                
                if is_valid_state:
                    valid_state_received = True
                else:
                    # 유효하지 않은 상태면 약간 다른 액션으로 재시도
                    retries += 1
                    if retries < max_retries:
                        # 재시도 전략 다양화
                        if retries == 1:
                            # 약간의 노이즈 추가
                            noise = np.random.normal(0, 0.1, size=action.shape)
                            new_action = np.clip(action + noise, -1.0, 1.0)
                        elif retries == 2:
                            # 살짝 축소
                            new_action = np.clip(action * 0.95, -1.0, 1.0)
                        elif retries == 3:
                            # 살짝 확대
                            new_action = np.clip(action * 1.05, -1.0, 1.0)
                        else:
                            # 더 큰 노이즈
                            noise = np.random.normal(0, 0.3, size=action.shape)
                            new_action = np.clip(action + noise, -1.0, 1.0)
                        
                        log_info(f"🔄 유효하지 않은 상태로 인한 재시도 {retries}/{max_retries}, 수정된 액션: {new_action}")
                        
                        # 수정된 액션으로 다시 시도
                        real_action = self._normalize_actions(new_action)
                        self._send_action(real_action)
                        time.sleep(self.wait_time)  # 처리 시간 대기
            else:
                log_warning("Grasshopper에서 상태를 받지 못했습니다. 이전 상태 사용.")
                state, reward, info = self._get_last_state()
                valid_state_received = True  # 이전 상태를 사용하므로 루프 종료
        
        # 모든 재시도 후에도 유효한 상태를 받지 못한 경우
        if not valid_state_received:
            log_warning(f"⚠️ {max_retries}번의 시도 후에도 유효한 상태를 받지 못했습니다.")
            
            # 학습에서 제외하기 위해 'truncated'를 True로 설정하고, reward는 0으로 설정
            # 'truncated'가 True이면 SB3의 PPO 알고리즘이 이 샘플을 학습에서 제외
            truncated = True
            reward = 0.0
            
            # 이전 상태 사용 (하지만 학습에서는 제외됨)
            state, _, info = self._get_last_state()
            info['excluded_from_training'] = True
            info['reason'] = "Invalid geometry (Not a Closed Brep) after max retries"
            
            # 에피소드 종료는 아님
            terminated = False
            
            # 이 상태에서는 에피소드 스텝과 총 스텝을 업데이트하지 않음
            return state, reward, terminated, truncated, info
        
        # 상태와 보상 로깅 - 표면적 체적비 포함
        bcr = state[0] * 100.0 if len(state) > 0 else 0.0
        far = state[1] * 100.0 if len(state) > 1 else 0.0

        # 현재 상태의 값들 추출
        if len(state) >= 4:
            winter_sunlight = state[2] if len(state) > 2 else 0.0
            sv_ratio = state[3] if len(state) > 3 else 0.0
            log_info(f"📊 BCR: {bcr:.1f}%, FAR: {far:.1f}%, 겨울 일사량: {winter_sunlight:.2f}, 표면적 체적비: {sv_ratio:.4f}")
        else:
            log_info(f"📊 BCR: {bcr:.1f}%, FAR: {far:.1f}%, 값 부족")
        
        # Closed Brep 상태 확인 및 표시
        is_closed_brep = True  # 기본값
        if 'reward_info' in info:
            reward_info = info['reward_info']
            if isinstance(reward_info, dict) and 'error' in reward_info and 'Not a Closed Brep' in reward_info['error']:
                is_closed_brep = False
                log_warning(f"⚠️ 유효하지 않은 형태: Closed Brep이 아님")
        
        log_info(f"💰 보상: {reward} (Closed Brep: {'Yes' if is_closed_brep else 'No'})")
        
        # 에피소드가 종료되는지 확인
        terminated = False
        truncated = self.episode_steps >= 50  # 최대 50 스텝
        
        # 정보 사전에 추가 정보 포함
        info['episode_steps'] = self.episode_steps
        info['total_steps'] = self.total_steps
        info['actual_action'] = real_action.tolist()
        info['is_closed_brep'] = is_closed_brep
        
        # 에피소드 스텝과 총 스텝 업데이트
        self.episode_steps += 1
        self.total_steps += 1
        
        return state, reward, terminated, truncated, info
    
    def close(self):
        """환경 리소스를 정리합니다"""
        log_info("🧹 환경 리소스 정리 중...")
        
        # ZMQ 소켓 정리
        if self.action_socket:
            try:
                self.action_socket.close()
                log_info("🔌 액션 ZMQ 소켓이 닫혔습니다.")
            except:
                pass
            self.action_socket = None
        
        # ZMQ 컨텍스트 정리
        if self.context:
            try:
                self.context.term()
                log_info("🔄 액션 ZMQ 컨텍스트가 종료되었습니다.")
            except:
                pass
            self.context = None
        
        super().close()

# PPO 학습 함수
def train_ppo(env, total_timesteps=10000, learning_rate=0.0003, save_dir=None, log_dir=None):
    """PPO 알고리즘을 사용하여 에이전트를 학습합니다"""
    # 저장 디렉토리 생성
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    if log_dir is None:
        log_dir = os.path.join(DATA_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 타임스탬프로 모델 저장 경로 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ppo_architecture_{timestamp}"
    model_path = os.path.join(save_dir, model_name)
    
    # 체크포인트 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_path,
        name_prefix="checkpoint"
    )
    
    # 모델 설정
    # PPO 모델 설정
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,          # 512 스텝마다 정책 업데이트 (2048에서 축소)
        batch_size=64,        # 적절한 배치 크기 유지
        n_epochs=5,           # 10에서 5로 축소 (더 빠른 학습)
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

# 모델 테스트 함수
def test_model(env, model=None, num_episodes=10):
    """학습된 모델이나 랜덤 액션으로 환경을 테스트합니다"""
    log_info(f"\n총 {num_episodes}회 테스트를 수행합니다...\n")
    
    total_rewards = []
    
    for i in range(num_episodes):
        if STOP_EVENT.is_set():
            break
        
        # 환경 초기화
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0  # 스텝 카운터 초기화
        max_steps = 5  # 각 에피소드에서 실행할 최대 스텝 수
        
        # 테스트 시작 메시지
        log_info(f"\n테스트 {i+1}/{num_episodes}")
        
        while not (done or truncated) and step_count < max_steps:
            if STOP_EVENT.is_set():
                break
            
            # 액션 선택
            if model is None:
                # 매 스텝마다 새로운 무작위 액션 생성
                action = np.random.uniform(-1.0, 1.0, size=4)
                log_info(f"🔍 무작위 액션 생성: {action}")
            else:
                # 모델의 액션 사용
                action, _ = model.predict(state)
            
            # 스텝 실행
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            print(f"에피소드 {i+1}, 스텝 {step_count}/{max_steps}")
            
            # 다음 스텝까지 작은 지연 추가 (선택 사항)
            time.sleep(0.5)
        
        total_rewards.append(episode_reward)
        log_info(f"다음 테스트까지 5초 대기 중...")
        time.sleep(5)
    
    # 테스트 결과 요약
    if total_rewards:
        mean_reward = np.mean(total_rewards)
        log_info(f"\n테스트 완료! 평균 보상: {mean_reward:.2f}")
    else:
        log_info("\n테스트가 중단되었습니다.")
    
    return total_rewards

# 메인 함수
def main():
    # 명령행 인수 처리
    parser = argparse.ArgumentParser(description="건축 설계 최적화를 위한 강화학습")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ 포트 번호 (액션 전송용)")
    parser.add_argument("--state-port", type=int, default=5557, help="ZMQ 포트 번호 (상태 수신용)")
    parser.add_argument("--steps", type=int, default=10000, help="총 학습 타임스텝 수")
    parser.add_argument("--test-only", action="store_true", help="학습 없이 환경 테스트만 수행")
    parser.add_argument("--episodes", type=int, default=10, help="테스트 에피소드 수")
    parser.add_argument("--model", type=str, help="테스트할 모델 경로")
    parser.add_argument("--debug", action="store_true", help="디버그 로그 활성화")
    parser.add_argument("--bcr-limit", type=float, default=70.0, help="BCR 제한 (백분율)")
    parser.add_argument("--far-min", type=float, default=200.0, help="최소 FAR (백분율)")
    parser.add_argument("--far-max", type=float, default=500.0, help="최대 FAR (백분율)")
    parser.add_argument("--use-seasonal-reward", action="store_true", 
                        help="계절별 일사량을 고려한 보상 함수 사용")
    parser.add_argument("--reward-type", type=str, default="optimized", choices=["original", "enhanced", "optimized"],
                    help="보상 함수 유형 (original, enhanced, optimized) (기본값: optimized)")
    args = parser.parse_args()
    
    # 강제 종료 핸들러 추가
    def force_stop_handler(sig, frame):
        print("\n강제 종료 신호를 받았습니다.")
        import os
        os._exit(1)
    
    # 원래 핸들러와 함께 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, force_stop_handler)
    
    # 디버그 모드 설정
    global DEBUG
    DEBUG = args.debug
    
    # 타임스탬프 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 헤더 출력
    print("\n" + "="*80)
    print(f"🏗️  건축 설계 최적화를 위한 강화학습 시작")
    print("="*80)
    print(f"🕒 세션 ID: {timestamp}")
    print(f"📁 로그 디렉토리: {ZMQ_LOGS_DIR}")
    print(f"💻 학습 디바이스: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"📏 건축 제한: BCR {args.bcr_limit}%, FAR {args.far_min}%~{args.far_max}%")
    print(f"🔧 보상 함수 유형: {args.reward_type}")
    print(f"🌞 계절별 보상: {'사용' if args.use_seasonal_reward else '미사용'}\n")
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    # 단계별 실행
    print("[1/5] 환경 초기화 중...")
    try:
        env = ArchitectureOptimizationEnv(
            action_port=args.port,
            state_port=args.state_port,
            bcr_limit=args.bcr_limit,
            far_min_limit=args.far_min,
            far_max_limit=args.far_max,
            use_seasonal_reward=args.use_seasonal_reward,
            reward_type=args.reward_type  # 명시적으로 reward_type 전달
        )
    except Exception as e:
        log_error(f"환경 초기화 중 오류: {e}")
        return
    
    # 이제 상태 수신기 초기화 (환경의 보상 함수 전달)
    print("[2/5] 상태 수신기 초기화 중...")
    try:
        state_receiver = StateReceiver(
            port=args.state_port, 
            save_dir=ZMQ_LOGS_DIR,
            reward_function=env.reward_function  # 환경의 보상 함수 전달
        )
        
        if not state_receiver.initialize():
            log_error("상태 수신기 초기화 실패, 프로그램을 종료합니다.")
            return
    except Exception as e:
        log_error(f"상태 수신기 초기화 중 오류: {e}")
        return
    
    # 상태 수신기 스레드 시작
    receiver_thread = threading.Thread(target=state_receiver.start)
    receiver_thread.daemon = True
    receiver_thread.start()
    
    # 테스트 모드 또는 학습 모드 확인
    if args.test_only:
        print("[테스트 모드] 환경 테스트만 수행합니다...")
    else:
        print(f"[학습 모드] {args.steps}개의 타임스텝 동안 학습합니다...")
    
    # 모델 로드 또는 생성
    model = None
    if args.model:
        try:
            print(f"[2/5] 모델 로드 중: {args.model}")
            model = PPO.load(args.model, env=env)
            print(f"✅ 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            log_error(f"모델 로드 실패: {e}")
            args.test_only = True  # 모델 로드 실패 시 테스트 모드로 전환
    
    try:
        if args.test_only:
            # 환경 테스트만 수행
            test_model(env, model, args.episodes)
        else:
            # PPO 학습 수행
            print("[3/5] PPO 학습 시작...")
            model, model_path = train_ppo(
                env,
                total_timesteps=args.steps,
                save_dir=os.path.join(DATA_DIR, "models")
            )
            
            if model and not STOP_EVENT.is_set():
                # 학습된 모델 테스트
                print("[4/5] 학습된 모델 테스트 중...")
                test_model(env, model, args.episodes)
            
            print("[5/5] 학습 데이터 분석 중...")
            # 분석 코드는 여기에 추가할 수 있습니다.
    except Exception as e:
        log_error(f"실행 중 오류 발생: {e}")
    finally:
        # 리소스 정리
        print("\n🧹 리소스 정리 중...")
        env.close()
        
        # 상태 수신기가 아직 실행 중이면 정리
        if receiver_thread.is_alive():
            STOP_EVENT.set()
            receiver_thread.join(timeout=5.0)
            state_receiver.cleanup()
        
        print("\n💯 프로그램 종료")

if __name__ == "__main__":
    main()