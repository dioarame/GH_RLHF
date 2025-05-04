# env_simple.py (ZMQ PUSH 소켓 관리 최적화 버전)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests, traceback
import zmq
import time
import json
import base64
import decimal
import threading
import atexit
from typing import Optional, Dict, Any, List, Union, Tuple

class SimpleGrasshopperEnv(gym.Env):
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
        use_push_mode: bool = True,  # PUSH 모드만 최적화
    ):
        super().__init__()

        self.compute_url = compute_url
        self.gh_definition_path = gh_definition_path
        self.state_name = state_output_param_name
        self.reward_name = reward_output_param_name
        self.slider_info_name = slider_info_param_name
        self._max_steps = max_episode_steps
        self._step_counter = 0
        self.retry_count = 3
        self.retry_delay = 0.5
        self.action_push_port = action_push_port
        self.slider_roundings = []

        # PUSH 모드만 사용 (REP 모드 제거)
        self.use_push_mode = True
        
        # 세션 설정 최적화
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_maxsize=16,
            max_retries=3,
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # 기본 Action/Observation Space 설정
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)
        self.slider_roundings = [0.01]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # ZMQ 변수 초기화
        self.zmq_context = None
        self.zmq_socket = None
        self.zmq_lock = threading.Lock()  # 스레드 안전한 소켓 접근을 위한 락
        self._send_counter = 0
        self._send_failures = 0
        self._socket_reconnect_count = 0
        self._last_message_time = 0
        self._health_check_interval = 10  # 소켓 상태 확인 간격 (초)
        
        # 프로그램 종료 시 정리 함수 등록
        atexit.register(self.close)

        # 초기화 진행
        self._initialize_gh_slider_info()
        self._init_zmq_server()

    def _parse_multiple_slider_info(self, data_raw):
        slider_infos = []
        print(f"\n[슬라이더 파싱] 원본 데이터: '{data_raw}'")
        try:
            if isinstance(data_raw, str) and len(data_raw) > 1:
                if data_raw[0] in "\"'" and data_raw[-1] == data_raw[0]:
                    data_raw = data_raw[1:-1]
                    print(f"따옴표 제거 후: '{data_raw}'")

            processed_data = data_raw.replace('\\r\\n', '\n')
            print(f"리터럴 '\\r\\n' 처리 후: '{processed_data}'")

            lines = processed_data.strip().splitlines()
            print(f"{len(lines)}개 라인 발견.")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                print(f"라인 {i+1}: '{line}'")
                parts = [part.strip() for part in line.split(',')]
                print(f"  분할 결과: {parts}")

                if len(parts) >= 3:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    rounding = float(parts[2])
                    slider_infos.append((min_val, max_val, rounding))
                    print(f"  파싱 성공: min={min_val}, max={max_val}, rounding={rounding}")
                elif len(parts) >= 2:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    slider_infos.append((min_val, max_val, None))
                    print(f"  파싱 성공 (Rounding 없음): min={min_val}, max={max_val}")
                else:
                    print(f"  라인 {i+1} 파싱 실패: 충분한 값이 없음")
        except Exception as e:
            print(f"  라인 파싱 중 오류 발생: {e}")
            traceback.print_exc()

        return slider_infos

    def _initialize_gh_slider_info(self):
        print("\n[초기화] 슬라이더 정보 가져오고 Action/Observation Space 정의 중...")
        slider_infos = []
        obs_dimension = 1
        initial_obs = None

        for attempt in range(self.retry_count):
            try:
                print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - Rhino.Compute 요청 중...")
                resp = self.session.post(self.compute_url, json=self._payload({}), timeout=30)
                resp.raise_for_status()
                response_json = resp.json()

                slider_data_raw = None
                state_data_raw = None

                for item in response_json.get("values", []):
                    param_name = item.get("ParamName")
                    inner_tree = item.get("InnerTree", {})
                    first_key = next(iter(inner_tree), None)
                    data_item_list = inner_tree.get(first_key, [])
                    raw_data = data_item_list[0].get("data") if data_item_list else None

                    if raw_data:
                        if isinstance(raw_data, str) and len(raw_data) > 1 and raw_data[0] in "\"'" and raw_data[-1] == raw_data[0]:
                            raw_data = raw_data[1:-1]

                        if param_name == self.slider_info_name:
                            slider_data_raw = raw_data
                        elif param_name == self.state_name:
                            state_data_raw = raw_data

                if slider_data_raw:
                    print(f"[초기화] '{self.slider_info_name}' 파라미터 데이터 발견.")
                    slider_infos = self._parse_multiple_slider_info(slider_data_raw)
                    if slider_infos:
                        print(f"[초기화] {len(slider_infos)}개 슬라이더 정보 파싱 성공!")
                        
                        # 라운딩 값의 정확한 소숫점 자릿수 로깅 추가
                        for i, info in enumerate(slider_infos):
                            min_val, max_val, rounding = info
                            decimal_places = self._get_decimal_places(rounding)
                            print(f"  슬라이더 {i+1}: 범위 [{min_val} ~ {max_val}], 라운딩: {rounding}, 자릿수: {decimal_places}")
                    else:
                        print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.slider_info_name}' 데이터 파싱 실패.")
                else:
                    print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.slider_info_name}' 파라미터 또는 데이터 없음.")

                if state_data_raw is not None:
                    print(f"[초기화] '{self.state_name}' 파라미터 데이터 발견: '{state_data_raw}'")
                    try:
                        parsed_state = [float(x.strip()) for x in state_data_raw.split(',') if x.strip()]
                        if parsed_state:
                            obs_dimension = len(parsed_state)
                            initial_obs = np.array(parsed_state, dtype=np.float32)
                            print(f"[초기화] 상태 파싱 성공. Observation 차원: {obs_dimension}")
                        else:
                            print(f"[초기화] 상태 데이터 파싱 실패 (빈 결과). 기본 1차원 사용.")
                            initial_obs = np.array([0.0], dtype=np.float32)
                    except ValueError:
                        print(f"[초기화] 상태 데이터 '{state_data_raw}'를 float 리스트로 변환 실패. 기본 1차원 사용.")
                        initial_obs = np.array([0.0], dtype=np.float32)
                else:
                    print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - '{self.state_name}' 파라미터 또는 데이터 없음. 기본 1차원 사용.")
                    initial_obs = np.array([0.0], dtype=np.float32)

                if slider_infos and initial_obs is not None:
                    break

            except Exception as e:
                print(f"[초기화] 시도 {attempt+1}/{self.retry_count} - 정보 가져오기/파싱 실패: {e}")
                if isinstance(e, requests.exceptions.HTTPError):
                    print(f"HTTP 상태 코드: {e.response.status_code}")
                    try: print(f"응답 내용: {e.response.text[:500]}...")
                    except: pass
                else: traceback.print_exc()

            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)

        # Action Space 설정
        if slider_infos:
            mins = np.array([info[0] for info in slider_infos], dtype=np.float32)
            maxs = np.array([info[1] for info in slider_infos], dtype=np.float32)
            self.slider_roundings = [info[2] if info[2] is not None else 0.01 for info in slider_infos]
            
            # 각 슬라이더의 소숫점 자릿수 저장
            self.slider_decimal_places = [self._get_decimal_places(rounding) for rounding in self.slider_roundings]
            
            self.action_space = spaces.Box(low=mins, high=maxs, dtype=np.float32)
            print(f"[초기화] {len(slider_infos)}차원 Action Space 설정 완료:")
            print(f"  - 최소값: {self.action_space.low}")
            print(f"  - 최대값: {self.action_space.high}")
            print(f"  - Rounding 값: {self.slider_roundings}")
            print(f"  - 소숫점 자릿수: {self.slider_decimal_places}")
        else:
            print(f"[초기화] 슬라이더 정보를 가져올 수 없어 기본 1차원 Action Space 사용.")
            print(f"  - Action Space: {self.action_space}")
            print(f"  - Rounding 값: {self.slider_roundings}")
            self.slider_decimal_places = [2]  # 기본 소숫점 자릿수

        # Observation Space 설정
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dimension,), dtype=np.float32)
        print(f"[초기화] Observation Space 설정 완료 (Shape: {self.observation_space.shape})")
        if initial_obs is not None:
            print(f"  - 감지된 초기 상태 값: {initial_obs}")
    def _get_decimal_places(self, rounding_value):
        """라운딩 값에 기반하여 소수점 자릿수를 계산하는 내부 함수"""
        try:
            if rounding_value is None or rounding_value <= 0:
                return 2  # 기본값: 소숫점 2자리
                
            if rounding_value >= 1.0:
                return 0  # 정수 슬라이더
                
            # 소수점 자릿수 계산
            decimal_places = 0
            temp_rounding = rounding_value
            
            while temp_rounding < 1.0 and decimal_places < 10:
                temp_rounding *= 10
                decimal_places += 1
                
            return decimal_places
        except Exception as e:
            print(f"라운딩 자릿수 계산 오류: {e}")
            return 2  # 오류 시 기본값

    def _init_zmq_server(self):
        """ZMQ PUSH 소켓 초기화 (최적화 버전)"""
        with self.zmq_lock:
            if self.zmq_socket:
                print("ZMQ 서버 소켓 이미 초기화됨.")
                return

            try:
                self.zmq_context = zmq.Context()
                
                # PUSH 소켓 생성
                self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
                bind_address = f"tcp://127.0.0.1:{self.action_push_port}"
                
                # 소켓 옵션 최적화
                self.zmq_socket.setsockopt(zmq.SNDHWM, 1000)  # 낮춘 값으로 수정
                self.zmq_socket.setsockopt(zmq.LINGER, 200)   # 짧은 linger 시간
                self.zmq_socket.setsockopt(zmq.BACKLOG, 100)  # 백로그 제한
                
                # TCP Keepalive 설정
                self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
                self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
                self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
                
                print(f"바인딩 시도 (PUSH): {bind_address}")
                self.zmq_socket.bind(bind_address)
                print(f"✅ ZMQ PUSH 서버가 {bind_address}에 바인딩되었습니다.")
                
                # 초기 테스트 메시지 전송 시도
                test_values = [-9.99] * max(1, len(self.slider_roundings))
                test_data = json.dumps(test_values)
                try:
                    self.zmq_socket.send_string(test_data, flags=zmq.NOBLOCK)
                    print(f"✅ 초기 테스트 메시지 전송됨: {test_data}")
                    self._send_counter = 1
                    self._last_message_time = time.time()
                except zmq.Again:
                    print(f"🟡 초기 테스트 메시지 전송 실패: 수신자(Grasshopper) 준비 안됨")
                    self._send_counter = 0
                    
                # 1초 간격으로 소켓 상태 확인 스레드 시작
                self._start_health_check_thread()
                    
            except Exception as e:
                print(f"❌ ZMQ 서버 초기화 중 오류 발생: {e}")
                traceback.print_exc()
                self._cleanup_zmq_resources()
    
    def _start_health_check_thread(self):
        """소켓 상태 확인을 위한 별도 스레드 시작"""
        self.health_check_running = True
        self.health_check_thread = threading.Thread(target=self._socket_health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        print("✅ 소켓 상태 확인 스레드 시작됨")
    
    def _socket_health_check_loop(self):
        """소켓 상태 확인 루프"""
        while getattr(self, 'health_check_running', True):
            try:
                # 마지막 메시지 전송 후 일정 시간이 지났고, 소켓이 존재하면 keepalive 메시지 전송
                current_time = time.time()
                time_since_last_msg = current_time - self._last_message_time
                
                if time_since_last_msg > self._health_check_interval and self.zmq_socket:
                    with self.zmq_lock:
                        if self.zmq_socket and not self.zmq_socket.closed:
                            try:
                                # Keepalive 메시지 전송 (실제 동작에 영향 없는 값)
                                keepalive_msg = json.dumps([-9.87])
                                self.zmq_socket.send_string(keepalive_msg, flags=zmq.NOBLOCK)
                                print(f"💓 Keepalive 메시지 전송됨 (마지막 메시지 후 {time_since_last_msg:.1f}초)")
                                self._last_message_time = current_time
                            except zmq.Again:
                                print("🟡 Keepalive 메시지 전송 실패: 수신자 준비 안됨")
                            except Exception as e:
                                print(f"❌ Keepalive 메시지 전송 중 오류: {e}")
                                # 소켓에 문제가 있으면 재초기화
                                self._reinit_zmq_server()
                
            except Exception as e:
                print(f"❌ 상태 확인 스레드 오류: {e}")
            
            # 다음 확인 전 대기
            time.sleep(1)
    
    def _cleanup_zmq_resources(self):
        """ZMQ 리소스 정리 (락 없이 호출하지 말 것)"""
        if self.zmq_socket: 
            try:
                if not self.zmq_socket.closed:
                    self.zmq_socket.close(linger=0)
                print("ZMQ 소켓 닫힘")
            except Exception as e:
                print(f"ZMQ 소켓 닫기 오류: {e}")
            self.zmq_socket = None
            
        if self.zmq_context:
            try:
                if not self.zmq_context.closed:
                    self.zmq_context.term()
                print("ZMQ 컨텍스트 종료됨")
            except Exception as e:
                print(f"ZMQ 컨텍스트 종료 오류: {e}")
            self.zmq_context = None

    def _send_action_to_grasshopper(self, action_values):
        """액션 값을 Grasshopper로 전송 (라운딩 정밀도 수정 버전)"""
        # 액션이 스칼라인 경우 리스트로 변환
        if not isinstance(action_values, (list, np.ndarray)):
            action_values = [action_values]
        
        # Clipping 및 라운딩 처리
        clipped_action = np.clip(action_values, self.action_space.low, self.action_space.high)
        rounded_values = []
        num_sliders = len(self.slider_roundings)
        
        for i, val in enumerate(clipped_action):
            if i < num_sliders:
                rounding = self.slider_roundings[i]
                if rounding is not None and rounding > 0:
                    # 라운딩 값에 따른 정확한 자릿수 처리
                    if rounding == 1.0:  # 정수 슬라이더
                        rounded_val = int(round(float(val)))
                    else:
                        # 소수점 자릿수 계산
                        decimal_places = 0
                        temp_rounding = rounding
                        while temp_rounding < 1.0 and decimal_places < 10:
                            temp_rounding *= 10
                            decimal_places += 1
                        
                        # 정확한 자릿수로 반올림
                        rounded_val = round(float(val), decimal_places)
                else:
                    rounded_val = float(val)
                
                # 최종 값을 슬라이더 범위 내로 제한
                rounded_val = np.clip(rounded_val, self.action_space.low[i], self.action_space.high[i])
                rounded_values.append(rounded_val)
            else:
                break
        
        if not rounded_values:
            return False
        
        # 디버그 출력 추가
        if self._send_counter % 100 == 0:
            print(f"원본 값: {clipped_action}")
            print(f"라운딩 후: {rounded_values}")
            print(f"슬라이더 라운딩: {self.slider_roundings[:len(rounded_values)]}")
        
        # ZMQ 소켓을 통한 전송 (스레드 안전하게)
        with self.zmq_lock:
            if self.zmq_socket is None or self.zmq_socket.closed:
                print("❌ ZMQ PUSH 서버 소켓이 초기화되지 않았거나 닫혔습니다. 재초기화 시도...")
                self._reinit_zmq_server()
                return False
            
            try:
                # 전송 관련 변수 관리
                if not hasattr(self, '_send_counter'): self._send_counter = 0
                if not hasattr(self, '_send_failures'): self._send_failures = 0
                
                # 전송 시도
                data = json.dumps(rounded_values)
                self._send_counter += 1
                
                try:
                    self.zmq_socket.send_string(data, flags=zmq.NOBLOCK)
                    self._last_message_time = time.time()  # 마지막 메시지 시간 갱신
                    self._send_failures = 0  # 성공 시 실패 카운터 리셋
                    
                    # 로그 간소화: 100개마다 출력
                    if self._send_counter % 100 == 0:
                        print(f"\r📤 ZMQ 전송 #{self._send_counter}: {data}", end="")
                    if self._send_counter % 1000 == 0:
                        print()
                    return True
                except zmq.Again:
                    self._send_failures += 1
                    if self._send_failures % 50 == 0:  # 로그 간소화
                        print(f"\n🟡 ZMQ 전송 실패 #{self._send_failures}: 수신자 준비 안됨 (메시지 #{self._send_counter})")
                    
                    # 연속 실패 시 소켓 재초기화
                    if self._send_failures >= 200:  # 재시도 횟수 증가
                        print(f"\n⚠️ 너무 많은 연속 실패 ({self._send_failures}). 소켓 재초기화 시도")
                        self._reinit_zmq_server()
                    
                    return False
            except Exception as e:
                print(f"❌ 액션 전송 중 오류: {e}")
                traceback.print_exc()
                # 오류 발생 시 소켓 재초기화
                self._reinit_zmq_server()
                return False

    def _reinit_zmq_server(self):
        """ZMQ 서버 재초기화 (개선된 버전)"""
        with self.zmq_lock:
            print("\n🔄 ZMQ 서버 소켓 재초기화 중...")
            self._socket_reconnect_count += 1
            
            # 기존 리소스 정리
            self._cleanup_zmq_resources()
            
            # 재초기화 전 잠시 대기
            time.sleep(1.0)  # 대기 시간 증가
            
            # 소켓 재초기화
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
            bind_address = f"tcp://127.0.0.1:{self.action_push_port}"
            
            # 소켓 옵션 최적화
            self.zmq_socket.setsockopt(zmq.SNDHWM, 1000)
            self.zmq_socket.setsockopt(zmq.LINGER, 200)
            self.zmq_socket.setsockopt(zmq.BACKLOG, 100)
            self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            self.zmq_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
            
            try:
                # 소켓 바인딩
                self.zmq_socket.bind(bind_address)
                print(f"✅ ZMQ 서버 소켓 재초기화 완료 (#{self._socket_reconnect_count})")
                self._send_failures = 0
                
                # 테스트 메시지 전송
                test_values = [-9.88] * max(1, len(self.slider_roundings))
                test_data = json.dumps(test_values)
                try:
                    self.zmq_socket.send_string(test_data, flags=zmq.NOBLOCK)
                    print(f"✅ 재초기화 후 테스트 메시지 전송 성공")
                    self._last_message_time = time.time()
                except zmq.Again:
                    print(f"🟡 재초기화 후 테스트 메시지 전송 실패: 수신자 준비 안됨")
            except Exception as e:
                print(f"❌ 소켓 재초기화 중 오류: {e}")
                self.zmq_socket = None
                self.zmq_context = None

    def _payload(self, inputs: Dict[str, Any]) -> dict:
        """Compute 요청 페이로드 생성"""
        try:
            with open(self.gh_definition_path, 'rb') as f:
                gh_definition_bytes = f.read()
            gh_definition_b64 = base64.b64encode(gh_definition_bytes).decode('utf-8')
        except Exception as e:
            print(f"❌ 페이로드 생성 중 GH 파일 읽기 오류: {e}")
            return {"algo": None, "values": []}

        values_list = []
        for name, value in inputs.items():
            inner_tree_data = [{"data": value}]
            values_list.append({"ParamName": name, "InnerTree": {"{ 0; }": inner_tree_data}})

        return {
            "algo": gh_definition_b64,
            "pointer": None,
            "values": values_list
        }

    def _call_compute(self, inputs: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Rhino.Compute 서버 호출 (최적화 버전)"""
        if not self.session:
            print("❌ HTTP session is not initialized.")
            return None

        payload = self._payload(inputs)
        if payload.get("algo") is None:
            return None

        try:
            response = self.session.post(self.compute_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"❌ Rhino.Compute request timed out after {timeout} seconds.")
            return None
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ Rhino.Compute HTTP Error: {http_err.response.status_code}")
            try:
                print(f"   Response: {http_err.response.text[:500]}...")
                error_json = http_err.response.json()
                if "errors" in error_json: print(f"   Compute Errors: {error_json['errors']}")
                if "warnings" in error_json: print(f"   Compute Warnings: {error_json['warnings']}")
            except: pass
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Rhino.Compute request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode JSON response: {e}")
            return None

    def _parse(self, js: dict):
        """Compute 결과 파싱"""
        state_raw, reward_raw = None, None
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = -1e3

        if js is None: return state, reward

        for item in js.get("values", []):
            pname = item.get("ParamName")
            inner_tree = item.get("InnerTree", {})
            first_key = next(iter(inner_tree), None)
            data_raw = inner_tree.get(first_key, [{}])[0].get("data") if first_key else None
            if data_raw is None: continue

            if isinstance(data_raw, str) and len(data_raw) > 1:
                if data_raw[0] in "\"'" and data_raw[-1] == data_raw[0]:
                    data_raw = data_raw[1:-1]

            if pname == self.state_name:
                state_raw = data_raw
            elif pname == self.reward_name:
                reward_raw = data_raw

        # 상태 값 파싱
        if state_raw is not None:
            try:
                parsed_state = [float(x.strip()) for x in state_raw.split(',') if x.strip()]
                if len(parsed_state) == self.observation_space.shape[0]:
                    state = np.array(parsed_state, dtype=np.float32)
                elif parsed_state:
                    print(f"[Parse Warning] State 차원 불일치 ({len(parsed_state)} vs {self.observation_space.shape[0]}). 0벡터 유지.")
            except ValueError:
                print(f"[Parse Warning] State '{state_raw}' 파싱 실패. 0벡터 유지.")

        # 보상 값 파싱
        if reward_raw is not None:
            try:
                reward = float(reward_raw)
            except ValueError:
                print(f"[Parse Warning] Reward '{reward_raw}' 파싱 실패. -1e3 유지.")

        return state, reward

    def reset(self, *, seed=None, options=None):
        """환경 초기화 (개선된 버전)"""
        super().reset(seed=seed)
        self._step_counter = 0

        # ZMQ 서버 소켓 확인 및 필요시 재초기화
        with self.zmq_lock:
            if self.zmq_socket is None or self.zmq_socket.closed:
                print("[Reset] ZMQ 서버 소켓 준비되지 않음. 재초기화 시도...")
                self._reinit_zmq_server()
                if self.zmq_socket is None:
                    return np.zeros(self.observation_space.shape, dtype=np.float32), {"error": "ZMQ initialization failed"}

        print("[Reset] 환경 초기화 중...")
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}

        # 초기 액션 샘플링 및 전송
        init_action = self.action_space.sample()
        print(f"[Reset] 초기 액션 샘플링: {init_action}")
        self._send_action_to_grasshopper(init_action)
        
        # Grasshopper가 액션을 처리하고 상태를 계산할 시간 부여
        time.sleep(0.3)  # 약간 증가된 대기 시간

        # 초기 상태 가져오기
        for attempt in range(self.retry_count):
            print(f"[Reset] 시도 {attempt+1}/{self.retry_count} - 초기 상태 가져오기...")
            compute_result = self._call_compute({})

            if compute_result is not None:
                state, _ = self._parse(compute_result)
                print(f"[Reset] 성공. 초기 상태: {state}")
                return state, info
            else:
                print(f"[Reset Error] 시도 {attempt+1}/{self.retry_count} - Compute 호출 실패.")
                if attempt < self.retry_count - 1:
                    print(f"[Reset] 재시도 전 액션 재전송: {init_action}")
                    self._send_action_to_grasshopper(init_action)
                    time.sleep(self.retry_delay + 0.2)  # 약간 증가된 대기 시간

        print("[Reset Error] 최대 시도 횟수 초과. 기본 상태 반환.")
        info["error"] = "Reset failed after retries"
        return state, info

    def step(self, action):
        """환경 진행 (개선된 버전)"""
        # 액션 전송
        send_success = self._send_action_to_grasshopper(action)
        if not send_success:
            print("[Step Warning] 액션 전송 실패. 계속 진행합니다.")

        # Grasshopper가 액션 처리 후 상태/보상 계산할 시간 부여
        time.sleep(0.15)  # 약간 증가된 대기 시간

        # 기본값 설정
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = -2e3
        terminated = False
        truncated = False
        info = {}

        # 상태 및 보상 가져오기
        compute_result = self._call_compute({})

        if compute_result is not None:
            state, reward = self._parse(compute_result)
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
        """환경 리소스 정리 (개선된 버전)"""
        print("Closing environment...")
        
        # 상태 확인 스레드 중지
        if hasattr(self, 'health_check_running'):
            self.health_check_running = False
            if hasattr(self, 'health_check_thread') and self.health_check_thread and self.health_check_thread.is_alive():
                try:
                    self.health_check_thread.join(2.0)
                    print("Health check thread stopped.")
                except Exception as e:
                    print(f"Error stopping health check thread: {e}")
        
        # HTTP 세션 정리
        if hasattr(self, 'session') and self.session:
            try: self.session.close()
            except Exception as e: print(f"Error closing requests session: {e}")
            self.session = None
        
        # ZMQ 리소스 정리
        with self.zmq_lock:
            self._cleanup_zmq_resources()
        
        print("Environment closed.")


# Decimal 자릿수 계산 헬퍼 함수 (클래스 외부)
def get_decimal_places(rounding_value):
    """Rounding 값에 기반하여 소수점 자릿수를 반환합니다."""
    try:
        if rounding_value is None or rounding_value <= 0:
            return 10  # 기본값
        d = decimal.Decimal(str(rounding_value))
        exponent = d.as_tuple().exponent
        if isinstance(exponent, int) and exponent < 0:
            return abs(exponent)
        else:
            return 0  # 정수 단위 Rounding
    except Exception as e:
        print(f"Error getting decimal places for {rounding_value}: {e}")
        return 10  # 예외 발생 시 기본값