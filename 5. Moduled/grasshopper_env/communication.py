# grasshopper_env/communication.py
import zmq
import json
import threading
import time
import traceback
import numpy as np
from typing import List, Optional, Union

class ZMQCommunicator:
    """ZMQ 통신을 위한 클래스 (PUSH 및 REP 모드 지원)"""
    
    def __init__(self, port: int, use_push_mode: bool = True):
        """
        ZMQ 통신 초기화
        
        Args:
            port: ZMQ 바인딩 포트
            use_push_mode: True=PUSH 모드, False=REP 모드
        """
        self.port = port
        self.use_push_mode = use_push_mode
        self.zmq_context = None
        self.zmq_socket = None
        self.zmq_running = False
        self.zmq_thread = None
        self._send_counter = 0
        self._send_failures = 0
        
        # REP 모드용 변수
        self.latest_action = None
        self.action_ready = False
        self.action_lock = threading.Lock()
        
        # 초기화
        self._init_zmq_server()
        
    def _init_zmq_server(self):
        """ZMQ 서버 소켓 초기화"""
        if self.zmq_socket:  # 이미 초기화되었으면 반환
            print("ZMQ 서버 소켓 이미 초기화됨.")
            return
            
        try:
            self.zmq_context = zmq.Context()
            
            if self.use_push_mode:
                # PUSH 모드
                self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
                
                # 안정성을 위한 소켓 옵션 설정
                self.zmq_socket.set(zmq.SNDHWM, 10000)     # 높은 HWM
                self.zmq_socket.set(zmq.LINGER, 5000)      # 닫을 때 5초 기다림
                self.zmq_socket.set(zmq.TCP_KEEPALIVE, 1)  # TCP keepalive 활성화
                self.zmq_socket.set(zmq.TCP_KEEPALIVE_IDLE, 120)
                self.zmq_socket.set(zmq.TCP_KEEPALIVE_INTVL, 60)
                
                bind_address = f"tcp://127.0.0.1:{self.port}"  # 명시적 IP 사용
                print(f"바인딩 시도: {bind_address}")
                self.zmq_socket.bind(bind_address)
                print(f"✅ ZMQ PUSH 서버가 {bind_address}에 바인딩되었습니다.")
                
                # 초기 테스트 메시지 전송 시도
                try:
                    test_data = json.dumps([-9.99, -9.99, -9.99])  # 구분하기 쉬운 테스트 값
                    self.zmq_socket.send_string(test_data)  # 블로킹 모드 사용
                    print(f"✅ 초기 테스트 메시지 전송됨: {test_data}")
                    self._send_counter = 1  # 메시지 카운터 초기화
                except Exception as e:
                    print(f"🟡 초기 메시지 전송 중 예상치 못한 오류: {e}")
                    self._send_counter = 0
            else:
                # REP 모드
                self.zmq_socket = self.zmq_context.socket(zmq.REP)
                
                # 소켓 옵션 설정
                self.zmq_socket.set(zmq.LINGER, 5000)
                self.zmq_socket.set(zmq.TCP_KEEPALIVE, 1)
                
                bind_address = f"tcp://127.0.0.1:{self.port}"
                print(f"REP 모드 바인딩 시도: {bind_address}")
                self.zmq_socket.bind(bind_address)
                print(f"✅ ZMQ REP 서버가 {bind_address}에 바인딩되었습니다.")
                
                # REP 응답 스레드 시작
                self.zmq_running = True
                self.zmq_thread = threading.Thread(target=self._zmq_response_thread)
                self.zmq_thread.daemon = True
                self.zmq_thread.start()
                print("REP 응답 스레드 시작됨")
                
            # 연결 안정화를 위한 대기 시간
            print("바인딩 후 안정화를 위해 1초 대기 중...")
            time.sleep(1)
                
        except Exception as e:
            print(f"❌ ZMQ 서버 초기화 중 오류 발생: {e}")
            traceback.print_exc()
            if self.zmq_socket: self.zmq_socket.close()
            if self.zmq_context: self.zmq_context.term()
            self.zmq_socket = None
            self.zmq_context = None
    
    def _zmq_response_thread(self):
        """REP 모드에서 요청을 받고 액션 값을 응답으로 보내는 스레드"""
        print("ZMQ REP 응답 스레드 시작됨")
        while self.zmq_running:
            try:
                if self.zmq_socket.poll(100, zmq.POLLIN):  # 100ms 대기
                    # 요청 메시지 수신
                    request = self.zmq_socket.recv_string()
                    print(f"📥 REP 요청 수신: '{request}'")
                    
                    # 요청 처리 및 응답 준비
                    if request.upper() == "CONNECT":
                        # 연결 요청 - 확인 응답
                        self.zmq_socket.send_string("CONNECTED")
                        print("✅ 연결 확인 응답 전송")
                    elif request.upper() == "GET_ACTION":
                        # 액션 요청 - 최신 액션 값 전송
                        with self.action_lock:
                            if self.action_ready and self.latest_action is not None:
                                action_json = json.dumps(self.latest_action)
                                self.zmq_socket.send_string(action_json)
                                print(f"📤 액션 응답 전송: {action_json}")
                                self.action_ready = False  # 전송 완료 표시
                            else:
                                # 새 액션이 없음 - 대기 요청
                                self.zmq_socket.send_string("WAIT")
                                print("⏳ 새 액션 없음, WAIT 응답 전송")
                    else:
                        # 알 수 없는 요청
                        self.zmq_socket.send_string("UNKNOWN")
                        print(f"❓ 알 수 없는 요청: {request}, UNKNOWN 응답 전송")
                
            except zmq.ZMQError as e:
                if self.zmq_running:  # 종료 중이 아닌 경우만 오류 출력
                    print(f"❌ ZMQ REP 스레드 오류: {e}")
                time.sleep(0.1)  # 오류 발생 시 잠시 대기
            except Exception as e:
                if self.zmq_running:
                    print(f"❌ ZMQ REP 스레드 예외: {e}")
                    traceback.print_exc()
                time.sleep(0.1)
        
        print("ZMQ REP 응답 스레드 종료됨")
    
    def send_action(self, action_values: Union[List[float], np.ndarray], 
                   action_space_low: np.ndarray, 
                   action_space_high: np.ndarray,
                   roundings: List[float]) -> bool:
        """
        액션 값을 ZMQ를 통해 전송
        
        Args:
            action_values: 전송할 액션 값 리스트/배열
            action_space_low: 액션 공간 최소값
            action_space_high: 액션 공간 최대값
            roundings: 각 액션 값의 rounding 크기
            
        Returns:
            bool: 전송 성공 여부
        """
        if self.zmq_socket is None or self.zmq_socket.closed:
            print("❌ ZMQ 서버 소켓이 초기화되지 않았거나 닫혔습니다. 액션을 전송할 수 없습니다.")
            return False

        try:
            # action_values가 스칼라 값일 경우 리스트로 변환
            if not isinstance(action_values, (list, np.ndarray)):
                action_values = [action_values]

            # NumPy 배열을 리스트로 변환
            if isinstance(action_values, np.ndarray):
                action_values = action_values.tolist()

            # 액션 값 clipping 및 라운딩
            rounded_values = []
            
            for i, val in enumerate(action_values):
                if i < len(roundings):
                    rounding = roundings[i]
                    if rounding is not None and rounding > 0:
                        # 라운딩 적용
                        rounded_val = round(float(val) / rounding) * rounding
                    else:
                        rounded_val = float(val)
                        
                    # 범위 제한
                    if i < len(action_space_low) and i < len(action_space_high):
                        rounded_val = max(action_space_low[i], min(action_space_high[i], rounded_val))
                    
                    rounded_values.append(rounded_val)
                else:
                    break  # 슬라이더 개수보다 많은 값은 무시

            if not rounded_values:
                return False  # 전송할 값이 없음

            data = json.dumps(rounded_values)
            
            # REP 모드인 경우, 응답 스레드에서 사용할 수 있도록 저장
            if not self.use_push_mode:
                with self.action_lock:
                    self.latest_action = rounded_values
                    self.action_ready = True
                    if self._send_counter % 10 == 0:  # 10개마다 출력
                        print(f"\r📝 ZMQ REP 모드: 액션 #{self._send_counter} 준비됨: {data}", end="")
                    if self._send_counter % 100 == 0:
                        print()  # 새 줄
                    self._send_counter += 1
                return True  # REP 모드에서는 액션 준비가 끝나면 성공으로 처리
            
            # PUSH 모드인 경우 직접 전송
            try:
                self.zmq_socket.send_string(data)  # 블로킹 모드로 전송
                if self._send_counter % 10 == 0:  # 10개마다 출력
                    print(f"\r📤 ZMQ 전송 #{self._send_counter}: {data}", end="")
                if self._send_counter % 100 == 0:
                    print()  # 새 줄
                self._send_counter += 1
                return True  # 전송 성공
                
            except zmq.Again:
                self._send_failures += 1
                if self._send_failures % 5 == 0:  # 실패 메시지 빈도 감소
                    print(f"\n🟡 ZMQ 전송 실패 #{self._send_failures}: 수신자(Grasshopper) 준비 안됨")
                
                if self._send_failures >= 20:  # 재초기화 임계값 
                    print(f"\n⚠️ 너무 많은 연속 실패 ({self._send_failures}). ZMQ 소켓 재초기화 시도")
                    self._reinit_zmq_server()
                    self._send_failures = 0  # 실패 카운터 리셋
                    
                return False  # 전송 실패
        
        except zmq.ZMQError as ze:
            print(f"❌ ZMQ 오류 발생: {ze}")
            return False
        
        except Exception as e:
            print(f"❌ 액션 전송 중 오류 발생: {e}")
            traceback.print_exc()
            return False
    
    def _reinit_zmq_server(self):
        """ZMQ 서버 소켓을 닫고 재초기화합니다."""
        print("\n🔄 ZMQ 서버 소켓 재초기화 중...")
        try:
            # REP 모드인 경우 스레드 종료
            if not self.use_push_mode and self.zmq_running:
                self.zmq_running = False
                if self.zmq_thread and self.zmq_thread.is_alive():
                    try:
                        self.zmq_thread.join(timeout=2.0)  # 최대 2초 대기
                        print("  REP 응답 스레드 종료됨")
                    except:
                        print("  REP 응답 스레드 종료 실패")
                self.zmq_thread = None
            
            # 기존 소켓/컨텍스트 정리
            if self.zmq_socket:
                try:
                    self.zmq_socket.close(linger=1000)  # 1초 기다림
                    print("  이전 소켓 닫힘")
                except:
                    print("  이전 소켓 닫기 실패")
                    
            if self.zmq_context:
                try:
                    self.zmq_context.term()
                    print("  이전 컨텍스트 종료됨")
                except:
                    print("  이전 컨텍스트 종료 실패")
                    
            self.zmq_socket = None
            self.zmq_context = None
            
            # 재초기화를 위한 대기
            print("  재초기화 전 2초 대기 중...")
            time.sleep(2)
            
            # 소켓 재초기화
            self._init_zmq_server()
            
            # 카운터 리셋
            self._send_failures = 0
            print("✅ ZMQ 서버 소켓 재초기화 완료")
            
        except Exception as e:
            print(f"❌ ZMQ 서버 재초기화 중 오류 발생: {e}")
            traceback.print_exc()
    
    def close(self):
        """ZMQ 리소스 정리"""
        print("Closing ZMQ communicator...")
        
        # REP 모드 스레드 종료
        if not self.use_push_mode and self.zmq_running:
            self.zmq_running = False
            if self.zmq_thread and self.zmq_thread.is_alive():
                try:
                    self.zmq_thread.join(timeout=2.0)
                    print("REP 응답 스레드 종료됨.")
                except:
                    print("REP 응답 스레드 종료 실패.")
        
        # ZMQ 소켓 닫기
        if hasattr(self, 'zmq_socket') and self.zmq_socket and not self.zmq_socket.closed:
            try:
                print("Closing ZMQ socket (waiting up to 5s)...")
                self.zmq_socket.close(linger=5000)  # 최대 5초 대기
                print("ZMQ socket closed.")
            except Exception as e: 
                print(f"Error closing ZMQ socket: {e}")
        
        # ZMQ 컨텍스트 종료
        if hasattr(self, 'zmq_context') and self.zmq_context and not self.zmq_context.closed:
            try:
                print("Terminating ZMQ context...")
                self.zmq_context.term()
                print("ZMQ context terminated.")
            except Exception as e: 
                print(f"Error terminating ZMQ context: {e}")
        
        self.zmq_socket = None
        self.zmq_context = None
        print("ZMQ resources closed.")