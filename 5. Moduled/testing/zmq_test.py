# testing/zmq_test.py
import zmq
import json
import time
import traceback
from typing import List, Optional

def test_zmq_push_connection(port: int, test_actions: Optional[List[List[float]]] = None) -> bool:
    """
    ZMQ PUSH 소켓 연결 및 메시지 전송 테스트
    
    Args:
        port: ZMQ 포트
        test_actions: 테스트할 액션 리스트
        
    Returns:
        bool: 테스트 성공 여부
    """
    if test_actions is None:
        test_actions = [[0.0, 0.0, 0.0]]
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    
    # 향상된 소켓 옵션
    socket.setsockopt(zmq.SNDHWM, 10000)
    socket.setsockopt(zmq.LINGER, 5000)
    
    connect_address = f"tcp://127.0.0.1:{port}"
    result = False
    
    try:
        print(f"⚠️ ZMQ 연결 테스트 (PUSH): {connect_address}에 연결 시도 중...")
        socket.connect(connect_address)
        
        # 연결 안정화를 위한 대기
        print("연결 후 2초 대기 중...")
        time.sleep(2)
        
        for i, action in enumerate(test_actions):
            data = json.dumps(action)
            print(f"📤 테스트 {i+1}/{len(test_actions)}: 값 {action} 전송 중...")
            socket.send_string(data)  # 블로킹 모드 사용
            print(f"   전송된 JSON: {data}")
            time.sleep(2)  # 메시지 사이 더 긴 대기
            
        print("✅ ZMQ 테스트 메시지 전송 완료! Grasshopper에서 슬라이더가 움직이는지 확인하세요.")
        result = True
        
    except Exception as e:
        print(f"❌ ZMQ 연결 테스트 (PUSH) 실패: {e}")
        traceback.print_exc()
        result = False
        
    finally:
        try:
            print("소켓 정리 중 (5초 대기)...")
            socket.close(linger=5000)  # 5초 대기 후 닫기
            context.term()
            print("ZMQ 리소스 정리 완료")
        except Exception as e_close:
            print(f"Error closing ZMQ resources in test: {e_close}")
            
    return result

def test_zmq_req_connection(port: int, test_actions: Optional[List[List[float]]] = None) -> bool:
    """
    ZMQ REQ 소켓 연결 및 요청-응답 테스트
    
    Args:
        port: ZMQ 포트
        test_actions: 테스트할 액션 리스트
        
    Returns:
        bool: 테스트 성공 여부
    """
    if test_actions is None:
        test_actions = [[0.0, 0.0, 0.0]]
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    # 향상된 소켓 옵션
    socket.setsockopt(zmq.LINGER, 5000)
    
    connect_address = f"tcp://127.0.0.1:{port}"
    result = False
    
    try:
        print(f"⚠️ ZMQ 연결 테스트 (REQ): {connect_address}에 연결 시도 중...")
        socket.connect(connect_address)
        
        # 연결 안정화를 위한 대기
        print("연결 후 2초 대기 중...")
        time.sleep(2)
        
        # 연결 확인 요청
        print("📤 연결 확인 요청 전송 중...")
        socket.send_string("CONNECT")
        
        try:
            if socket.poll(5000):  # 5초 대기
                response = socket.recv_string()
                print(f"📥 연결 응답 수신: {response}")
                result = True
                
                # 액션 요청 테스트
                for i, action in enumerate(test_actions):
                    print(f"📤 액션 요청 {i+1}/{len(test_actions)} 전송 중...")
                    socket.send_string("GET_ACTION")
                    
                    if socket.poll(5000):  # 5초 대기
                        response = socket.recv_string()
                        print(f"📥 액션 응답 수신: {response}")
                    else:
                        print("❌ 액션 응답 대기 타임아웃")
                        
                    time.sleep(2)  # 메시지 사이 더 긴 대기
            else:
                print("❌ 연결 응답 대기 타임아웃")
                result = False
        except Exception as recv_e:
            print(f"❌ 응답 수신 오류: {recv_e}")
            result = False
            
    except Exception as e:
        print(f"❌ ZMQ 연결 테스트 (REQ) 실패: {e}")
        traceback.print_exc()
        result = False
        
    finally:
        try:
            print("소켓 정리 중 (5초 대기)...")
            socket.close(linger=5000)
            context.term()
            print("ZMQ 리소스 정리 완료")
        except Exception as e_close:
            print(f"Error closing ZMQ resources in test: {e_close}")
            
    return result

def check_compute_server(url: str) -> bool:
    """
    Rhino.Compute 서버 상태 확인
    
    Args:
        url: Rhino.Compute 서버 URL
        
    Returns:
        bool: 서버 연결 가능 여부
    """
    try:
        import requests
        base_url = url.split('/grasshopper')[0]
        r = requests.get(f"{base_url}/version", timeout=5)
        r.raise_for_status()
        print(f"✅ Rhino.Compute 서버가 작동 중입니다. 버전: {r.json()}")
        return True
    except Exception as e:
        print(f"❌ Rhino.Compute 서버 연결 실패: {e}")
        return False