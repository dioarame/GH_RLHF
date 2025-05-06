# zmq_state_receiver_fixed.py
import zmq
import json
import time
import sys
import os
import atexit
from datetime import datetime

# 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "data", "zmq_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"state_reward_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
port = 5557  # 새 포트 사용
max_idle_time = 10  # 이 시간(초) 동안 상태/보상 데이터가 없으면 자동 종료

# ZMQ 설정
context = None
socket = None

def cleanup_resources():
    global socket, context
    print("리소스 정리 중...")
    if socket:
        try:
            socket.close(linger=0)
            print("소켓 닫힘")
        except:
            pass
        socket = None
    
    if context:
        try:
            context.term()
            print("컨텍스트 종료됨")
        except:
            pass
        context = None

# 종료 시 정리 함수 등록
atexit.register(cleanup_resources)

# 소켓 초기화
try:
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)  # 명시적 linger 설정
    bind_address = f"tcp://localhost:{port}"
    
    print(f"ZMQ PULL 소켓 바인딩 중: {bind_address}")
    socket.bind(bind_address)
    print(f"수신 대기 중... 데이터는 {log_file}에 저장됩니다")
    print(f"종료하려면 Ctrl+C를 누르거나, {max_idle_time}초 동안 상태/보상 데이터가 없으면 자동 종료됩니다")
except Exception as e:
    print(f"소켓 초기화 오류: {e}")
    cleanup_resources()
    sys.exit(1)

# 수신 및 로깅
received_count = 0
real_data_count = 0  # health_check를 제외한 실제 데이터 개수
start_time = time.time()
last_real_data_time = time.time()  # 마지막으로 실제 데이터를 받은 시간

try:
    with open(log_file, 'w') as f:
        f.write("[\n")  # JSON 배열 시작
        
        while True:
            # 자동 종료 확인
            if time.time() - last_real_data_time > max_idle_time:
                print(f"\n{max_idle_time}초 동안 상태/보상 데이터가 없어 자동 종료합니다")
                break
            
            try:
                # 메시지 수신 (타임아웃 설정)
                try:
                    message = socket.recv_string(flags=zmq.NOBLOCK)
                    received_count += 1
                    
                    # JSON 파싱
                    data = json.loads(message)
                    
                    # health_check 메시지 확인
                    is_health_check = data.get("type") == "health_check"
                    
                    if not is_health_check:
                        real_data_count += 1
                        last_real_data_time = time.time()  # 실제 데이터 수신 시간 업데이트
                    
                    # 콘솔에 출력
                    if received_count % 10 == 0:  # 10개마다 출력
                        elapsed = time.time() - start_time
                        rate = received_count / elapsed if elapsed > 0 else 0
                        print(f"수신: {received_count}개 (실제 데이터: {real_data_count}개, {rate:.1f}개/초)")
                        
                        # 데이터 샘플 출력 (health_check가 아닌 경우에만)
                        if not is_health_check:
                            print(f"샘플: {message[:100]}..." if len(message) > 100 else f"샘플: {message}")
                    
                    # 파일에 저장 (health_check 제외)
                    if not is_health_check:
                        f.write(message)
                        f.write(",\n")  # 레코드 구분
                    
                    # 주기적으로 파일 버퍼 플러시
                    if received_count % 50 == 0:
                        f.flush()
                        
                except zmq.Again:
                    # 수신할 메시지가 없음, 잠시 대기
                    time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n사용자에 의해 종료됨")
                break
                
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(0.5)  # 에러 발생 시 잠시 대기
        
        # JSON 배열 종료 (마지막 쉼표 처리)
        f.seek(f.tell() - 2, 0) if f.tell() > 3 else f.seek(0, 2)  # 파일 위치 조정
        f.write("\n]")  # JSON 배열 종료
                
except KeyboardInterrupt:
    print("\n사용자에 의해 종료됨")
    
finally:
    # 소켓 정리
    cleanup_resources()
    
    # 통계 출력
    elapsed = time.time() - start_time
    print(f"\n총 수신: {received_count}개 메시지 (실제 데이터: {real_data_count}개, health_check: {received_count - real_data_count}개)")
    print(f"실행 시간: {elapsed:.1f}초")
    if elapsed > 0:
        print(f"평균 수신 속도: {received_count / elapsed:.1f}개/초")
    print(f"로그 파일: {log_file}")
