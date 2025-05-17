# zmq_state_receiver.py (건축 설계 최적화 버전)
import zmq
import json
import time
import sys
import os
import atexit
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "data", "zmq_logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f"state_reward_log_{timestamp}.json")
csv_file = os.path.join(log_dir, f"architecture_metrics_{timestamp}.csv")
port = 5557  # 새 포트 사용
max_idle_time = 30  # 늘어난 계산 시간 고려

# ZMQ 설정
context = None
socket = None

# 건축 지표 통계 저장
architecture_metrics = {
    'timestamp': [],
    'bcr': [],
    'far': [],
    'sunlight': [],
    'bcr_limit_exceeded': [],
    'far_limit_exceeded': [],
    'reward': []
}

# 건축 제한 값 (명령줄 인자로 변경 가능)
BCR_LIMIT = 0.6  # 60%
FAR_LIMIT = 4.0  # 400%

def cleanup_resources():
    global socket, context
    print("리소스 정리 중...")
    
    # CSV 파일로 통계 저장
    try:
        if architecture_metrics['timestamp']:
            df = pd.DataFrame(architecture_metrics)
            df.to_csv(csv_file, index=False)
            print(f"건축 지표 통계가 {csv_file}에 저장되었습니다.")
            
            # 간단한 시각화 생성
            create_architecture_visualizations(df)
    except Exception as e:
        print(f"통계 저장 중 오류: {e}")
    
    # ZMQ 리소스 정리
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

def create_architecture_visualizations(df):
    """수집된 건축 지표 시각화"""
    try:
        # 출력 폴더 생성
        viz_dir = os.path.join(log_dir, f"visualizations_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. BCR/FAR 시계열 플롯
        plt.figure(figsize=(12, 6))
        plt.plot(df['bcr'], label='BCR')
        plt.plot(df['far'] / 10, label='FAR (scaled ÷10)')  # FAR는 스케일 조정
        plt.axhline(y=BCR_LIMIT, color='r', linestyle='--', label=f'BCR Limit ({BCR_LIMIT*100:.0f}%)')
        plt.axhline(y=FAR_LIMIT/10, color='orange', linestyle='--', label=f'FAR Limit ({FAR_LIMIT*100:.0f}%)')
        plt.title('Building Coverage & Floor Area Ratio')
        plt.xlabel('Sample')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'bcr_far_trend.png'))
        
        # 2. 일조량 시계열 플롯
        plt.figure(figsize=(12, 6))
        plt.plot(df['sunlight'], label='Sunlight', color='orange')
        plt.title('Sunlight Exposure')
        plt.xlabel('Sample')
        plt.ylabel('Normalized Value (0-1)')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'sunlight_trend.png'))
        
        # 3. 보상값 시계열 플롯
        plt.figure(figsize=(12, 6))
        plt.plot(df['reward'], label='Reward', color='green')
        plt.title('Reward Values')
        plt.xlabel('Sample')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'reward_trend.png'))
        
        # 4. BCR vs FAR 산점도
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(df['bcr']*100, df['far']*100, c=df['reward'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Reward')
        plt.axvline(x=BCR_LIMIT*100, color='r', linestyle='--', label=f'BCR Limit ({BCR_LIMIT*100:.0f}%)')
        plt.axhline(y=FAR_LIMIT*100, color='r', linestyle='--', label=f'FAR Limit ({FAR_LIMIT*100:.0f}%)')
        plt.title('BCR vs FAR with Reward')
        plt.xlabel('BCR (%)')
        plt.ylabel('FAR (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'bcr_far_scatter.png'))
        
        print(f"시각화 파일이 {viz_dir}에 저장되었습니다.")
    except Exception as e:
        print(f"시각화 생성 중 오류: {e}")

# 종료 시 정리 함수 등록
atexit.register(cleanup_resources)

# 명령행 인자 파싱
import argparse
parser = argparse.ArgumentParser(description='ZMQ State Receiver for Architecture Optimization')
parser.add_argument('--port', type=int, default=5557, help='ZMQ 수신 포트 (기본값: 5557)')
parser.add_argument('--bcr-limit', type=float, default=0.6, help='BCR 법적 한도 (기본값: 0.6)')
parser.add_argument('--far-limit', type=float, default=4.0, help='FAR 법적 한도 (기본값: 4.0)')
args = parser.parse_args()

# 설정 업데이트
port = args.port
BCR_LIMIT = args.bcr_limit
FAR_LIMIT = args.far_limit

# 소켓 초기화
try:
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)  # 명시적 linger 설정
    bind_address = f"tcp://localhost:{port}"
    
    print(f"ZMQ PULL 소켓 바인딩 중: {bind_address}")
    socket.bind(bind_address)
    print(f"수신 대기 중... 데이터는 {log_file}에 저장됩니다")
    print(f"건축 지표: BCR 한도 {BCR_LIMIT*100:.1f}%, FAR 한도 {FAR_LIMIT*100:.1f}%")
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
                        
                        # 건축 지표 추출 및 저장
                        try:
                            state = data.get('state', [])
                            if isinstance(state, list) and len(state) >= 3:
                                bcr = state[0]
                                far = state[1]
                                sunlight = state[2]
                                
                                architecture_metrics['timestamp'].append(int(time.time()))
                                architecture_metrics['bcr'].append(bcr)
                                architecture_metrics['far'].append(far)
                                architecture_metrics['sunlight'].append(sunlight)
                                architecture_metrics['bcr_limit_exceeded'].append(bcr > BCR_LIMIT)
                                architecture_metrics['far_limit_exceeded'].append(far > FAR_LIMIT)
                                architecture_metrics['reward'].append(data.get('reward', 0))
                        except Exception as e:
                            print(f"건축 지표 처리 중 오류: {e}")
                    
                    # 콘솔에 출력
                    if received_count % 10 == 0:  # 10개마다 출력
                        elapsed = time.time() - start_time
                        rate = received_count / elapsed if elapsed > 0 else 0
                        print(f"수신: {received_count}개 (실제 데이터: {real_data_count}개, {rate:.1f}개/초)")
                        
                        # 데이터 샘플 출력 (health_check가 아닌 경우에만)
                        if not is_health_check:
                            # 건축 지표 형식으로 출력
                            try:
                                state = data.get('state', [])
                                if isinstance(state, list) and len(state) >= 3:
                                    bcr = state[0]
                                    far = state[1]
                                    sunlight = state[2]
                                    reward = data.get('reward', 0)
                                    
                                    print(f"BCR: {bcr*100:.1f}% {'(초과!)' if bcr > BCR_LIMIT else ''}, " +
                                          f"FAR: {far*100:.1f}% {'(초과!)' if far > FAR_LIMIT else ''}, " +
                                          f"일조량: {sunlight:.2f}, 보상: {reward:.1f}")
                                else:
                                    print(f"샘플: {message[:100]}..." if len(message) > 100 else f"샘플: {message}")
                            except:
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
    # 소켓 정리는 cleanup_resources에서 처리
    
    # 통계 출력
    elapsed = time.time() - start_time
    print(f"\n총 수신: {received_count}개 메시지 (실제 데이터: {real_data_count}개, health_check: {received_count - real_data_count}개)")
    print(f"실행 시간: {elapsed:.1f}초")
    if elapsed > 0:
        print(f"평균 수신 속도: {received_count / elapsed:.1f}개/초")
    print(f"로그 파일: {log_file}")
    
    # 건축 지표 통계 출력
    if architecture_metrics['bcr']:
        avg_bcr = sum(architecture_metrics['bcr']) / len(architecture_metrics['bcr'])
        avg_far = sum(architecture_metrics['far']) / len(architecture_metrics['far'])
        avg_sunlight = sum(architecture_metrics['sunlight']) / len(architecture_metrics['sunlight'])
        bcr_violations = sum(architecture_metrics['bcr_limit_exceeded'])
        far_violations = sum(architecture_metrics['far_limit_exceeded'])
        
        print("\n=== 건축 지표 통계 ===")
        print(f"평균 BCR: {avg_bcr*100:.1f}% (한도: {BCR_LIMIT*100:.1f}%)")
        print(f"평균 FAR: {avg_far*100:.1f}% (한도: {FAR_LIMIT*100:.1f}%)")
        print(f"평균 일조량: {avg_sunlight:.3f}")
        print(f"BCR 위반 횟수: {bcr_violations}/{len(architecture_metrics['bcr'])} ({bcr_violations/len(architecture_metrics['bcr'])*100:.1f}%)")
        print(f"FAR 위반 횟수: {far_violations}/{len(architecture_metrics['far'])} ({far_violations/len(architecture_metrics['far'])*100:.1f}%)")