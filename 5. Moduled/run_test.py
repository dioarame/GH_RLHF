# run_test.py
import argparse
import os
import time
import traceback
import json
import numpy as np

# 모듈화된 구성요소 임포트
from testing import test_zmq_push_connection, test_zmq_req_connection, check_compute_server
from grasshopper_env.utils import ComputeClient

def parse_test_args():
    """테스트용 명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Grasshopper ZMQ Communication Test')
    parser.add_argument('--gh-path', type=str, default=r"C:/Users/valen/Desktop/Dev/AS_B.gh",
                        help='Grasshopper 정의 파일 경로')
    parser.add_argument('--compute-url', type=str, default="http://localhost:6500/grasshopper",
                        help='Rhino.Compute 서버 URL')
    parser.add_argument('--port', type=int, default=5556,
                        help='ZMQ 서버 포트 (Python Bind)')
    parser.add_argument('--zmq-mode', type=str, choices=['push', 'rep'], default='push',
                        help='ZMQ 통신 모드 (push=PUSH-PULL, rep=REQ-REP) (기본값: push)')
    parser.add_argument('--slider-info-param', type=str, default="SliderInfo",
                        help='슬라이더 정보 파라미터 이름')
    parser.add_argument('--test-count', type=int, default=10,
                        help='전송할 테스트 메시지 수')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='테스트 메시지 전송 간격 (초)')
    return parser.parse_args()

def test_slider_control():
    """슬라이더 제어 테스트 실행"""
    args = parse_test_args()
    
    print("===== Grasshopper 슬라이더 제어 테스트 =====")
    
    # 설정
    compute_url = args.compute_url
    gh_path = args.gh_path
    zmq_port = args.port
    use_push_mode = args.zmq_mode == 'push'
    slider_info_param = args.slider_info_param
    test_count = args.test_count
    send_interval = args.interval
    
    # 1. Compute 서버 상태 확인
    print("\n[1/4] Rhino.Compute 서버 상태 확인...")
    if not check_compute_server(compute_url):
        print("Compute 서버에 연결할 수 없습니다. 테스트를 중단합니다.")
        return
    
    # 2. Compute 클라이언트 생성 및 슬라이더 정보 가져오기
    print("\n[2/4] 슬라이더 정보 가져오는 중...")
    compute_client = ComputeClient(compute_url, gh_path)
    
    try:
        response = compute_client.call_compute()
        if not response:
            print("Compute API 응답을 받을 수 없습니다. 테스트를 중단합니다.")
            return
            
        slider_info_raw = compute_client.get_param_data(response, slider_info_param)
        if not slider_info_raw:
            print(f"슬라이더 정보({slider_info_param})를 찾을 수 없습니다. 테스트를 중단합니다.")
            return
            
        slider_infos = compute_client.parse_slider_info(slider_info_raw)
        if not slider_infos:
            print("슬라이더 정보를 파싱할 수 없습니다. 테스트를 중단합니다.")
            return
            
        print(f"\n[3/4] {len(slider_infos)}개 슬라이더 정보 파싱 완료:")
        for i, info in enumerate(slider_infos):
            print(f"  슬라이더 {i+1}: Min={info[0]}, Max={info[1]}, Rounding={info[2]}")
            
    except Exception as e:
        print(f"슬라이더 정보 처리 중 오류 발생: {e}")
        traceback.print_exc()
        return
    
    # 3. ZMQ 테스트 실행
    print(f"\n[4/4] ZMQ 테스트 ({args.zmq_mode.upper()} 모드)...")
    
    # 테스트 액션 생성
    test_actions = []
    for i in range(test_count):
        action = []
        for min_val, max_val, _ in slider_infos:
            # 각 슬라이더별 범위 내에서 랜덤 값 생성
            if i == 0:  # 첫 번째는 최소값
                action.append(min_val)
            elif i == test_count - 1:  # 마지막은 최대값
                action.append(max_val)
            else:  # 나머지는 랜덤값
                import random
                action.append(random.uniform(min_val, max_val))
        test_actions.append(action)
    
    # 통신 모드에 따라 다른 테스트 수행
    if use_push_mode:
        test_result = test_zmq_push_connection(zmq_port, test_actions=test_actions)
    else:
        test_result = test_zmq_req_connection(zmq_port, test_actions=test_actions)
        
    print(f"\nZMQ 테스트 결과: {'성공' if test_result else '실패'}")
    
    # 리소스 정리
    compute_client.close()
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_slider_control()