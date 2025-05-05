#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
디자인 재생성 스크립트

이 스크립트는 분석 결과에서 최적 디자인을 선택하고
그 액션 값을 ZMQ를 통해 그래스호퍼로 전송하여 디자인을 재생성합니다.
재생성된 디자인의 메시를 내보내고 저장합니다.
"""

import os
import sys
import json
import time
import zmq
import argparse
import numpy as np
from pathlib import Path

# 프로젝트 루트 디렉토리 계산
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_latest_session_dir():
    """최신 세션 디렉토리 찾기"""
    data_dir = os.path.join(project_root, "data")
    
    # 'session_' 로 시작하는 모든 디렉토리 찾기
    session_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("session_")]
    
    if not session_dirs:
        return None
    
    # 타임스탬프 기준으로 정렬하여 최신 세션 찾기
    latest_session = sorted(session_dirs, reverse=True)[0]
    return os.path.join(data_dir, latest_session)

def load_reference_data(session_dir=None):
    """RLHF 기준 데이터 로드"""
    if session_dir is None:
        session_dir = find_latest_session_dir()
        if session_dir is None:
            print("오류: 세션 디렉토리를 찾을 수 없습니다.")
            return None
    
    reference_file = os.path.join(session_dir, "rlhf_reference_data.json")
    
    if not os.path.exists(reference_file):
        print(f"오류: {reference_file} 파일을 찾을 수 없습니다.")
        return None
    
    try:
        with open(reference_file, 'r') as f:
            reference_data = json.load(f)
        return reference_data
    except Exception as e:
        print(f"기준 데이터 로드 중 오류: {e}")
        return None

def send_action_to_grasshopper(action_values, zmq_port=5556):
    """액션 값을 ZMQ를 통해 그래스호퍼로 전송"""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    
    try:
        # 소켓 설정
        socket.setsockopt(zmq.LINGER, 200)
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
        
        # 여기가 달라진 부분: bind 사용
        bind_address = f"tcp://*:{zmq_port}"
        print(f"ZMQ 바인딩 시도: {bind_address}")
        socket.bind(bind_address)
        
        # 그래스호퍼가 연결할 시간 부여
        print("ZMQ 소켓이 바인딩되었습니다. 그래스호퍼 연결 대기 중...")
        time.sleep(1.0)
        
        # NaN 값 확인 및 처리
        clean_actions = []
        for val in action_values:
            if isinstance(val, (int, float)) and not np.isnan(val):
                clean_actions.append(val)
            else:
                # NaN이면 0으로 대체
                clean_actions.append(0.0)
                print(f"경고: NaN 값이 0.0으로 대체되었습니다.")
        
        # 액션 값 전송
        action_json = json.dumps(clean_actions)
        print(f"전송할 액션 값: {action_json}")
        socket.send_string(action_json)
        
        # 전송 후 약간 대기
        time.sleep(0.5)
        
        print("✅ 액션 값이 성공적으로 전송되었습니다.")
        return True
    
    except Exception as e:
        print(f"❌ 액션 전송 중 오류: {e}")
        return False
    
    finally:
        # 리소스 정리
        socket.close()
        context.term()

def check_mesh_exporter(zmq_port=5558):
    """MeshExporter가 응답하는지 확인"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    try:
        connect_address = f"tcp://localhost:{zmq_port}"
        print(f"MeshExporter 연결 시도: {connect_address}")
        socket.connect(connect_address)
        
        # 짧은 타임아웃 설정
        socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3초
        
        # ping 요청
        request = {"request": "ping"}
        socket.send_string(json.dumps(request))
        
        # 응답 대기
        response = socket.recv_string()
        response_data = json.loads(response)
        
        if response_data.get("status") == "success":
            print("✅ MeshExporter가 응답합니다.")
            return True
        else:
            print(f"⚠️ MeshExporter 응답이 예상과 다릅니다: {response}")
            return False
    
    except Exception as e:
        print(f"❌ MeshExporter 확인 중 오류: {e}")
        return False
    
    finally:
        socket.close()
        context.term()

def export_mesh(custom_filename, dataset_key="1", zmq_port=5558):
    """메시 내보내기 요청"""
    # 먼저 MeshExporter가 응답하는지 확인
    if not check_mesh_exporter(zmq_port):
        print("❌ MeshExporter가 응답하지 않습니다.")
        return None
    
    # 그래스호퍼가 메시를 생성할 충분한 시간 부여
    print("메시 생성을 위해 5초 대기 중...")
    time.sleep(5.0)
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    try:
        connect_address = f"tcp://localhost:{zmq_port}"
        print(f"메시 내보내기 요청: {connect_address}")
        socket.connect(connect_address)
        
        # 요청 타임아웃 설정
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10초
        
        # 메시 내보내기 요청 - 전달받은 dataset_key 사용
        request = {
            "request": "get_mesh",
            "datasetKey": dataset_key  # 이 부분이 매개변수와 일치해야 함
        }
        
        request_json = json.dumps(request)
        print(f"메시 요청 데이터: {request_json}")
        socket.send_string(request_json)
        
        # 응답 대기
        print("응답 대기 중...")
        response = socket.recv_string()
        response_data = json.loads(response)
        
        # 응답 확인
        if response_data.get("status") == "error":
            print(f"❌ 메시 내보내기 오류: {response_data.get('message', '알 수 없는 오류')}")
            return None
        
        # 메시 데이터 디렉토리 생성
        meshes_dir = os.path.join(project_root, "data", "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        
        # 메시 데이터를 커스텀 파일명으로 저장
        mesh_file = os.path.join(meshes_dir, f"{custom_filename}.json")
        with open(mesh_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        print(f"✅ 메시 데이터가 {mesh_file}에 저장되었습니다.")
        return mesh_file
    
    except zmq.error.Again:
        print("❌ 메시 응답 타임아웃. 그래스호퍼가 응답하지 않습니다.")
        return None
    except Exception as e:
        print(f"❌ 메시 내보내기 중 오류: {e}")
        return None
    
    finally:
        # 리소스 정리
        socket.close()
        context.term()

def regenerate_top_designs(reference_data, top_n=3, session_id=1):
    """최고 보상을 받은 상위 N개 디자인 재생성"""
    if not reference_data or "top_designs" not in reference_data:
        print("오류: 기준 데이터에 top_designs가 없습니다.")
        return False
    
    top_designs = reference_data["top_designs"][:top_n]
    
    if not top_designs:
        print("오류: 기준 데이터에 디자인이 없습니다.")
        return False
    
    print(f"\n=== 상위 {len(top_designs)}개 디자인 재생성 ===")
    
    for i, design in enumerate(top_designs):
        print(f"\n[디자인 {i+1}/{len(top_designs)}] 재생성 중...")
        
        # 액션 값 추출
        action_values = design.get("action", [])
        if not action_values:
            print("  경고: 액션 값이 없습니다. 건너뜁니다.")
            continue
        
        # 액션 값 디버그 출력
        print(f"  액션 값: {action_values}")
        print(f"  보상: {design.get('reward', 'N/A')}")
        
        # 액션 전송 및 디자인 재생성
        success = send_action_to_grasshopper(action_values)
        
        if success:
            # 그래스호퍼가 처리할 시간 부여
            print("  그래스호퍼가 처리할 시간 부여 중...")
            time.sleep(1.0)
            
            # 세션 ID를 포함한 파일명 생성
            custom_filename = f"top_design_{i+1}_session{session_id}"
            
            # 세션 ID를 datasetKey로 사용하여 메시 내보내기
            mesh_file = export_mesh(custom_filename, str(session_id))
            
            if mesh_file:
                # 디자인 정보 저장
                designs_dir = os.path.join(project_root, "data", "designs")
                os.makedirs(designs_dir, exist_ok=True)
                
                design_info = {
                    "id": custom_filename,
                    "timestamp": int(time.time() * 1000),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": action_values,
                    "state": design.get("state", []),
                    "reward": design.get("reward", 0.0),
                    "mesh_file": os.path.basename(mesh_file),
                    "reference": True,
                    "type": "top",
                    "rank": i + 1,
                    "feedback_session": session_id  # 세션 ID 추가
                }
                
                design_file = os.path.join(designs_dir, f"{custom_filename}.json")
                with open(design_file, 'w') as f:
                    json.dump(design_info, f, indent=2)
                
                print(f"  ✅ 디자인 정보가 {design_file}에 저장되었습니다.")
            
            # 다음 디자인 전에 잠시 대기
            time.sleep(0.5)
        else:
            print("  ❌ 액션 전송 실패. 다음 디자인으로 넘어갑니다.")
    
    return True

def regenerate_diverse_designs(reference_data):
    """다양한 클러스터를 대표하는 디자인 재생성"""
    if not reference_data or "diverse_designs" not in reference_data:
        print("오류: 기준 데이터에 diverse_designs가 없습니다.")
        return False
    
    diverse_designs = reference_data["diverse_designs"]
    
    if not diverse_designs:
        print("오류: 기준 데이터에 다양한 디자인이 없습니다.")
        return False
    
    print(f"\n=== 다양한 클러스터 디자인 {len(diverse_designs)}개 재생성 ===")
    
    for i, design in enumerate(diverse_designs):
        print(f"\n[다양한 디자인 {i+1}/{len(diverse_designs)}] 재생성 중...")
        
        # 액션 값 추출
        action_values = design.get("action", [])
        if not action_values:
            print("  경고: 액션 값이 없습니다. 건너뜁니다.")
            continue
        
        # 액션 값 디버그 출력
        print(f"  액션 값: {action_values}")
        print(f"  보상: {design.get('reward', 'N/A')}")
        print(f"  클러스터: {design.get('cluster', 'N/A')}")
        
        # 액션 전송 및 디자인 재생성
        success = send_action_to_grasshopper(action_values)
        
        if success:
            # 그래스호퍼가 처리할 시간 부여
            print("  그래스호퍼가 처리할 시간 부여 중...")
            time.sleep(1.0)
            
            # 커스텀 파일명 생성
            custom_filename = f"diverse_design_c{design.get('cluster', i)}_{int(time.time())}"
            
            # 메시 내보내기
            mesh_file = export_mesh(custom_filename)
            
            if mesh_file:
                # 디자인 정보 저장
                designs_dir = os.path.join(project_root, "data", "designs")
                os.makedirs(designs_dir, exist_ok=True)
                
                design_info = {
                    "id": custom_filename,  # dataset_key 대신 custom_filename 사용
                    "timestamp": int(time.time() * 1000),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": action_values,
                    "state": design.get("state", []),
                    "reward": design.get("reward", 0.0),
                    "mesh_file": os.path.basename(mesh_file),
                    "reference": True,
                    "type": "diverse",
                    "cluster": design.get("cluster", i)
                }
                
                design_file = os.path.join(designs_dir, f"{custom_filename}.json")
                with open(design_file, 'w') as f:
                    json.dump(design_info, f, indent=2)
                
                print(f"  ✅ 디자인 정보가 {design_file}에 저장되었습니다.")
            
            # 다음 디자인 전에 잠시 대기
            time.sleep(0.5)
        else:
            print("  ❌ 액션 전송 실패. 다음 디자인으로 넘어갑니다.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='최적 디자인 재생성 및 메시 내보내기')
    parser.add_argument('--session-dir', type=str, default=None,
                        help='분석 세션 디렉토리 경로 (지정하지 않으면 최신 디렉토리 사용)')
    parser.add_argument('--action-port', type=int, default=5556,
                        help='ZMQ 액션 전송 포트 (기본값: 5556)')
    parser.add_argument('--mesh-port', type=int, default=5558,
                        help='ZMQ 메시 내보내기 포트 (기본값: 5558)')
    parser.add_argument('--top-n', type=int, default=3,
                        help='재생성할 최적 디자인 개수 (기본값: 3)')
    parser.add_argument('--regenerate-top', action='store_true',
                        help='최적 디자인만 재생성')
    parser.add_argument('--regenerate-diverse', action='store_true',
                        help='다양한 디자인만 재생성')
    # 피드백 세션 번호 인자 추가
    parser.add_argument('--feedback-session', type=int, default=None,
                      help='현재 인간 피드백 세션 번호 (지정하지 않으면 입력 요청)')
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # 피드백 세션 번호 확인
    feedback_session = args.feedback_session
    if feedback_session is None:
        # 사용자에게 입력 요청
        while True:
            try:
                feedback_session = int(input("현재 인간 피드백 세션 번호를 입력하세요 (1부터 시작): "))
                if feedback_session > 0:
                    break
                print("세션 번호는 1 이상의 정수여야 합니다.")
            except ValueError:
                print("숫자를 입력해주세요.")
    
    print(f"\n=== 인간 피드백 세션 #{feedback_session} 디자인 재생성 시작 ===\n")
    
    # 기본값: 모든 디자인 재생성
    regenerate_top = args.regenerate_top
    regenerate_diverse = args.regenerate_diverse
    
    if not regenerate_top and not regenerate_diverse:
        regenerate_top = True
        regenerate_diverse = True
    
    # 기준 데이터 로드
    reference_data = load_reference_data(args.session_dir)
    
    if reference_data is None:
        print("오류: 기준 데이터를 로드할 수 없습니다.")
        return 1
    
    # 메타데이터 출력
    metadata = reference_data.get("metadata", {})
    print("\n=== 기준 데이터 메타데이터 ===")
    print(f"총 샘플 수: {metadata.get('total_samples', 'N/A')}")
    print(f"상태 차원: {metadata.get('state_dimensions', 'N/A')}")
    print(f"액션 차원: {metadata.get('action_dimensions', 'N/A')}")
    print(f"생성 시간: {metadata.get('generated_at', 'N/A')}")
    
    # 최적 디자인 재생성 시 세션 번호 전달
    if regenerate_top:
        regenerate_top_designs(reference_data, args.top_n, feedback_session)
    
    # 다양한 디자인 재생성 시 세션 번호 전달
    if regenerate_diverse:
        regenerate_diverse_designs(reference_data, feedback_session)
    
    print("\n✅ 디자인 재생성 및 메시 내보내기 완료!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
