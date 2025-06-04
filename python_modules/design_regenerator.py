#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
디자인 재생성 스크립트 (30개 디자인 버전)

이 스크립트는 분석 결과에서 선별된 30개 디자인을 
ZMQ를 통해 그래스호퍼로 전송하여 재생성합니다.
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
    
    # 'rlhf_session_' 로 시작하는 모든 디렉토리 찾기
    session_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("rlhf_session_")]
    
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
        
        # 바인딩
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
                clean_actions.append(float(val))
            else:
                # NaN이면 중간값으로 대체
                clean_actions.append(0.5)
                print(f"경고: NaN 값이 0.5로 대체되었습니다.")
        
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
        
        # 메시 내보내기 요청
        request = {
            "request": "get_mesh",
            "datasetKey": dataset_key
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

def regenerate_all_designs(reference_data, session_id=1):
    """모든 선별된 디자인 재생성 (상위 + 랜덤)"""
    if not reference_data:
        print("오류: 기준 데이터가 없습니다.")
        return False
    
    # 상위 디자인과 랜덤 디자인 합치기
    all_designs = []
    
    if "top_designs" in reference_data:
        top_designs = reference_data["top_designs"]
        for i, design in enumerate(top_designs):
            design['type'] = 'top'
            design['rank'] = i + 1
            all_designs.append(design)
        print(f"상위 디자인: {len(top_designs)}개 로드됨")
    
    if "random_designs" in reference_data:
        random_designs = reference_data["random_designs"]
        for i, design in enumerate(random_designs):
            design['type'] = 'random'
            design['rank'] = i + 1
            all_designs.append(design)
        print(f"랜덤 디자인: {len(random_designs)}개 로드됨")
    
    if not all_designs:
        print("오류: 재생성할 디자인이 없습니다.")
        return False
    
    print(f"\n=== 총 {len(all_designs)}개 디자인 재생성 시작 ===")
    
    success_count = 0
    for i, design in enumerate(all_designs):
        print(f"\n[디자인 {i+1}/{len(all_designs)}] 재생성 중...")
        
        # 액션 값 추출
        action_values = design.get("action", [])
        if not action_values:
            print("  경고: 액션 값이 없습니다. 건너뜁니다.")
            continue
        
        # 상태 값 추출 (건축 지표)
        state_values = design.get("state", [])
        if len(state_values) >= 4:
            bcr = state_values[0]
            far = state_values[1]
            sunlight = state_values[2]
            svr = state_values[3]
            print(f"  건축 지표: BCR={bcr*100:.1f}%, FAR={far*100:.1f}%, 일조량={sunlight/1000:.1f}k, SV비율={svr:.3f}")
        
        # 액션 값 디버그 출력
        print(f"  액션 값: {action_values}")
        print(f"  보상: {design.get('reward', 'N/A')}")
        print(f"  타입: {design.get('type', 'unknown')}")
        
        # 액션 전송 및 디자인 재생성
        success = send_action_to_grasshopper(action_values)
        
        if success:
            # 그래스호퍼가 처리할 시간 부여
            print(f"  그래스호퍼가 처리할 시간 부여 중 (4초)...")
            time.sleep(4.0)
            
            # 세션 ID와 타입을 포함한 파일명 생성
            design_type = design.get('type', 'unknown')
            rank = design.get('rank', i+1)
            custom_filename = f"{design_type}_design_{rank}_session{session_id}"
            
            # 메시 내보내기
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
                    "state": state_values,
                    "reward": design.get("reward", 0.0),
                    "quality_score": design.get("quality_score", 0.0),
                    "composite_score": design.get("composite_score", 0.0),
                    "mesh_file": os.path.basename(mesh_file),
                    "reference": True,
                    "type": design_type,
                    "rank": rank,
                    "feedback_session": session_id,
                    "legal_compliance": design.get("legal_compliance", False),
                    "architecture_metrics": {
                        "bcr": state_values[0] if len(state_values) > 0 else None,
                        "far": state_values[1] if len(state_values) > 1 else None,
                        "winter_sunlight": state_values[2] if len(state_values) > 2 else None,
                        "sv_ratio": state_values[3] if len(state_values) > 3 else None
                    }
                }
                
                design_file = os.path.join(designs_dir, f"{custom_filename}.json")
                with open(design_file, 'w') as f:
                    json.dump(design_info, f, indent=2)
                
                print(f"  ✅ 디자인 정보가 {design_file}에 저장되었습니다.")
                success_count += 1
            
            # 다음 디자인 전에 잠시 대기
            time.sleep(0.5)
        else:
            print("  ❌ 액션 전송 실패. 다음 디자인으로 넘어갑니다.")
    
    print(f"\n✅ 디자인 재생성 완료: {success_count}/{len(all_designs)}개 성공")
    return True

def main():
    parser = argparse.ArgumentParser(description='선별된 모든 디자인 재생성 및 메시 내보내기')
    parser.add_argument('--session-dir', type=str, default=None,
                      help='분석 세션 디렉토리 경로 (지정하지 않으면 최신 디렉토리 사용)')
    parser.add_argument('--action-port', type=int, default=5556,
                      help='ZMQ 액션 전송 포트 (기본값: 5556)')
    parser.add_argument('--mesh-port', type=int, default=5558,
                      help='ZMQ 메시 내보내기 포트 (기본값: 5558)')
    parser.add_argument('--feedback-session', type=int, default=None,
                      help='현재 인간 피드백 세션 번호 (지정하지 않으면 입력 요청)')
    
    args = parser.parse_args()
    
    # 피드백 세션 번호 확인
    feedback_session = args.feedback_session
    if feedback_session is None:
        while True:
            try:
                feedback_session = int(input("현재 인간 피드백 세션 번호를 입력하세요 (1부터 시작): "))
                if feedback_session > 0:
                    break
                print("세션 번호는 1 이상의 정수여야 합니다.")
            except ValueError:
                print("숫자를 입력해주세요.")
    
    print(f"\n=== 인간 피드백 세션 #{feedback_session} 모든 디자인 재생성 시작 ===\n")
    
    # 기준 데이터 로드
    reference_data = load_reference_data(args.session_dir)
    
    if reference_data is None:
        print("오류: 기준 데이터를 로드할 수 없습니다.")
        return 1
    
    # 메타데이터 출력
    metadata = reference_data.get("metadata", {})
    print("\n=== 기준 데이터 메타데이터 ===")
    print(f"총 샘플 수: {metadata.get('total_samples', 'N/A')}")
    print(f"유효 샘플 수: {metadata.get('valid_samples', 'N/A')}")
    print(f"상태 차원: {metadata.get('state_dimensions', 'N/A')}")
    print(f"액션 차원: {metadata.get('action_dimensions', 'N/A')}")
    print(f"생성 시간: {metadata.get('generated_at', 'N/A')}")
    
    selection_criteria = metadata.get('selection_criteria', {})
    print(f"상위 디자인: {selection_criteria.get('top_designs', 'N/A')}개")
    print(f"랜덤 디자인: {selection_criteria.get('random_designs', 'N/A')}개")
    print(f"총 대상: {selection_criteria.get('total_target', 'N/A')}개")
    
    # 모든 디자인 재생성
    success = regenerate_all_designs(reference_data, feedback_session)
    
    if success:
        print("\n✅ 모든 디자인 재생성 및 메시 내보내기 완료!")
    else:
        print("\n❌ 디자인 재생성 중 오류가 발생했습니다.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())