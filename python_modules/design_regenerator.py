#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 디자인 재생성 스크립트 (쌍대비교용 업데이트)

이 스크립트는 분석 결과에서 선별된 디자인들을 ZMQ를 통해 그래스호퍼로 전송하여 
실제 3D 메시를 생성하고, 웹 인터페이스에서 쌍대비교에 사용할 수 있도록 준비합니다.
"""

import os
import sys
import json
import time
import zmq
import argparse
import numpy as np
import math
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리 계산
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_latest_session_dir():
    """최신 RLHF 세션 디렉토리 찾기"""
    data_dir = os.path.join(project_root, "data")
    
    # 'rlhf_session_' 로 시작하는 모든 디렉토리 찾기
    session_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and 
                   (d.startswith("rlhf_session_") or d.startswith("session_"))]
    
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
            print("❌ RLHF 세션 디렉토리를 찾을 수 없습니다.")
            return None
    
    reference_file = os.path.join(session_dir, "rlhf_reference_data.json")
    
    if not os.path.exists(reference_file):
        print(f"❌ {reference_file} 파일을 찾을 수 없습니다.")
        return None
    
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
        print(f"✅ RLHF 기준 데이터 로드 완료: {reference_file}")
        return reference_data
    except Exception as e:
        print(f"❌ 기준 데이터 로드 중 오류: {e}")
        return None

def validate_action_values(action_values):
    """액션 값 유효성 검사 및 정리"""
    if not action_values:
        return []
    
    clean_actions = []
    for val in action_values:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                clean_actions.append(0.0)
                print(f"⚠️ 경고: 유효하지 않은 액션 값 {val}을 0.0으로 대체")
            else:
                clean_actions.append(float(val))
        else:
            try:
                clean_val = float(val)
                if math.isnan(clean_val) or math.isinf(clean_val):
                    clean_actions.append(0.0)
                else:
                    clean_actions.append(clean_val)
            except (ValueError, TypeError):
                clean_actions.append(0.0)
                print(f"⚠️ 경고: 변환할 수 없는 액션 값 {val}을 0.0으로 대체")
    
    return clean_actions

def send_action_to_grasshopper(action_values, zmq_port=5556, timeout=3000):
    """액션 값을 ZMQ를 통해 그래스호퍼로 전송"""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    
    try:
        # 소켓 설정
        socket.setsockopt(zmq.LINGER, 1000)
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
        
        # 액션 값 검증 및 정리
        clean_actions = validate_action_values(action_values)
        
        if not clean_actions:
            print("⚠️ 경고: 유효한 액션 값이 없습니다.")
            return False
        
        # ZMQ 바인딩
        bind_address = f"tcp://*:{zmq_port}"
        print(f"🔗 ZMQ 바인딩: {bind_address}")
        socket.bind(bind_address)
        
        # 그래스호퍼 연결 대기
        print("⏳ 그래스호퍼 연결 대기 중...")
        time.sleep(1.5)
        
        # 액션 값 전송
        action_json = json.dumps(clean_actions)
        print(f"📤 전송 액션: {action_json}")
        socket.send_string(action_json)
        
        # 전송 후 대기
        time.sleep(0.8)
        
        print("✅ 액션 값 전송 완료")
        return True
    
    except Exception as e:
        print(f"❌ 액션 전송 중 오류: {e}")
        return False
    
    finally:
        socket.close()
        context.term()

def check_mesh_exporter(zmq_port=5558, timeout=5000):
    """MeshExporter 상태 확인"""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    try:
        connect_address = f"tcp://localhost:{zmq_port}"
        print(f"🔍 MeshExporter 연결 확인: {connect_address}")
        socket.connect(connect_address)
        
        socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        # ping 요청
        request = {"request": "ping"}
        socket.send_string(json.dumps(request))
        
        # 응답 대기
        response = socket.recv_string()
        response_data = json.loads(response)
        
        if response_data.get("status") == "success":
            print("✅ MeshExporter 응답 확인")
            return True
        else:
            print(f"⚠️ MeshExporter 응답 이상: {response}")
            return False
    
    except zmq.error.Again:
        print("❌ MeshExporter 응답 타임아웃")
        return False
    except Exception as e:
        print(f"❌ MeshExporter 확인 중 오류: {e}")
        return False
    
    finally:
        socket.close()
        context.term()

def export_mesh(design_id, dataset_key="1", zmq_port=5558, timeout=15000):
    """메시 내보내기 요청"""
    # MeshExporter 상태 확인
    if not check_mesh_exporter(zmq_port, timeout//3):
        print("❌ MeshExporter가 응답하지 않습니다.")
        return None
    
    # 메시 생성 대기
    print("⏳ 메시 생성 대기 중 (7초)...")
    time.sleep(7.0)
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    try:
        connect_address = f"tcp://localhost:{zmq_port}"
        print(f"📥 메시 내보내기 요청: {connect_address}")
        socket.connect(connect_address)
        
        socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        # 메시 내보내기 요청
        request = {
            "request": "get_mesh",
            "datasetKey": str(dataset_key)
        }
        
        request_json = json.dumps(request)
        print(f"📋 메시 요청: {request_json}")
        socket.send_string(request_json)
        
        # 응답 대기
        print("⏳ 메시 응답 대기 중...")
        response = socket.recv_string()
        response_data = json.loads(response)
        
        # 응답 검증
        if response_data.get("status") == "error":
            print(f"❌ 메시 내보내기 오류: {response_data.get('message', '알 수 없는 오류')}")
            return None
        
        # 메시 데이터 저장
        meshes_dir = os.path.join(project_root, "data", "meshes")
        os.makedirs(meshes_dir, exist_ok=True)
        
        mesh_file = os.path.join(meshes_dir, f"{design_id}.json")
        with open(mesh_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 메시 데이터 저장: {mesh_file}")
        return mesh_file
    
    except zmq.error.Again:
        print("❌ 메시 응답 타임아웃")
        return None
    except Exception as e:
        print(f"❌ 메시 내보내기 중 오류: {e}")
        return None
    
    finally:
        socket.close()
        context.term()

def regenerate_comparison_designs(reference_data, design_types=['top', 'diverse'], 
                                session_id=1, max_designs_per_type=10):
    """쌍대비교용 디자인들 재생성"""
    if not reference_data:
        print("❌ 기준 데이터가 없습니다.")
        return False
    
    total_generated = 0
    failed_generations = 0
    
    print(f"\n🎯 RLHF 쌍대비교용 디자인 재생성 시작 (세션 #{session_id})")
    print(f"📋 생성할 타입: {', '.join(design_types)}")
    
    # designs 및 meshes 디렉토리 생성
    designs_dir = os.path.join(project_root, "data", "designs")
    meshes_dir = os.path.join(project_root, "data", "meshes") 
    os.makedirs(designs_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)
    
    # 각 타입별 디자인 처리
    for design_type in design_types:
        if design_type == 'top' and 'top_designs' in reference_data:
            designs_to_process = reference_data['top_designs'][:max_designs_per_type]
            type_label = "최고 성능"
        elif design_type == 'diverse' and 'diverse_designs' in reference_data:
            designs_to_process = reference_data['diverse_designs'][:max_designs_per_type]
            type_label = "다양한 탐색"
        else:
            print(f"⚠️ '{design_type}' 타입 데이터를 찾을 수 없습니다.")
            continue
        
        print(f"\n🏗️ {type_label} 디자인 재생성 중... ({len(designs_to_process)}개)")
        
        for i, design in enumerate(designs_to_process):
            print(f"\n[{design_type.upper()} {i+1}/{len(designs_to_process)}] 재생성 중...")
            
            # 액션 값 추출
            action_values = design.get("action", [])
            if not action_values:
                print("⚠️ 액션 값이 없습니다. 건너뜁니다.")
                failed_generations += 1
                continue
            
            # 디자인 정보 출력
            state_labels = design.get("state_labels", {})
            quality_score = design.get("quality_score", 0.0)
            legal_compliance = design.get("legal_compliance", False)
            
            print(f"  🎯 품질점수: {quality_score:.3f}")
            print(f"  ⚖️ 법적준수: {'예' if legal_compliance else '아니오'}")
            print(f"  📊 보상값: {design.get('reward', 0.0):.4f}")
            
            if state_labels:
                print(f"  🏢 BCR: {state_labels.get('BCR', 0)*100:.1f}%")
                print(f"  🏙️ FAR: {state_labels.get('FAR', 0)*100:.1f}%") 
                print(f"  ☀️ 일조량: {state_labels.get('Winter_Sunlight', 0)/1000:.1f}k kWh")
                print(f"  📐 SV비율: {state_labels.get('SV_Ratio', 0):.3f}")
            
            print(f"  🎮 액션: {action_values}")
            
            # 액션 전송 및 메시 생성
            success = send_action_to_grasshopper(action_values)
            
            if success:
                # 그래스호퍼 처리 대기
                print("⏳ 그래스호퍼 계산 대기 중...")
                time.sleep(5.0)
                
                # 고유 디자인 ID 생성
                original_id = design.get('id', f'{design_type}_{i}')
                unique_id = f"{original_id}_session{session_id}_{int(time.time())}"
                
                # 메시 내보내기 (세션 ID를 datasetKey로 사용)
                mesh_file = export_mesh(unique_id, str(session_id))
                
                if mesh_file:
                    # 디자인 메타데이터 생성
                    design_metadata = {
                        "id": unique_id,
                        "original_id": original_id,
                        "session_id": session_id,
                        "timestamp": int(time.time() * 1000),
                        "created_at": datetime.now().isoformat(),
                        "type": design_type,
                        "source": "rlhf_regeneration",
                        
                        # RL 학습 데이터
                        "action": action_values,
                        "state": design.get("state", []),
                        "reward": design.get("reward", 0.0),
                        
                        # RLHF 평가 데이터
                        "quality_score": quality_score,
                        "legal_compliance": legal_compliance,
                        "sustainability_score": design.get("sustainability_score", 0.0),
                        "constraint_violations": design.get("constraint_violations", 0),
                        "composite_score": design.get("composite_score", 0.0),
                        
                        # 건축 지표
                        "state_labels": state_labels,
                        "architecture_metrics": {
                            "bcr": state_labels.get("BCR", 0.0),
                            "far": state_labels.get("FAR", 0.0), 
                            "winter_sunlight": state_labels.get("Winter_Sunlight", 0.0),
                            "sv_ratio": state_labels.get("SV_Ratio", 0.0)
                        },
                        
                        # 메시 파일 정보
                        "mesh_file": os.path.basename(mesh_file),
                        "mesh_file_path": mesh_file,
                        
                        # 클러스터 정보 (diverse 타입의 경우)
                        "cluster": design.get("cluster", -1) if design_type == 'diverse' else -1,
                        
                        # 웹 인터페이스용 메타데이터
                        "web_interface": {
                            "ready_for_comparison": True,
                            "display_name": f"{type_label} 디자인 {i+1}",
                            "quality_tier": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low",
                            "legal_status": "compliant" if legal_compliance else "non_compliant"
                        }
                    }
                    
                    # 디자인 메타데이터 저장
                    design_file = os.path.join(designs_dir, f"{unique_id}.json")
                    with open(design_file, 'w', encoding='utf-8') as f:
                        json.dump(design_metadata, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ 디자인 완료: {unique_id}")
                    print(f"📁 메타데이터: {design_file}")
                    total_generated += 1
                else:
                    print("❌ 메시 생성 실패")
                    failed_generations += 1
                
                # 다음 디자인 전 대기
                time.sleep(1.0)
            else:
                print("❌ 액션 전송 실패")
                failed_generations += 1
    
    print(f"\n🎉 RLHF 디자인 재생성 완료!")
    print(f"✅ 성공: {total_generated}개")
    print(f"❌ 실패: {failed_generations}개")
    print(f"📁 저장 위치:")
    print(f"   - 디자인 메타데이터: {designs_dir}")
    print(f"   - 3D 메시 데이터: {meshes_dir}")
    
    return total_generated > 0

def create_web_interface_index(designs_dir, output_file=None):
    """웹 인터페이스용 디자인 인덱스 생성"""
    if output_file is None:
        output_file = os.path.join(project_root, "data", "designs_index.json")
    
    try:
        designs_index = {
            "generated_at": datetime.now().isoformat(),
            "total_designs": 0,
            "designs_by_type": {},
            "designs_by_quality": {},
            "designs_list": []
        }
        
        # 디자인 파일들 스캔
        design_files = [f for f in os.listdir(designs_dir) if f.endswith('.json')]
        
        for design_file in design_files:
            try:
                with open(os.path.join(designs_dir, design_file), 'r', encoding='utf-8') as f:
                    design_data = json.load(f)
                
                # 웹 인터페이스용 요약 정보
                design_summary = {
                    "id": design_data.get("id"),
                    "type": design_data.get("type", "unknown"),
                    "quality_score": design_data.get("quality_score", 0.0),
                    "legal_compliance": design_data.get("legal_compliance", False),
                    "reward": design_data.get("reward", 0.0),
                    "mesh_file": design_data.get("mesh_file"),
                    "display_name": design_data.get("web_interface", {}).get("display_name", design_data.get("id")),
                    "quality_tier": design_data.get("web_interface", {}).get("quality_tier", "medium"),
                    "legal_status": design_data.get("web_interface", {}).get("legal_status", "unknown"),
                    "architecture_metrics": design_data.get("architecture_metrics", {}),
                    "created_at": design_data.get("created_at")
                }
                
                designs_index["designs_list"].append(design_summary)
                
                # 타입별 분류
                design_type = design_summary["type"]
                if design_type not in designs_index["designs_by_type"]:
                    designs_index["designs_by_type"][design_type] = 0
                designs_index["designs_by_type"][design_type] += 1
                
                # 품질별 분류
                quality_tier = design_summary["quality_tier"]
                if quality_tier not in designs_index["designs_by_quality"]:
                    designs_index["designs_by_quality"][quality_tier] = 0
                designs_index["designs_by_quality"][quality_tier] += 1
                
            except Exception as e:
                print(f"⚠️ 디자인 파일 처리 중 오류: {design_file} - {e}")
                continue
        
        designs_index["total_designs"] = len(designs_index["designs_list"])
        
        # 품질 점수 기준으로 정렬
        designs_index["designs_list"].sort(key=lambda x: x["quality_score"], reverse=True)
        
        # 인덱스 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(designs_index, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 웹 인터페이스 인덱스 생성: {output_file}")
        print(f"📊 총 디자인: {designs_index['total_designs']}개")
        print(f"📋 타입별: {designs_index['designs_by_type']}")
        print(f"🎯 품질별: {designs_index['designs_by_quality']}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ 인덱스 생성 중 오류: {e}")
        return None

def validate_generated_designs(designs_dir, meshes_dir):
    """생성된 디자인들의 유효성 검증"""
    print("\n🔍 생성된 디자인 유효성 검증 중...")
    
    design_files = [f for f in os.listdir(designs_dir) if f.endswith('.json')]
    
    valid_designs = 0
    invalid_designs = 0
    missing_meshes = 0
    
    for design_file in design_files:
        try:
            with open(os.path.join(designs_dir, design_file), 'r', encoding='utf-8') as f:
                design_data = json.load(f)
            
            # 필수 필드 검증
            required_fields = ['id', 'action', 'state', 'mesh_file']
            missing_fields = [field for field in required_fields if field not in design_data]
            
            if missing_fields:
                print(f"❌ {design_file}: 누락된 필드 {missing_fields}")
                invalid_designs += 1
                continue
            
            # 메시 파일 존재 확인
            mesh_file_path = os.path.join(meshes_dir, design_data['mesh_file'])
            if not os.path.exists(mesh_file_path):
                print(f"❌ {design_file}: 메시 파일 없음 {design_data['mesh_file']}")
                missing_meshes += 1
                continue
            
            # 액션 값 유효성 확인
            action_values = design_data.get('action', [])
            if not action_values or not isinstance(action_values, list):
                print(f"❌ {design_file}: 유효하지 않은 액션 값")
                invalid_designs += 1
                continue
            
            valid_designs += 1
            
        except Exception as e:
            print(f"❌ {design_file}: 검증 중 오류 {e}")
            invalid_designs += 1
    
    print(f"\n📊 디자인 검증 결과:")
    print(f"✅ 유효한 디자인: {valid_designs}개")
    print(f"❌ 유효하지 않은 디자인: {invalid_designs}개") 
    print(f"📂 메시 파일 누락: {missing_meshes}개")
    
    return valid_designs, invalid_designs, missing_meshes

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 쌍대비교용 디자인 재생성 및 메시 생성')
    parser.add_argument('--session-dir', type=str, default=None,
                        help='RLHF 분석 세션 디렉토리 경로 (지정하지 않으면 최신 디렉토리 사용)')
    parser.add_argument('--action-port', type=int, default=5556,
                        help='ZMQ 액션 전송 포트 (기본값: 5556)')
    parser.add_argument('--mesh-port', type=int, default=5558,
                        help='ZMQ 메시 내보내기 포트 (기본값: 5558)')
    parser.add_argument('--feedback-session', type=int, default=None,
                        help='인간 피드백 세션 번호 (지정하지 않으면 입력 요청)')
    parser.add_argument('--design-types', nargs='+', default=['top', 'diverse'],
                        choices=['top', 'diverse'],
                        help='재생성할 디자인 타입 (기본값: top diverse)')
    parser.add_argument('--max-per-type', type=int, default=10,
                        help='타입별 최대 재생성 개수 (기본값: 10)')
    parser.add_argument('--validate', action='store_true',
                        help='생성된 디자인 유효성 검증 수행')
    parser.add_argument('--create-index', action='store_true',
                        help='웹 인터페이스용 디자인 인덱스 생성')
    
    args = parser.parse_args()
    
    # 피드백 세션 번호 확인
    feedback_session = args.feedback_session
    if feedback_session is None:
        while True:
            try:
                feedback_session = int(input("🔢 인간 피드백 세션 번호를 입력하세요 (1부터 시작): "))
                if feedback_session > 0:
                    break
                print("❌ 세션 번호는 1 이상의 정수여야 합니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                return 1
    
    print(f"\n🎯 RLHF 쌍대비교 세션 #{feedback_session} 시작")
    print(f"🔧 설정:")
    print(f"   - 액션 포트: {args.action_port}")
    print(f"   - 메시 포트: {args.mesh_port}")
    print(f"   - 디자인 타입: {', '.join(args.design_types)}")
    print(f"   - 타입별 최대: {args.max_per_type}개")
    
    # 기준 데이터 로드
    print("\n📂 RLHF 기준 데이터 로드 중...")
    reference_data = load_reference_data(args.session_dir)
    
    if reference_data is None:
        print("❌ 기준 데이터를 로드할 수 없습니다.")
        print("💡 먼저 analyze_integrated_data.py를 실행하여 기준 데이터를 생성하세요.")
        return 1
    
    # 메타데이터 출력
    metadata = reference_data.get("metadata", {})
    print(f"\n📊 기준 데이터 정보:")
    print(f"   - 총 샘플: {metadata.get('total_samples', 'N/A')}개")
    print(f"   - 상태 차원: {metadata.get('state_dimensions', 'N/A')}개")
    print(f"   - 상태 레이블: {', '.join(metadata.get('state_labels', []))}")
    print(f"   - 최고 성능 디자인: {len(reference_data.get('top_designs', []))}개")
    print(f"   - 다양한 디자인: {len(reference_data.get('diverse_designs', []))}개")
    print(f"   - 생성 시간: {metadata.get('generated_at', 'N/A')}")
    
    # 디자인 재생성 실행
    print(f"\n🚀 디자인 재생성 시작...")
    success = regenerate_comparison_designs(
        reference_data, 
        design_types=args.design_types,
        session_id=feedback_session,
        max_designs_per_type=args.max_per_type
    )
    
    if not success:
        print("❌ 디자인 재생성에 실패했습니다.")
        return 1
    
    # 디렉토리 경로
    designs_dir = os.path.join(project_root, "data", "designs")
    meshes_dir = os.path.join(project_root, "data", "meshes")
    
    # 유효성 검증
    if args.validate:
        valid_count, invalid_count, missing_count = validate_generated_designs(designs_dir, meshes_dir)
        
        if invalid_count > 0 or missing_count > 0:
            print(f"⚠️ 일부 디자인에 문제가 있습니다. 웹 인터페이스에서 오류가 발생할 수 있습니다.")
    
    # 웹 인터페이스 인덱스 생성
    if args.create_index:
        index_file = create_web_interface_index(designs_dir)
        if index_file:
            print(f"✅ 웹 인터페이스 준비 완료: {index_file}")
    
    print(f"\n🎉 RLHF 쌍대비교용 디자인 재생성 완료!")
    print(f"📁 결과 위치:")
    print(f"   - 디자인 메타데이터: {designs_dir}")
    print(f"   - 3D 메시 파일: {meshes_dir}")
    print(f"\n🔄 다음 단계:")
    print(f"   1. Flask 서버 실행: cd server && python app.py")
    print(f"   2. 웹 브라우저에서 http://localhost:5000 접속")
    print(f"   3. 쌍대비교 피드백 수집 시작")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())