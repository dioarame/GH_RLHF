#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 간단한 Flask 서버
단계별 구축을 위한 최소 기능 구현
"""

import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DESIGNS_DIR = os.path.join(DATA_DIR, 'designs')
MESHES_DIR = os.path.join(DATA_DIR, 'meshes')
FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')
ENVIRONMENT_DIR = os.path.join(DATA_DIR, 'environment')

# 디렉토리 생성
for directory in [DATA_DIR, DESIGNS_DIR, MESHES_DIR, FEEDBACK_DIR, ENVIRONMENT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask 앱 생성
app = Flask(__name__)

# 세션 상태 (메모리에 임시 저장)
current_session = {
    'id': f"session_{int(time.time())}",
    'total_comparisons': 0,
    'target_comparisons': 100,
    'feedback_data': []
}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('simple_index.html')

@app.route('/api/comparison/next', methods=['POST'])
def get_next_comparison():
    """다음 비교 쌍 가져오기 (실제 디자인 데이터 기반)"""
    try:
        # 생성된 디자인 파일들 가져오기
        design_files = []
        if os.path.exists(DESIGNS_DIR):
            for filename in os.listdir(DESIGNS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(DESIGNS_DIR, filename)
                    with open(filepath, 'r') as f:
                        design_data = json.load(f)
                        design_files.append(design_data)
        
        # 디자인이 충분하지 않으면 목 데이터 사용
        if len(design_files) < 2:
            print("디자인 파일이 부족합니다. 목 데이터를 사용합니다.")
            # 목 데이터로 두 개의 가상 디자인 생성
            design_a = {
                'id': f"mock_design_a_{int(time.time())}",
                'state': [0.45, 3.2, 85000, 0.78],  # BCR, FAR, Sunlight, SVR
                'action': [0.5, 0.6, 0.7, 0.8],
                'reward': 0.75
            }
            
            design_b = {
                'id': f"mock_design_b_{int(time.time())}",
                'state': [0.62, 4.1, 72000, 0.92],
                'action': [0.6, 0.7, 0.5, 0.9],
                'reward': 0.68
            }
        else:
            # 실제 디자인에서 랜덤하게 2개 선택
            import random
            selected_designs = random.sample(design_files, 2)
            
            design_a = {
                'id': selected_designs[0]['id'],
                'state': selected_designs[0]['state'],
                'action': selected_designs[0]['action'],
                'reward': selected_designs[0]['reward']
            }
            
            design_b = {
                'id': selected_designs[1]['id'],
                'state': selected_designs[1]['state'],
                'action': selected_designs[1]['action'],
                'reward': selected_designs[1]['reward']
            }
        
        return jsonify({
            'status': 'success',
            'design_a': design_a,
            'design_b': design_b
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/feedback/comparison', methods=['POST'])
def submit_comparison_feedback():
    """비교 피드백 제출"""
    try:
        data = request.get_json()
        
        # 피드백 데이터 저장
        feedback = {
            'id': f"feedback_{int(time.time())}",
            'session_id': data.get('session_id'),
            'design_a_id': data.get('design_a_id'),
            'design_b_id': data.get('design_b_id'),
            'selected_design': data.get('selected_design'),
            'timestamp': datetime.now().isoformat(),
            'design_a_state': data.get('design_a_state'),
            'design_b_state': data.get('design_b_state')
        }
        
        # 메모리에 저장
        current_session['feedback_data'].append(feedback)
        current_session['total_comparisons'] += 1
        
        # 파일에도 저장
        feedback_file = os.path.join(FEEDBACK_DIR, f"{feedback['id']}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        return jsonify({'status': 'success', 'feedback_id': feedback['id']})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/designs', methods=['GET'])
def get_designs():
    """디자인 목록 가져오기"""
    try:
        designs = []
        
        # designs 디렉토리에서 JSON 파일들 읽기
        if os.path.exists(DESIGNS_DIR):
            for filename in os.listdir(DESIGNS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(DESIGNS_DIR, filename)
                    with open(filepath, 'r') as f:
                        design_data = json.load(f)
                        designs.append(design_data)
        
        return jsonify({'status': 'success', 'designs': designs})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/mesh/<design_id>', methods=['GET'])
def get_mesh(design_id):
    """메시 데이터 가져오기"""
    try:
        # 디자인 정보 찾기
        design_file = os.path.join(DESIGNS_DIR, f"{design_id}.json")
        if not os.path.exists(design_file):
            return jsonify({'status': 'error', 'message': 'Design not found'}), 404
        
        with open(design_file, 'r') as f:
            design_data = json.load(f)
        
        # 메시 파일 찾기
        if 'mesh_file' in design_data:
            mesh_file = os.path.join(MESHES_DIR, design_data['mesh_file'])
            if os.path.exists(mesh_file):
                with open(mesh_file, 'r') as f:
                    mesh_data = json.load(f)
                return jsonify({'status': 'success', 'mesh': mesh_data})
        
        # 메시 파일이 없으면 기본 큐브 데이터 반환
        default_mesh = {
            'format': 'json',
            'meshes': [{
                'vertices': [
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ],
                'faces': [
                    [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                    [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
                ]
            }]
        }
        
        return jsonify({'status': 'success', 'mesh': default_mesh})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/session/stats', methods=['GET'])
def get_session_stats():
    """현재 세션 통계"""
    return jsonify({
        'status': 'success',
        'session': current_session
    })

@app.route('/api/designs/stats', methods=['GET'])
def get_design_stats():
    """디자인 통계 정보 (카테고리별 개수, 가능한 비교 쌍 수)"""
    try:
        top_designs = 0
        random_designs = 0
        other_designs = 0
        
        # designs 디렉토리에서 파일들 분석
        if os.path.exists(DESIGNS_DIR):
            for filename in os.listdir(DESIGNS_DIR):
                if filename.endswith('.json'):
                    if 'top' in filename.lower():
                        top_designs += 1
                    elif 'random' in filename.lower():
                        random_designs += 1
                    else:
                        other_designs += 1
        
        total_designs = top_designs + random_designs + other_designs
        
        # 가능한 비교 쌍 수 계산 (조합: nC2 = n*(n-1)/2)
        max_comparisons = total_designs * (total_designs - 1) // 2 if total_designs > 1 else 0
        
        return jsonify({
            'status': 'success',
            'stats': {
                'top_designs': top_designs,
                'random_designs': random_designs,
                'other_designs': other_designs,
                'total_designs': total_designs,
                'max_comparisons': max_comparisons
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 정적 파일 제공
@app.route('/static/<path:filename>')
def static_files(filename):
    """정적 파일 제공"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, filename)

# 데이터 파일 제공
@app.route('/data/<path:filename>')
def data_files(filename):
    """데이터 파일 제공 (환경 데이터 포함)"""
    return send_from_directory(DATA_DIR, filename)

if __name__ == '__main__':
    print("RLHF 간단한 서버 시작...")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"접속 URL: http://localhost:5000")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"서버 시작 오류: {e}")
        print("포트 5000이 사용 중일 수 있습니다. 포트 5001로 시도 중...")
        app.run(host='127.0.0.1', port=5001, debug=True)