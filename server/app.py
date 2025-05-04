#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF Grasshopper - Flask 백엔드 서버

이 서버는 Grasshopper에서 추출한 메쉬 데이터를 관리하고
Three.js 기반 인터페이스에 제공합니다.
"""

import os
import json
import time
import uuid
import logging
import zmq
import socket
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, render_template, Blueprint
from werkzeug.utils import secure_filename

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 설정
class Config:
    # 서버 설정
    PORT = 5000
    DEBUG = True
    
    # 데이터 디렉토리
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    MESHES_DIR = os.path.join(DATA_DIR, 'meshes')
    FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')
    DESIGNS_DIR = os.path.join(DATA_DIR, 'designs')
    
    # ZMQ 설정
    ZMQ_MESH_EXPORTER_ADDRESS = "tcp://localhost:5558"
    ZMQ_REQUEST_TIMEOUT = 5000  # 밀리초
    
    # 허용된 확장자
    ALLOWED_EXTENSIONS = {'json', 'gltf', 'glb'}

# 포트 가용성 확인 함수
def is_port_available(port):
    """지정된 포트 사용 가능 여부 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except:
            return False

# ZMQ 포트 가용성 확인 및 조정
zmq_port = 5558
if not is_port_available(zmq_port):
    # 사용 가능한 포트 찾기
    for test_port in range(zmq_port + 1, zmq_port + 50):
        if is_port_available(test_port):
            zmq_port = test_port
            logger.info(f"기본 ZMQ 포트(5558)가 사용 중입니다. 포트 {zmq_port}로 조정합니다.")
            Config.ZMQ_MESH_EXPORTER_ADDRESS = f"tcp://localhost:{zmq_port}"
            break
    else:
        logger.warning("사용 가능한 ZMQ 포트를 찾을 수 없습니다. 기본 포트를 사용합니다.")

# 디렉토리 생성
for directory in [Config.DATA_DIR, Config.MESHES_DIR, Config.FEEDBACK_DIR, Config.DESIGNS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask 앱 생성
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# 에러 처리 데코레이터
def api_error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API 오류: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500
    return decorated_function

# ZMQ 유틸리티 클래스
class ZmqClient:
    def __init__(self, address, timeout=5000):
        self.address = address
        self.timeout = timeout
        self.context = zmq.Context()
    
    def request(self, data):
        """ZMQ REQ 소켓을 통한 요청 전송"""
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)
        
        try:
            # 요청 직렬화 및 전송
            request_json = json.dumps(data)
            socket.send_string(request_json)
            
            # 폴링을 사용한 응답 대기
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            
            if poller.poll(self.timeout):
                response = socket.recv_string()
                return json.loads(response)
            else:
                logger.error("ZMQ 요청 타임아웃")
                return {"status": "error", "message": "ZMQ 요청 타임아웃"}
        
        except Exception as e:
            logger.error(f"ZMQ 통신 오류: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            socket.close()
    
    def close(self):
        """컨텍스트 종료"""
        if self.context:
            self.context.term()

# ZMQ 클라이언트 인스턴스
zmq_client = ZmqClient(Config.ZMQ_MESH_EXPORTER_ADDRESS, Config.ZMQ_REQUEST_TIMEOUT)

# 유틸리티 함수
def allowed_file(filename):
    """파일 확장자 검사"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_timestamp():
    """현재 타임스탬프 생성"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_design_id():
    """고유 디자인 ID 생성"""
    return f"{get_timestamp()}_{uuid.uuid4().hex[:8]}"

# API 블루프린트 생성
designs_bp = Blueprint('designs', __name__, url_prefix='/api/designs')
feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')
zmq_bp = Blueprint('zmq', __name__, url_prefix='/api/zmq')
mesh_bp = Blueprint('mesh', __name__, url_prefix='/api/mesh')

# 디자인 API 라우트
@designs_bp.route('/', methods=['GET'])
@api_error_handler
def get_designs():
    """모든 디자인 목록 가져오기"""
    designs = []
    
    # 디자인 디렉토리에서 모든 JSON 파일 읽기
    for filename in os.listdir(Config.DESIGNS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(Config.DESIGNS_DIR, filename)
            with open(filepath, 'r') as f:
                design_data = json.load(f)
                designs.append(design_data)
    
    return jsonify({"status": "success", "designs": designs})

@designs_bp.route('/<design_id>', methods=['GET'])
@api_error_handler
def get_design(design_id):
    """특정 디자인 정보 가져오기"""
    filepath = os.path.join(Config.DESIGNS_DIR, f"{design_id}.json")
    
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "디자인을 찾을 수 없습니다."}), 404
    
    with open(filepath, 'r') as f:
        design_data = json.load(f)
    
    return jsonify({"status": "success", "design": design_data})

@designs_bp.route('/', methods=['POST'])
@api_error_handler
def create_design():
    """새 디자인 생성"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "JSON 데이터가 필요합니다."}), 400
    
    data = request.get_json()
    
    # 필수 필드 확인
    required_fields = ['state', 'action', 'reward']
    for field in required_fields:
        if field not in data:
            return jsonify({"status": "error", "message": f"'{field}' 필드가 필요합니다."}), 400
    
    # 디자인 ID 생성
    design_id = data.get('id') or create_design_id()
    
    # 디자인 데이터 구성
    design_data = {
        "id": design_id,
        "timestamp": int(time.time() * 1000),
        "created_at": datetime.now().isoformat(),
        "state": data['state'],
        "action": data['action'],
        "reward": data['reward'],
        "metadata": data.get('metadata', {})
    }
    
    # 메쉬 데이터 요청 (ZMQ 통신)
    if 'datasetKey' in data and data['datasetKey']:
        mesh_request = {
            "request": "get_mesh",
            "datasetKey": data['datasetKey']
        }
        
        # ZMQ 요청 전송
        mesh_response = zmq_client.request(mesh_request)
        
        if mesh_response and mesh_response.get('status') != 'error':
            # 메쉬 데이터 저장
            mesh_filename = f"{design_id}.json"
            mesh_filepath = os.path.join(Config.MESHES_DIR, mesh_filename)
            
            with open(mesh_filepath, 'w') as f:
                if isinstance(mesh_response, str):
                    f.write(mesh_response)
                else:
                    json.dump(mesh_response, f)
            
            design_data['mesh_file'] = mesh_filename
    
    # 디자인 데이터 저장
    design_filepath = os.path.join(Config.DESIGNS_DIR, f"{design_id}.json")
    with open(design_filepath, 'w') as f:
        json.dump(design_data, f)
    
    return jsonify({"status": "success", "design_id": design_id, "design": design_data})

@designs_bp.route('/<design_id>', methods=['DELETE'])
@api_error_handler
def delete_design(design_id):
    """디자인 삭제"""
    design_filepath = os.path.join(Config.DESIGNS_DIR, f"{design_id}.json")
    
    if not os.path.exists(design_filepath):
        return jsonify({"status": "error", "message": "디자인을 찾을 수 없습니다."}), 404
    
    # 디자인 JSON 파일 삭제
    os.remove(design_filepath)
    
    # 연관된 메쉬 파일 삭제
    mesh_filepath = os.path.join(Config.MESHES_DIR, f"{design_id}.json")
    if os.path.exists(mesh_filepath):
        os.remove(mesh_filepath)
    
    return jsonify({"status": "success", "message": "디자인이 삭제되었습니다."})

# 메쉬 API 라우트
@mesh_bp.route('/<design_id>', methods=['GET'])
@api_error_handler
def get_mesh(design_id):
    """메쉬 데이터 가져오기"""
    # 디자인 정보 가져오기
    design_filepath = os.path.join(Config.DESIGNS_DIR, f"{design_id}.json")
    
    if not os.path.exists(design_filepath):
        return jsonify({"status": "error", "message": "디자인을 찾을 수 없습니다."}), 404
    
    with open(design_filepath, 'r') as f:
        design_data = json.load(f)
    
    # 메쉬 파일 확인
    if 'mesh_file' not in design_data:
        return jsonify({"status": "error", "message": "이 디자인에 연결된 메쉬 데이터가 없습니다."}), 404
    
    mesh_filepath = os.path.join(Config.MESHES_DIR, design_data['mesh_file'])
    
    if not os.path.exists(mesh_filepath):
        return jsonify({"status": "error", "message": "메쉬 파일을 찾을 수 없습니다."}), 404
    
    # 메쉬 데이터 반환
    with open(mesh_filepath, 'r') as f:
        mesh_data = json.load(f)
    
    return jsonify({"status": "success", "mesh": mesh_data})

# 피드백 API 라우트
@feedback_bp.route('/', methods=['POST'])
@api_error_handler
def submit_feedback():
    """사용자 피드백 제출"""
    if not request.is_json:
        return jsonify({"status": "error", "message": "JSON 데이터가 필요합니다."}), 400
    
    data = request.get_json()
    
    # 필수 필드 확인
    required_fields = ['design_id', 'ratings']
    for field in required_fields:
        if field not in data:
            return jsonify({"status": "error", "message": f"'{field}' 필드가 필요합니다."}), 400
    
    # 피드백 ID 생성
    feedback_id = f"feedback_{get_timestamp()}_{uuid.uuid4().hex[:8]}"
    
    # 피드백 데이터 구성
    feedback_data = {
        "id": feedback_id,
        "design_id": data['design_id'],
        "ratings": data['ratings'],
        "comment": data.get('comment', ''),
        "submitted_at": datetime.now().isoformat()
    }
    
    # 피드백 데이터 저장
    feedback_filepath = os.path.join(Config.FEEDBACK_DIR, f"{feedback_id}.json")
    with open(feedback_filepath, 'w') as f:
        json.dump(feedback_data, f)
    
    return jsonify({"status": "success", "feedback_id": feedback_id})

@feedback_bp.route('/<design_id>', methods=['GET'])
@api_error_handler
def get_design_feedback(design_id):
    """특정 디자인의 피드백 가져오기"""
    feedback_list = []
    
    # 피드백 디렉토리에서 일치하는 피드백 찾기
    for filename in os.listdir(Config.FEEDBACK_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(Config.FEEDBACK_DIR, filename)
            with open(filepath, 'r') as f:
                feedback_data = json.load(f)
                if feedback_data.get('design_id') == design_id:
                    feedback_list.append(feedback_data)
    
    return jsonify({"status": "success", "feedback": feedback_list})

# ZMQ API 라우트
@zmq_bp.route('/ping', methods=['GET'])
@api_error_handler
def ping_mesh_exporter():
    """MeshExporter 연결 상태 확인"""
    response = zmq_client.request({"request": "ping"})
    return jsonify({"status": "success", "response": response})

@zmq_bp.route('/datasets', methods=['GET'])
@api_error_handler
def list_datasets():
    """사용 가능한 데이터셋 목록 가져오기"""
    response = zmq_client.request({"request": "list_datasets"})
    return jsonify({"status": "success", "response": response})

# 기준 데이터 API 라우트
@app.route('/api/reference-data', methods=['GET'])
@api_error_handler
def get_reference_data():
    """RLHF 기준 데이터 가져오기"""
    # 가장 최신 기준 데이터 파일 찾기
    reference_files = []
    for root, _, files in os.walk(Config.DATA_DIR):
        for file in files:
            if file.startswith('rlhf_reference_data') and file.endswith('.json'):
                filepath = os.path.join(root, file)
                reference_files.append((filepath, os.path.getmtime(filepath)))
    
    if not reference_files:
        return jsonify({"status": "error", "message": "기준 데이터 파일을 찾을 수 없습니다."}), 404
    
    # 수정 시간 기준 정렬 (최신 파일이 첫 번째)
    reference_files.sort(key=lambda x: x[1], reverse=True)
    latest_file = reference_files[0][0]
    
    # 파일 데이터 읽기
    with open(latest_file, 'r') as f:
        reference_data = json.load(f)
    
    return jsonify({"status": "success", "reference_data": reference_data})

# 정적 파일 제공
@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """데이터 파일 제공"""
    return send_from_directory(Config.DATA_DIR, filename)

# 메인 페이지
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

# 블루프린트 등록
app.register_blueprint(designs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(zmq_bp)
app.register_blueprint(mesh_bp)

# 프로그램 종료 시 리소스 정리
@app.teardown_appcontext
def cleanup_resources(exception=None):
    """애플리케이션 종료 시 리소스 정리"""
    if zmq_client:
        zmq_client.close()

# 애플리케이션 실행
if __name__ == '__main__':
    try:
        logger.info(f"서버 시작: http://localhost:{Config.PORT}")
        logger.info(f"ZMQ 포트: {zmq_port}")
        app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)
    except KeyboardInterrupt:
        logger.info("서버 종료")
    except Exception as e:
        logger.error(f"서버 실행 중 오류: {e}", exc_info=True)