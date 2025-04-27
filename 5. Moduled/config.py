# config.py
import os
import argparse
from datetime import datetime

def parse_args():
    """명령행 인자 파싱 함수"""
    parser = argparse.ArgumentParser(description='PPO Training for Grasshopper with ZMQ communication')
    parser.add_argument('--gh-path', type=str, default=r"C:/Users/valen/Desktop/Dev/AS_B.gh",
                        help='Grasshopper 정의 파일 경로')
    parser.add_argument('--compute-url', type=str, default="http://localhost:6500/grasshopper",
                        help='Rhino.Compute 서버 URL')
    parser.add_argument('--port', type=int, default=5556,
                        help='ZMQ 서버 포트 (Python Bind)')
    parser.add_argument('--steps', type=int, default=100,
                        help='학습할 총 타임스텝 수 (기본값: 100)')
    parser.add_argument('--fps', type=float, default=2.0,
                        help='초당 처리할 최대 스텝 수 (기본값: 2.0)')
    parser.add_argument('--test-only', action='store_true',
                        help='학습 없이 ZMQ 통신 테스트만 수행')
    parser.add_argument('--log-path', type=str, 
                        default=os.path.join("logs", f"ppo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                        help='데이터 로그 CSV 파일 경로')
    parser.add_argument('--zmq-mode', type=str, choices=['push', 'rep'], default='push',
                        help='ZMQ 통신 모드 (push=PUSH-PULL, rep=REQ-REP) (기본값: push)')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='학습 디바이스 (auto=자동감지, cuda=GPU, cpu=CPU) (기본값: auto)')
    return parser.parse_args()


# 기본 환경 설정
class Config:
    def __init__(self, args=None):
        """설정 초기화 함수"""
        if args is None:
            args = parse_args()
        
        # 경로 설정
        self.gh_path = args.gh_path
        self.compute_url = args.compute_url
        self.log_path = args.log_path
        
        # ZMQ 설정
        self.zmq_port = args.port
        self.use_push_mode = args.zmq_mode == 'push'
        
        # 학습 설정
        self.total_timesteps = args.steps
        self.fps_limit = args.fps
        self.device = args.device
        self.test_only = args.test_only
        
        # 환경 설정
        self.state_param_name = "CurrentState"
        self.reward_param_name = "CalculatedReward"
        self.slider_info_param_name = "SliderInfo"
        self.max_episode_steps = 100