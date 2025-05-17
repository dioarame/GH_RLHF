# ppo_train.py (ZMQ 연결 안정성 개선 버전)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# env_simple.py 파일이 같은 디렉토리에 있다고 가정
from env_simple import SimpleGrasshopperEnv
import torch
import os
import time
import sys
import requests
import json
import zmq
import argparse
import traceback
import csv
import numpy as np
import signal
import atexit

# +++ 데이터 로깅 콜백 클래스 정의 +++
class DataLoggingCallback(BaseCallback):
    """
    매 스텝마다 환경 데이터를 CSV 파일에 로깅하는 콜백.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None
        self.log_header_written = False
        self.episode_num = 0
        print(f"📊 Data logging enabled. Saving logs to: {self.log_path}")

    def _on_training_start(self) -> None:
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")
            self.csv_file = open(self.log_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
        except Exception as e:
            print(f"❌ Error opening log file {self.log_path}: {e}")
            self.csv_file = None
            self.csv_writer = None

    def _write_header(self, obs_shape, action_shape):
        if self.csv_writer and not self.log_header_written:
            header = ['episode', 'timestep']
            if obs_shape and len(obs_shape) >= 1 and obs_shape[0] > 0:
                header.extend([f'obs_{i}' for i in range(obs_shape[0])])
            else: header.append('observation')
            if action_shape and len(action_shape) >= 1 and action_shape[0] > 0 :
                header.extend([f'action_{i}' for i in range(action_shape[0])])
            else: header.append('action')
            header.extend(['reward', 'done'])
            try:
                self.csv_writer.writerow(header)
                self.log_header_written = True
                print("CSV log header written.")
            except Exception as e:
                print(f"Error writing CSV header: {e}")

    def _on_step(self) -> bool:
        if self.csv_writer is None: return True

        if not self.log_header_written:
            try:
                obs_space = self.training_env.observation_space
                act_space = self.training_env.action_space
                if obs_space and act_space:
                     self._write_header(obs_space.shape, act_space.shape)
                else: print("Warning: Could not get observation/action space shape for logging header.")
            except Exception as e: print(f"Error accessing env spaces for CSV header: {e}")

        if not self.log_header_written or self.csv_writer is None: return True

        try:
            obs = self.locals.get('new_obs', self.locals.get('obs'))
            reward = self.locals.get('rewards')[0]
            done = self.locals.get('dones')[0]
            action = self.locals.get('actions')

            if obs is None or action is None:
                 print("Warning: Observation or Action data not found in locals for logging.")
                 return True

            # 데이터 처리 및 변환 (Numpy -> List, float/int)
            obs_data = obs[0].flatten().tolist() if isinstance(obs, np.ndarray) and obs.ndim > 1 else [obs] if isinstance(obs, (int, float, bool)) else obs.flatten().tolist() if isinstance(obs, np.ndarray) else obs
            action_data = action[0].flatten().tolist() if isinstance(action, np.ndarray) and action.ndim > 1 else [action] if isinstance(action, (int, float, bool)) else action.flatten().tolist() if isinstance(action, np.ndarray) else action

            def try_convert(item):
                if isinstance(item, (np.float32, np.float64)): return float(item)
                if isinstance(item, (np.int32, np.int64)): return int(item)
                return item
            obs_data = [try_convert(o) for o in obs_data]
            action_data = [try_convert(a) for a in action_data]

            if done: self.episode_num += 1
            row_data = [self.episode_num, self.num_timesteps] + obs_data + action_data + [float(reward), int(done)]
            self.csv_writer.writerow(row_data)
            
            # 주기적으로 파일 버퍼 플러시 (추가됨)
            if self.num_timesteps % 50 == 0 and self.csv_file and not self.csv_file.closed:
                try:
                    self.csv_file.flush()
                except Exception as e:
                    print(f"Error flushing log file: {e}")
            
        except Exception as e:
             print(f"Error logging data at step {self.num_timesteps}: {e}")
        return True

    def _on_training_end(self) -> None:
        if self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.close()
                print("CSV log file closed on training end.")
            except Exception as e: print(f"Error closing log file on training end: {e}")
            self.csv_file = None
            self.csv_writer = None
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 명령줄 파라미터 파싱 함수
def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training for Grasshopper with ZMQ communication')
    parser.add_argument('--gh-path', type=str, default=r"C:/Users/valen/Desktop/Dev/AS_B.gh",
                        help='Grasshopper 정의 파일 경로')
    parser.add_argument('--compute-url', type=str, default="http://localhost:6500/grasshopper",
                        help='Rhino.Compute 서버 URL')
    parser.add_argument('--port', type=int, default=5556,
                        help='ZMQ 서버 포트 (Python Bind)')
    parser.add_argument('--steps', type=int, default=100,
                        help='학습할 총 타임스텝 수 (기본값: 100)')
    parser.add_argument('--fps', type=float, default=0.15,  # FPS 더 낮게 설정 (4초 처리 시간 고려)
                        help='초당 처리할 최대 스텝 수 (기본값: 0.15)')
    parser.add_argument('--test-only', action='store_true',
                        help='학습 없이 ZMQ 통신 테스트만 수행')
    parser.add_argument('--log-path', type=str, default=os.path.join("logs", f"ppo_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"),
                        help='데이터 로그 CSV 파일 경로')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='학습 디바이스 (auto=자동감지, cuda=GPU, cpu=CPU) (기본값: auto)')
    parser.add_argument('--checkpoint-freq', type=int, default=50,  # 더 자주 체크포인트 저장
                        help='체크포인트 저장 주기 (스텝) - 0이면 비활성화 (기본값: 50)')
    parser.add_argument('--computation-delay', type=float, default=4.0,  # 계산 지연 시간 기본값 4초
                        help='Grasshopper 계산 대기 시간 (초) (기본값: 4.0)')
                        
    # 보상 함수 매개변수 추가
    parser.add_argument('--bcr-limit', type=float, default=0.6,
                        help='BCR 법적 한도 (기본값: 0.6, 60%)')
    parser.add_argument('--far-limit', type=float, default=4.0,
                        help='FAR 법적 한도 (기본값: 4.0, 400%)')
    parser.add_argument('--bcr-weight', type=float, default=1.0,
                        help='BCR 보상 가중치 (기본값: 1.0)')
    parser.add_argument('--far-weight', type=float, default=1.5, 
                        help='FAR 보상 가중치 (기본값: 1.5)')
    parser.add_argument('--sunlight-weight', type=float, default=2.0,
                        help='일조량 보상 가중치 (기본값: 2.0)')
    parser.add_argument('--other-weight', type=float, default=0.5,
                        help='기타 지표 보상 가중치 (기본값: 0.5)')
                        
    return parser.parse_args()

# Rhino.Compute 서버 상태 확인 함수
def check_compute_server(url):
    try:
        base_url = url.split('/grasshopper')[0]
        r = requests.get(f"{base_url}/version", timeout=5)
        r.raise_for_status()
        print(f"✅ Rhino.Compute 서버가 작동 중입니다. 버전: {r.json()}")
        return True
    except Exception as e:
        print(f"❌ Rhino.Compute 서버 연결 실패: {e}")
        return False

# TCP 포트 가용성 확인 함수 (추가됨)
def is_port_available(port):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind(("127.0.0.1", port))
        result = True
    except:
        print(f"⚠️ 포트 {port}가 이미 사용 중입니다.")
        result = False
    finally:
        sock.close()
    return result

# ZMQ PUSH 통신 테스트 함수 (개선됨)
def test_zmq_push_connection(port, test_actions=None):
    """ZMQ PUSH 소켓 연결 및 메시지 전송 테스트"""
    if test_actions is None:
        test_actions = [[0.0]]
    
    if not is_port_available(port):
        print(f"⚠️ 포트 {port}가 이미 사용 중입니다. 테스트를 건너뜁니다.")
        return False
        
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    connect_address = f"tcp://localhost:{port}"
    result = False
    
    try:
        # 소켓 옵션 설정
        socket.setsockopt(zmq.LINGER, 200)
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
        
        print(f"⚠️ ZMQ 연결 테스트 (PUSH): {connect_address}에 바인딩 시도 중...")
        socket.bind(connect_address)
        
        for i, action in enumerate(test_actions):
            data = json.dumps(action)
            print(f"📤 테스트 {i+1}/{len(test_actions)}: 값 {action} 전송 중...")
            socket.send_string(data)
            print(f"   전송된 JSON: {data}")
            time.sleep(1) # GH 처리 시간
        print("✅ ZMQ 테스트 메시지 전송 완료! Grasshopper에서 슬라이더가 움직이는지 확인하세요.")
        result = True
    except Exception as e:
        print(f"❌ ZMQ 연결 테스트 (PUSH) 실패: {e}")
        traceback.print_exc()
        result = False
    finally:
        try:
            time.sleep(0.2)
            socket.close(linger=0)
            context.term()
        except Exception as e_close:
            print(f"Error closing ZMQ resources in test: {e_close}")
    return result

# 학습 속도 조절 및 체크포인트 콜백 클래스 (확장됨)
class TrainingCallback(BaseCallback):
    # ppo_train.py의 TrainingCallback 클래스에 추가
    def __init__(self, limit_fps, checkpoint_freq=0, checkpoint_prefix="ppo_checkpoint", random_actions_steps=1000):
        super().__init__(verbose=0)
        self.limit_fps = min(limit_fps, 5.0)  # 최대 5 FPS로 제한
        self.min_interval = 1.0 / self.limit_fps if self.limit_fps > 0 else 0
        self.last_time = time.time()
        self.start_time = time.time()
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_count = 0
        
        # 무작위 액션 관련 변수 추가
        self.random_actions_steps = random_actions_steps  # 무작위 액션 사용 스텝 수
        self.random_actions_count = 0  # 현재까지 사용한 무작위 액션 수
        
        print(f"  속도 제한: {self.limit_fps} steps/sec (최소 간격: {self.min_interval:.3f}초)")
        if checkpoint_freq > 0:
            print(f"  체크포인트 간격: {checkpoint_freq} 스텝마다 저장")
        print(f"  초기 무작위 액션: {random_actions_steps} 스텝")

    def _on_step(self) -> bool:
        current_time = time.time()
        elapsed = current_time - self.last_time
        total_elapsed = current_time - self.start_time
        
        # 무작위 액션 사용 여부 확인
        use_random = self.random_actions_count < self.random_actions_steps
        
        if use_random:
            # 모델의 predict 메서드를 재정의하여 무작위 액션 사용
            original_predict = self.model.predict
            
            def random_predict(observation, state=None, episode_start=None, deterministic=False):
                action = self.model.env.action_space.sample()
                return action, state
            
            self.model.predict = random_predict
            self.random_actions_count += 1
            
            # 로깅
            if self.random_actions_count % 100 == 0 or self.random_actions_count == 1:
                steps_per_sec = self.num_timesteps / total_elapsed if total_elapsed > 0 else 0
                print(f"무작위 액션 사용 중: {self.random_actions_count}/{self.random_actions_steps} ({steps_per_sec:.2f} steps/sec)")
        
        # 로깅
        if self.num_timesteps > 0 and self.num_timesteps % 10 == 0:
            steps_per_sec = self.num_timesteps / total_elapsed if total_elapsed > 0 else 0
            print(f"🔄 스텝 {self.num_timesteps}/{self.locals.get('total_timesteps', '?')}: {steps_per_sec:.2f} steps/sec")

        # 체크포인트 저장
        if self.checkpoint_freq > 0 and self.num_timesteps % self.checkpoint_freq == 0:
            self.checkpoint_count += 1
            checkpoint_path = f"{self.checkpoint_prefix}_{self.num_timesteps}"
            self.model.save(checkpoint_path)
            print(f"💾 체크포인트 저장됨 (#{self.checkpoint_count}): {checkpoint_path}")

        # FPS 제한
        if self.min_interval > 0:
            wait_time = self.min_interval - elapsed
            if wait_time > 0: time.sleep(wait_time)
        self.last_time = time.time()
        
        # 무작위 액션 모드였으면 원래 predict 메서드로 복원
        if use_random:
            self.model.predict = original_predict
        
        return True

# 시그널 핸들러 설정 (추가됨)
def setup_signal_handlers(env, model):
    """Ctrl+C와 같은 시그널 처리 설정"""
    def signal_handler(sig, frame):
        print("\n🛑 시그널 수신. 정리 중...")
        if model:
            try:
                interrupted_model_path = f"ppo_model_interrupted_{time.strftime('%Y%m%d_%H%M%S')}"
                model.save(interrupted_model_path)
                print(f"💾 중단된 모델 저장 완료: {interrupted_model_path}.zip")
            except Exception as e:
                print(f"모델 저장 중 오류: {e}")
        
        if env:
            try:
                env.close()
                print("환경 리소스 정리 완료.")
            except Exception as e:
                print(f"환경 정리 중 오류: {e}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Windows에서 Ctrl+Break 처리
    try:
        signal.signal(signal.SIGBREAK, signal_handler)
    except:
        pass
    
    print("📡 시그널 핸들러 설정 완료 (Ctrl+C로 안전하게 종료 가능)")

# 메인 함수
def main():
    print("=" * 80)
    print("PPO Training for Grasshopper with ZMQ Communication")
    print("=" * 80)

    args = parse_args()
    COMPUTE_URL = args.compute_url
    GH_DEFINITION_PATH = args.gh_path
    ZMQ_SERVER_PORT = args.port
    TOTAL_TIMESTEPS = args.steps
    STEPS_PER_SECOND = args.fps
    LOG_FILE_PATH = args.log_path
    DEVICE = args.device
    CHECKPOINT_FREQ = args.checkpoint_freq

    # 실제 학습 디바이스 결정
    if DEVICE == 'auto':
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env = None
    model = None
    logging_callback = None

    try:
        # 1. 서버 상태 확인
        print("\n[1/5] Rhino.Compute 서버 상태 확인 중...")
        if not check_compute_server(COMPUTE_URL):
            sys.exit(1)

        # 2. 테스트 모드 확인
        if args.test_only:
            print("\n[TEST MODE] ZMQ 통신 테스트만 수행합니다...")
            print(f"  ZMQ 포트: {ZMQ_SERVER_PORT}")
            
            # 테스트 액션 설정
            test_actions = [[-5.0, -5.0, -5.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
            test_result = test_zmq_push_connection(ZMQ_SERVER_PORT, test_actions=test_actions)
                
            print(f"ZMQ 테스트 결과: {test_result}")
            sys.exit(0)

        # 3. 환경 생성
        print("\n[2/5] Grasshopper RL 환경 생성 중...")
        try:
            print(f"  Grasshopper 파일: {GH_DEFINITION_PATH}")
            if not os.path.exists(GH_DEFINITION_PATH):
                raise FileNotFoundError(f"Grasshopper 파일을 찾을 수 없습니다: {GH_DEFINITION_PATH}")

            print(f"  ZMQ 서버 포트: {ZMQ_SERVER_PORT}")
            print(f"  Grasshopper 계산 대기 시간: {args.computation_delay}초")
            print(f"  BCR 한도: {args.bcr_limit*100:.1f}%, 가중치: {args.bcr_weight}")
            print(f"  FAR 한도: {args.far_limit*100:.1f}%, 가중치: {args.far_weight}")
            print(f"  일조량 가중치: {args.sunlight_weight}")

            env = SimpleGrasshopperEnv(
                compute_url=COMPUTE_URL,
                gh_definition_path=GH_DEFINITION_PATH,
                state_output_param_name="DesignState",  # Grasshopper에서 사용할 상태 파라미터 이름
                # reward_output_param_name은 제거됨 - 내부 보상 함수 사용
                slider_info_param_name="SliderInfo",
                max_episode_steps=100,
                action_push_port=ZMQ_SERVER_PORT,
                use_push_mode=True,
                computation_delay=args.computation_delay,
                # 보상 함수 매개변수 전달
                bcr_limit=args.bcr_limit,
                far_limit=args.far_limit,
                bcr_weight=args.bcr_weight,
                far_weight=args.far_weight,
                sunlight_weight=args.sunlight_weight,
                other_weight=args.other_weight
            )


            print("✅ 환경이 성공적으로 생성되었습니다.")
            print(f"   - Action Space: {env.action_space}")
            print(f"   - Observation Space: {env.observation_space}")

        except FileNotFoundError as fnf_err:
            print(f"❌ 환경 생성 실패: {fnf_err}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 환경 생성 중 예외 발생: {e}")
            traceback.print_exc()
            sys.exit(1)

        # 4. ZMQ 통신 테스트는 이제 env 초기화 과정에서 수행됨
        print("\n[3/5] ZMQ 통신이 초기화되었습니다.")

        # 5. 로깅 콜백 생성
        print("\n[+] 데이터 로깅 콜백 설정 중...")
        try:
            logging_callback = DataLoggingCallback(log_path=LOG_FILE_PATH)
        except Exception as e:
            print(f"❌ 로깅 콜백 생성 실패: {e}")
            logging_callback = None
        
        # PPO 환경 생성 전, 강화된 무작위 탐색 단계 추가
        print("\n[+] 강화된 무작위 탐색 단계 시작...")
        for ep in range(20):  # 20 에피소드 동안
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 50:  # 각 에피소드 최대 50 스텝
                # 매우 극단적인 값을 포함하는 무작위 액션 생성
                action = np.zeros(env.action_space.shape)
                for i in range(len(action)):
                    # 1/3 확률로 경계값, 2/3 확률로 균등 분포
                    if np.random.random() < 0.33:
                        # 경계값 (-10, -5, 0, 5, 10 중 하나)
                        action[i] = np.random.choice([-10, -5, 0, 5, 10])
                    else:
                        # 전체 범위에서 균등 분포
                        action[i] = np.random.uniform(
                            env.action_space.low[i], 
                            env.action_space.high[i]
                        )
                    
                    # 정수 슬라이더인 경우 반올림
                    if env.slider_roundings[i] == 1.0:
                        action[i] = int(round(action[i]))
                
                # 100 스텝마다 로깅
                if steps % 5 == 0:
                    print(f"  에피소드 {ep+1}, 스텝 {steps+1}: 액션 = {action}")
                
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                if done:
                    break
            
            print(f"  에피소드 {ep+1} 완료: {steps} 스텝")

        print("[+] 강화된 무작위 탐색 완료. PPO 학습 시작.")

        # 6. PPO 모델 생성
        print("\n[4/5] PPO 모델 생성 중...")
        try:
            print(f"  사용 디바이스: {DEVICE}")

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=DEVICE,
                learning_rate=3e-4,
                n_steps=256,  # 더 작은 배치 크기 사용
                batch_size=64,
                n_epochs=5,   # 더 적은 epoch
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.1,  # 더 큰 엔트로피 계수 (탐색 증가)
                # tensorboard_log="./ppo_gh_tensorboard/"
            )
            print("✅ PPO 모델이 성공적으로 생성되었습니다.")

        except Exception as e:
            print(f"❌ 모델 생성 중 오류 발생: {e}")
            traceback.print_exc()
            if env: env.close()
            sys.exit(1)
            
        # 시그널 핸들러 설정
        setup_signal_handlers(env, model)

        # 7. 학습 시작
        print("\n[5/5] PPO 모델 학습 시작...")
        try:
            print(f"  총 타임스텝: {TOTAL_TIMESTEPS}")
            print(f"  학습 속도 제한: {STEPS_PER_SECOND} steps/sec")
            if CHECKPOINT_FREQ > 0:
                print(f"  체크포인트 빈도: {CHECKPOINT_FREQ} 스텝마다")

            # 콜백 리스트 생성 부분 수정
            callbacks_list = [TrainingCallback(STEPS_PER_SECOND, 
                                            checkpoint_freq=CHECKPOINT_FREQ, 
                                            checkpoint_prefix=f"ppo_grasshopper_checkpoint_{time.strftime('%Y%m%d_%H%M%S')}",
                                            random_actions_steps=2000)]  # 처음 2000 스텝은 완전히 무작위 액션 사용

            if logging_callback:
                callbacks_list.append(logging_callback)
                print("  데이터 로깅 콜백 활성화됨.")
            else:
                print("  데이터 로깅 콜백 비활성화됨.")

            start_time = time.time()
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callbacks_list,
                log_interval=1
            )
            end_time = time.time()

            print("-" * 50)
            print(f"✅ 학습 완료! (소요 시간: {end_time - start_time:.2f} 초)")

            model_path = f"ppo_grasshopper_model_{time.strftime('%Y%m%d_%H%M%S')}"
            model.save(model_path)
            print(f"💾 모델 저장 완료: {model_path}.zip")

        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 학습이 중단되었습니다.")
            if model:
                interrupted_model_path = f"ppo_grasshopper_model_interrupted_{time.strftime('%Y%m%d_%H%M%S')}"
                model.save(interrupted_model_path)
                print(f"💾 중단된 모델 저장 완료: {interrupted_model_path}.zip")
        except Exception as e:
            print(f"\n❌ 학습 중 오류 발생: {e}")
            traceback.print_exc()

    finally:
        # 로깅 파일 닫기
        if logging_callback and hasattr(logging_callback, 'csv_file') and logging_callback.csv_file and not logging_callback.csv_file.closed:
            try:
                logging_callback.csv_file.close()
                print("Ensured log file is closed in finally block.")
            except Exception as e_close_log:
                print(f"Error closing log file in finally block: {e_close_log}")

        print("\n🧹 환경 리소스를 정리합니다...")
        if env:
            env.close()
        print("💯 작업 완료. 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()