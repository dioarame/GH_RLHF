# ppo_train.py
from stable_baselines3 import PPO
import torch
import os
import time
import sys
import traceback
import numpy as np

# 모듈화된 구성요소 임포트
from config import Config, parse_args
from grasshopper_env import SimpleGrasshopperEnv
from callbacks import DataLoggingCallback, FPSLimiter
from testing import test_zmq_push_connection, test_zmq_req_connection, check_compute_server

# ppo_train.py (계속)
def main():
    print("=" * 80)
    print("PPO Training for Grasshopper with ZMQ Communication")
    print("=" * 80)

    # 설정 로드
    args = parse_args()
    config = Config(args)

    env = None
    model = None
    logging_callback = None

    try:
        # 1. 서버 상태 확인
        print("\n[1/5] Rhino.Compute 서버 상태 확인 중...")
        if not check_compute_server(config.compute_url):
            sys.exit(1)

        # 2. 테스트 모드 확인
        if config.test_only:
            print("\n[TEST MODE] ZMQ 통신 테스트만 수행합니다...")
            print(f"  ZMQ 통신 모드: {'PUSH' if config.use_push_mode else 'REP'}")
            print(f"  ZMQ 포트: {config.zmq_port}")
            
            # 테스트 액션 설정
            test_actions = [[-5.0, -5.0, -5.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
            
            # 통신 모드에 따라 다른 테스트 수행
            if config.use_push_mode:
                test_result = test_zmq_push_connection(config.zmq_port, test_actions=test_actions)
            else:
                test_result = test_zmq_req_connection(config.zmq_port, test_actions=test_actions)
                
            print(f"ZMQ 테스트 결과: {test_result}")
            sys.exit(0)

        # 3. 환경 생성
        print("\n[2/5] Grasshopper RL 환경 생성 중...")
        try:
            print(f"  Grasshopper 파일: {config.gh_path}")
            if not os.path.exists(config.gh_path):
                raise FileNotFoundError(f"Grasshopper 파일을 찾을 수 없습니다: {config.gh_path}")

            print(f"  ZMQ 서버 포트: {config.zmq_port}")
            print(f"  ZMQ 통신 모드: {'PUSH' if config.use_push_mode else 'REP'}")

            env = SimpleGrasshopperEnv(
                compute_url=config.compute_url,
                gh_definition_path=config.gh_path,
                state_output_param_name=config.state_param_name,
                reward_output_param_name=config.reward_param_name,
                slider_info_param_name=config.slider_info_param_name,
                max_episode_steps=config.max_episode_steps,
                action_push_port=config.zmq_port,
                use_push_mode=config.use_push_mode
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

        # 4. ZMQ 통신 테스트
        print("\n[3/5] ZMQ 통신 테스트 중...")
        try:
            sample_action = env.action_space.sample()
            test_actions = [sample_action.tolist()]
        except Exception as sample_err:
            print(f"   Action space 샘플링 실패 ({sample_err}), 기본 값으로 테스트합니다.")
            test_actions = [[0.0, 0.0, 0.0]]

        print(f"   ZMQ 테스트 액션: {test_actions}")
        
        if config.use_push_mode:
            test_result = test_zmq_push_connection(config.zmq_port, test_actions=test_actions)
        else:
            test_result = test_zmq_req_connection(config.zmq_port, test_actions=test_actions)
            
        print(f">>> ZMQ 테스트 결과: {test_result} <<<")

        if not test_result:
            print("⚠️ ZMQ 테스트 실패. Grasshopper 연결 상태 확인 필요.")
            print("   - Grasshopper에서 ZMQ Listener 컴포넌트가 실행 중인지 확인하세요.")
            print(f"   - 'Use PULL' 설정이 {'True' if config.use_push_mode else 'False'}로 설정되어 있는지 확인하세요.")
            print("   - 포트 번호가 일치하는지 확인하세요.")
            
            choice = input("   계속 진행하시겠습니까? (y/n): ").strip().lower()
            if choice != 'y':
                if env: env.close()
                sys.exit(1)
            print("   경고를 무시하고 계속 진행합니다.")

        # 5. 로깅 콜백 생성
        print("\n[+] 데이터 로깅 콜백 설정 중...")
        try:
            logging_callback = DataLoggingCallback(log_path=config.log_path)
        except Exception as e:
            print(f"❌ 로깅 콜백 생성 실패: {e}")
            logging_callback = None

        # 6. PPO 모델 생성
        print("\n[4/5] PPO 모델 생성 중...")
        try:
            # 학습 디바이스 결정
            if config.device == 'auto':
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = config.device
                
            print(f"  사용 디바이스: {device}")

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=device,
                learning_rate=3e-4,
                n_steps=256,  # 더 작은 배치 크기 사용
                batch_size=64,
                n_epochs=5,   # 더 적은 epoch
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # 더 큰 엔트로피 계수 (탐색 증가)
                # tensorboard_log="./ppo_gh_tensorboard/"
            )
            print("✅ PPO 모델이 성공적으로 생성되었습니다.")

        except Exception as e:
            print(f"❌ 모델 생성 중 오류 발생: {e}")
            traceback.print_exc()
            if env: env.close()
            sys.exit(1)

        # 7. 학습 시작
        print("\n[5/5] PPO 모델 학습 시작...")
        try:
            print(f"  총 타임스텝: {config.total_timesteps}")
            print(f"  학습 속도 제한: {config.fps_limit} steps/sec")

            callbacks_list = [FPSLimiter(config.fps_limit)]
            if logging_callback:
                callbacks_list.append(logging_callback)
                print("  데이터 로깅 콜백 활성화됨.")
            else:
                print("  데이터 로깅 콜백 비활성화됨.")

            start_time = time.time()
            model.learn(
                total_timesteps=config.total_timesteps,
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