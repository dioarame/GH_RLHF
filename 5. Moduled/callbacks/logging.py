# callbacks/logging.py
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv
import numpy as np
import traceback

class DataLoggingCallback(BaseCallback):
    """매 스텝마다 환경 데이터를 CSV 파일에 로깅하는 콜백."""
    
    def __init__(self, log_path: str, verbose: int = 0):
        """
        로깅 콜백 초기화
        
        Args:
            log_path: 로그 CSV 파일 경로
            verbose: 상세 출력 레벨
        """
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None
        self.log_header_written = False
        self.episode_num = 0
        print(f"📊 Data logging enabled. Saving logs to: {self.log_path}")
        
    def _on_training_start(self) -> None:
        """학습 시작 시 로그 파일 열기"""
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
        """CSV 파일 헤더 작성"""
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
        """매 스텝 마다 로그 기록"""
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
        except Exception as e:
             print(f"Error logging data at step {self.num_timesteps}: {e}")
             # traceback.print_exc() # 상세 오류 확인 필요 시
        return True

    def _on_training_end(self) -> None:
        """학습 종료 시 파일 닫기"""
        if self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.close()
                print("CSV log file closed on training end.")
            except Exception as e: print(f"Error closing log file on training end: {e}")
            self.csv_file = None
            self.csv_writer = None