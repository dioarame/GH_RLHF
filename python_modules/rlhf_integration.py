#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 통합 시스템

이 모듈은 인간 피드백을 PPO 학습에 통합하는 완전한 RLHF 파이프라인을 제공합니다.
기존 환경 보상과 학습된 인간 선호도 보상을 결합하여 최종 보상을 계산합니다.
"""

import os
import json
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Any, Optional, Tuple

# 로컬 모듈 임포트
from reward_model_trainer import RewardModelTrainer, RewardModel
from enhanced_env import EnhancedGrasshopperEnv  # 기존 환경 임포트

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLHFRewardWrapper(gym.Wrapper):
    """
    RLHF 보상을 통합하는 환경 래퍼
    환경 보상과 인간 선호도 보상을 결합합니다.
    """
    
    def __init__(self, env, reward_model_path=None, reward_weights=None):
        """
        Args:
            env: 기본 Grasshopper 환경
            reward_model_path: 학습된 보상 모델 경로
            reward_weights: 보상 가중치 {'env': 0.5, 'human': 0.5}
        """
        super().__init__(env)
        
        # 보상 가중치 설정
        self.reward_weights = reward_weights or {'env': 0.7, 'human': 0.3}
        
        # 인간 선호도 보상 모델 로드
        self.human_reward_model = None
        if reward_model_path and os.path.exists(reward_model_path):
            self.load_reward_model(reward_model_path)
            logger.info(f"인간 선호도 보상 모델 로드됨: {reward_model_path}")
        else:
            logger.warning("인간 선호도 보상 모델이 제공되지 않았습니다. 환경 보상만 사용합니다.")
        
        # 보상 통계
        self.reward_stats = {
            'env_rewards': [],
            'human_rewards': [],
            'combined_rewards': []
        }
    
    def load_reward_model(self, model_path):
        """보상 모델 로드"""
        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dim = checkpoint['state_dim']
            
            # 트레이너 초기화 및 모델 로드
            self.reward_trainer = RewardModelTrainer(state_dim=state_dim, device='cpu')
            self.reward_trainer.load_model(model_path)
            self.human_reward_model = self.reward_trainer.model
            
            logger.info("인간 선호도 보상 모델 로드 성공")
            
        except Exception as e:
            logger.error(f"보상 모델 로드 실패: {e}")
            self.human_reward_model = None
    
    def calculate_human_reward(self, state):
        """인간 선호도 기반 보상 계산"""
        if self.human_reward_model is None:
            return 0.0
        
        try:
            # 상태를 모델에 맞는 형식으로 변환
            if isinstance(state, dict):
                # 상태가 딕셔너리인 경우 주요 지표 추출
                state_vector = [
                    state.get('bcr', 0.0),
                    state.get('far', 0.0), 
                    state.get('sunlight', 0.0),
                    state.get('sv_ratio', 0.0)
                ]
            elif isinstance(state, (list, np.ndarray)):
                state_vector = state[:4]  # 처음 4개 요소 사용
            else:
                return 0.0
            
            # 보상 예측
            human_reward = self.reward_trainer.predict_reward(state_vector)
            return float(human_reward)
            
        except Exception as e:
            logger.warning(f"인간 보상 계산 오류: {e}")
            return 0.0
    
    def step(self, action):
        """환경 스텝 실행 및 RLHF 보상 계산"""
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # 인간 선호도 보상 계산
        human_reward = self.calculate_human_reward(info)
        
        # 최종 보상 계산 (가중 합)
        combined_reward = (
            self.reward_weights['env'] * env_reward + 
            self.reward_weights['human'] * human_reward
        )
        
        # 통계 업데이트
        self.reward_stats['env_rewards'].append(env_reward)
        self.reward_stats['human_rewards'].append(human_reward)
        self.reward_stats['combined_rewards'].append(combined_reward)
        
        # 정보 업데이트
        info.update({
            'env_reward': env_reward,
            'human_reward': human_reward,
            'combined_reward': combined_reward,
            'reward_weights': self.reward_weights
        })
        
        return obs, combined_reward, terminated, truncated, info
    
    def get_reward_statistics(self):
        """보상 통계 반환"""
        if not self.reward_stats['combined_rewards']:
            return {}
        
        return {
            'env_reward_mean': np.mean(self.reward_stats['env_rewards']),
            'human_reward_mean': np.mean(self.reward_stats['human_rewards']),
            'combined_reward_mean': np.mean(self.reward_stats['combined_rewards']),
            'env_reward_std': np.std(self.reward_stats['env_rewards']),
            'human_reward_std': np.std(self.reward_stats['human_rewards']),
            'combined_reward_std': np.std(self.reward_stats['combined_rewards']),
            'total_steps': len(self.reward_stats['combined_rewards'])
        }

class RLHFCallback(BaseCallback):
    """
    RLHF 학습 과정을 모니터링하는 콜백
    """
    
    def __init__(self, reward_model_update_interval=1000, feedback_data_dir='data/feedback', verbose=0):
        """
        Args:
            reward_model_update_interval: 보상 모델 업데이트 간격 (스텝 수)
            feedback_data_dir: 피드백 데이터 디렉토리
            verbose: 로그 레벨
        """
        super().__init__(verbose)
        self.reward_model_update_interval = reward_model_update_interval
        self.feedback_data_dir = feedback_data_dir
        self.last_model_update = 0
        
        # 통계 저장
        self.rlhf_stats = {
            'reward_model_updates': 0,
            'human_reward_history': [],
            'env_reward_history': []
        }
    
    def _on_step(self) -> bool:
        """매 스텝마다 호출되는 함수"""
        
        # 보상 통계 수집
        if hasattr(self.training_env.envs[0], 'get_reward_statistics'):
            stats = self.training_env.envs[0].get_reward_statistics()
            if stats:
                self.rlhf_stats['human_reward_history'].append(stats.get('human_reward_mean', 0))
                self.rlhf_stats['env_reward_history'].append(stats.get('env_reward_mean', 0))
        
        # 주기적 보상 모델 업데이트 체크
        if (self.num_timesteps - self.last_model_update) >= self.reward_model_update_interval:
            self.update_reward_model()
            self.last_model_update = self.num_timesteps
        
        return True
    
    def update_reward_model(self):
        """새로운 피드백 데이터로 보상 모델 업데이트"""
        try:
            # 새로운 피드백 데이터 확인
            feedback_files = self.find_new_feedback_data()
            
            if feedback_files:
                logger.info(f"새로운 피드백 데이터 발견: {len(feedback_files)}개 파일")
                
                # 보상 모델 재학습 (실제 구현에서는 더 정교한 로직 필요)
                self.retrain_reward_model(feedback_files)
                self.rlhf_stats['reward_model_updates'] += 1
                
                logger.info(f"보상 모델 업데이트 완료 (업데이트 횟수: {self.rlhf_stats['reward_model_updates']})")
        
        except Exception as e:
            logger.error(f"보상 모델 업데이트 오류: {e}")
    
    def find_new_feedback_data(self):
        """새로운 피드백 데이터 파일 찾기"""
        feedback_files = []
        
        if not os.path.exists(self.feedback_data_dir):
            return feedback_files
        
        for filename in os.listdir(self.feedback_data_dir):
            if filename.endswith('_feedback_data.json'):
                filepath = os.path.join(self.feedback_data_dir, filename)
                # 파일 수정 시간을 기반으로 새로운 데이터 판단 (간단한 예시)
                if os.path.getmtime(filepath) > self.last_model_update:
                    feedback_files.append(filepath)
        
        return feedback_files
    
    def retrain_reward_model(self, feedback_files):
        """피드백 데이터로 보상 모델 재학습"""
        # 실제 구현에서는 여기서 보상 모델을 재학습하고
        # 환경의 보상 모델을 업데이트해야 합니다.
        # 지금은 로그만 출력
        logger.info(f"보상 모델 재학습 시뮬레이션: {len(feedback_files)}개 파일 처리")

class RLHFTrainer:
    """
    RLHF 전체 파이프라인을 관리하는 트레이너
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 기본 설정
        self.config = {
            'env': {
                'gh_file_path': 'path/to/your/definition.gh',
                'state_port': 5557,
                'action_port': 5556,
                'mesh_port': 5558,
                'timeout': 10.0
            },
            'rlhf': {
                'reward_weights': {'env': 0.7, 'human': 0.3},
                'reward_model_update_interval': 1000,
                'feedback_data_dir': 'data/feedback'
            },
            'ppo': {
                'total_timesteps': 50000,
                'learning_rate': 3e-4,
                'n_steps': 512,
                'batch_size': 64,
                'n_epochs': 5,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5
            },
            'logging': {
                'log_dir': 'logs/rlhf',
                'model_save_dir': 'models/rlhf',
                'save_interval': 5000
            }
        }
        
        # 설정 파일이 있으면 로드
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # 디렉토리 생성
        for dir_path in [self.config['rlhf']['feedback_data_dir'], 
                        self.config['logging']['log_dir'],
                        self.config['logging']['model_save_dir']]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_config(self, config_path):
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        # 기본 설정과 병합
        self.config.update(loaded_config)
        logger.info(f"설정 파일 로드됨: {config_path}")
    
    def create_environment(self, reward_model_path=None):
        """RLHF 환경 생성"""
        # 기본 Grasshopper 환경 생성
        base_env = EnhancedGrasshopperEnv(
            gh_file_path=self.config['env']['gh_file_path'],
            state_port=self.config['env']['state_port'],
            action_port=self.config['env']['action_port'],
            mesh_port=self.config['env']['mesh_port'],
            timeout=self.config['env']['timeout']
        )
        
        # RLHF 래퍼 적용
        rlhf_env = RLHFRewardWrapper(
            base_env, 
            reward_model_path=reward_model_path,
            reward_weights=self.config['rlhf']['reward_weights']
        )
        
        # 벡터화된 환경으로 래핑
        vec_env = DummyVecEnv([lambda: rlhf_env])
        
        return vec_env
    
    def train_initial_policy(self):
        """초기 정책 학습 (환경 보상만 사용)"""
        logger.info("초기 정책 학습 시작 (환경 보상만 사용)...")
        
        # 환경 생성 (보상 모델 없음)
        env = self.create_environment()
        
        # PPO 에이전트 생성
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config['ppo']['learning_rate'],
            n_steps=self.config['ppo']['n_steps'],
            batch_size=self.config['ppo']['batch_size'],
            n_epochs=self.config['ppo']['n_epochs'],
            gamma=self.config['ppo']['gamma'],
            gae_lambda=self.config['ppo']['gae_lambda'],
            clip_range=self.config['ppo']['clip_range'],
            ent_coef=self.config['ppo']['ent_coef'],
            vf_coef=self.config['ppo']['vf_coef'],
            verbose=1,
            tensorboard_log=self.config['logging']['log_dir']
        )
        
        # 학습
        model.learn(total_timesteps=self.config['ppo']['total_timesteps'] // 4)
        
        # 초기 모델 저장
        initial_model_path = os.path.join(
            self.config['logging']['model_save_dir'], 
            'initial_policy.zip'
        )
        model.save(initial_model_path)
        
        logger.info(f"초기 정책 학습 완료: {initial_model_path}")
        return initial_model_path
    
    def train_with_human_feedback(self, initial_model_path, reward_model_path):
        """인간 피드백을 사용한 정책 학습"""
        logger.info("인간 피드백을 사용한 정책 학습 시작...")
        
        # RLHF 환경 생성
        env = self.create_environment(reward_model_path)
        
        # 초기 모델 로드
        model = PPO.load(initial_model_path, env=env)
        
        # RLHF 콜백 생성
        rlhf_callback = RLHFCallback(
            reward_model_update_interval=self.config['rlhf']['reward_model_update_interval'],
            feedback_data_dir=self.config['rlhf']['feedback_data_dir']
        )
        
        # 추가 학습
        model.learn(
            total_timesteps=self.config['ppo']['total_timesteps'],
            callback=rlhf_callback,
            reset_num_timesteps=False
        )
        
        # 최종 모델 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = os.path.join(
            self.config['logging']['model_save_dir'], 
            f'rlhf_policy_{timestamp}.zip'
        )
        model.save(final_model_path)
        
        logger.info(f"RLHF 정책 학습 완료: {final_model_path}")
        return final_model_path, rlhf_callback.rlhf_stats
    
    def run_full_pipeline(self, feedback_data_path=None):
        """전체 RLHF 파이프라인 실행"""
        logger.info("RLHF 전체 파이프라인 시작")
        
        try:
            # 1. 초기 정책 학습
            initial_model_path = self.train_initial_policy()
            
            # 2. 인간 피드백이 있는 경우 보상 모델 학습
            reward_model_path = None
            if feedback_data_path and os.path.exists(feedback_data_path):
                logger.info("인간 피드백 데이터로 보상 모델 학습...")
                
                # 보상 모델 트레이너 생성
                trainer = RewardModelTrainer(state_dim=4, device='cpu')
                train_loader, val_loader = trainer.prepare_data(feedback_data_path)
                
                # 모델 학습
                trainer.train(train_loader, val_loader)
                
                # 모델 저장
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                reward_model_path = os.path.join(
                    self.config['logging']['model_save_dir'],
                    f'reward_model_{timestamp}.pt'
                )
                trainer.save_model(reward_model_path)
                
                logger.info(f"보상 모델 학습 완료: {reward_model_path}")
            
            # 3. RLHF 정책 학습
            final_model_path, rlhf_stats = self.train_with_human_feedback(
                initial_model_path, 
                reward_model_path
            )
            
            logger.info("RLHF 전체 파이프라인 완료!")
            return {
                'initial_model': initial_model_path,
                'reward_model': reward_model_path,
                'final_model': final_model_path,
                'stats': rlhf_stats
            }
            
        except Exception as e:
            logger.error(f"RLHF 파이프라인 오류: {e}")
            raise

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 통합 학습')
    parser.add_argument('--config', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--feedback-data', type=str, default=None,
                       help='인간 피드백 데이터 파일 경로')
    parser.add_argument('--gh-file', type=str, required=True,
                       help='Grasshopper 파일 경로')
    parser.add_argument('--output-dir', type=str, default='rlhf_output',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # RLHF 트레이너 생성
    trainer = RLHFTrainer(args.config)
    
    # Grasshopper 파일 경로 설정
    trainer.config['env']['gh_file_path'] = args.gh_file
    
    # 결과 저장 경로 설정
    trainer.config['logging']['model_save_dir'] = args.output_dir
    trainer.config['logging']['log_dir'] = os.path.join(args.output_dir, 'logs')
    
    try:
        # 전체 파이프라인 실행
        results = trainer.run_full_pipeline(args.feedback_data)
        
        # 결과 저장
        results_path = os.path.join(args.output_dir, 'rlhf_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"RLHF 학습 완료! 결과: {results_path}")
        
    except Exception as e:
        logger.error(f"RLHF 학습 실패: {e}")
        raise

if __name__ == "__main__":
    main()
