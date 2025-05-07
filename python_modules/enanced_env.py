#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 강화된 환경

이 모듈은 env_simple.py를 확장하여 인간 피드백 보상 모델을 통합합니다.
인간 선호도를 반영한 보상 함수를 사용하여 강화학습을 개선합니다.
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import time
import json
import traceback
import logging
import torch
from pathlib import Path
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_env.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 기존 환경 클래스 불러오기
try:
    # 현재 스크립트 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 환경 변수에 현재 디렉토리 추가
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # 기존 환경 모듈 불러오기
    from env_simple import SimpleGrasshopperEnv
    
    # 보상 모델 불러오기
    from reward_model import RewardModelInference
    
    logger.info("의존성 모듈 로드 성공: SimpleGrasshopperEnv, RewardModelInference")
    
except ImportError as e:
    logger.error(f"의존성 모듈 로드 실패: {e}")
    logger.error("python_modules 디렉토리 내에 env_simple.py와 reward_model.py가 있는지 확인하세요.")
    raise

class EnhancedGrasshopperEnv(SimpleGrasshopperEnv):
    """
    인간 피드백이 통합된 확장 그래스호퍼 환경
    
    이 클래스는 SimpleGrasshopperEnv를 상속받아 확장하며,
    인간 피드백을 기반으로 훈련된 보상 모델을 통합합니다.
    """
    
    def __init__(
        self,
        compute_url,
        gh_definition_path,
        state_output_param_name,
        reward_output_param_name,
        reward_model_path,
        slider_info_param_name="SliderInfo",
        human_reward_weight=0.5,
        max_episode_steps=100,
        action_push_port=5556,
        use_push_mode=True,
        reward_model_device=None
    ):
        """
        초기화 함수
        
        Args:
            compute_url: Rhino.Compute 서버 URL
            gh_definition_path: Grasshopper 정의 파일 경로
            state_output_param_name: 상태 출력 파라미터 이름
            reward_output_param_name: 보상 출력 파라미터 이름
            reward_model_path: 보상 모델 파일 경로
            slider_info_param_name: 슬라이더 정보 파라미터 이름
            human_reward_weight: 인간 보상 가중치 (0.0 ~ 1.0)
            max_episode_steps: 에피소드 최대 스텝 수
            action_push_port: ZMQ 액션 전송 포트
            use_push_mode: PUSH 모드 사용 여부
            reward_model_device: 보상 모델 장치 ('cpu', 'cuda', 또는 None)
        """
        # 부모 클래스 초기화
        super().__init__(
            compute_url=compute_url,
            gh_definition_path=gh_definition_path,
            state_output_param_name=state_output_param_name,
            reward_output_param_name=reward_output_param_name,
            slider_info_param_name=slider_info_param_name,
            max_episode_steps=max_episode_steps,
            action_push_port=action_push_port,
            use_push_mode=use_push_mode
        )
        
        # 보상 모델 로드
        self.reward_model_path = reward_model_path
        self.reward_model_device = reward_model_device
        self.human_reward_weight = np.clip(human_reward_weight, 0.0, 1.0)
        self.original_reward_weight = 1.0 - self.human_reward_weight
        
        # 보상 모델 초기화 여부
        self.reward_model_initialized = False
        
        # 보상 통계 추적
        self.original_rewards = []
        self.human_rewards = []
        self.combined_rewards = []
        
        # 보상 정규화 파라미터
        self.reward_stats = {
            "original": {"min": float('inf'), "max": float('-inf')},
            "human": {"min": float('inf'), "max": float('-inf')}
        }
        
        # 초기화 성공 메시지
        logger.info(f"강화된 환경 초기화 성공.")
        logger.info(f"인간 보상 가중치: {self.human_reward_weight}, 원본 보상 가중치: {self.original_reward_weight}")
        
        # 보상 모델 초기화
        self._initialize_reward_model()
    
    def _initialize_reward_model(self):
        """보상 모델 초기화"""
        if not os.path.exists(self.reward_model_path):
            logger.error(f"보상 모델 파일을 찾을 수 없습니다: {self.reward_model_path}")
            return False
        
        try:
            # 보상 모델 인스턴스 생성
            self.reward_model = RewardModelInference(
                model_path=self.reward_model_path,
                device=self.reward_model_device
            )
            
            # 테스트 상태 벡터로 추론 시도
            test_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            test_reward = self.reward_model.predict_reward(test_state)
            
            logger.info(f"보상 모델 초기화 성공. 테스트 보상: {test_reward}")
            self.reward_model_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"보상 모델 초기화 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            self.reward_model_initialized = False
            return False
    
    def _update_reward_stats(self, reward_type, value):
        """보상 통계 업데이트"""
        if reward_type not in self.reward_stats:
            return
        
        stats = self.reward_stats[reward_type]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
    
    def _normalize_reward(self, reward_type, value):
        """보상 값 정규화 (min-max 스케일링)"""
        if reward_type not in self.reward_stats:
            return value
        
        stats = self.reward_stats[reward_type]
        
        # 충분한 데이터가 없으면 정규화 건너뜀
        if stats["min"] == float('inf') or stats["max"] == float('-inf') or stats["min"] == stats["max"]:
            return value
        
        # Min-max 정규화
        normalized = (value - stats["min"]) / (stats["max"] - stats["min"])
        
        # [-1, 1] 범위로 스케일
        scaled = normalized * 2.0 - 1.0
        
        return scaled
    
    def _compute_human_reward(self, state):
        """인간 피드백 보상 모델로부터 보상 계산"""
        if not self.reward_model_initialized:
            return 0.0
        
        try:
            # 보상 모델 추론
            human_reward = self.reward_model.predict_reward(state)
            
            # 통계 업데이트
            self._update_reward_stats("human", human_reward)
            
            # 나중에 정규화를 위해 저장
            self.human_rewards.append(human_reward)
            
            return human_reward
            
        except Exception as e:
            logger.error(f"인간 보상 계산 중 오류: {e}")
            return 0.0
    
    def _combine_rewards(self, original_reward, human_reward, apply_normalization=True):
        """원본 보상과 인간 피드백 보상 결합"""
        # 정규화 적용
        if apply_normalization:
            original_normalized = self._normalize_reward("original", original_reward)
            human_normalized = self._normalize_reward("human", human_reward)
        else:
            original_normalized = original_reward
            human_normalized = human_reward
        
        # 가중치를 적용한 결합
        combined = (
            self.original_reward_weight * original_normalized + 
            self.human_reward_weight * human_normalized
        )
        
        # 저장
        self.combined_rewards.append(combined)
        
        return combined
    
    def set_reward_weights(self, human_weight):
        """보상 가중치 설정"""
        self.human_reward_weight = np.clip(human_weight, 0.0, 1.0)
        self.original_reward_weight = 1.0 - self.human_reward_weight
        logger.info(f"보상 가중치 갱신: 인간={self.human_reward_weight}, 원본={self.original_reward_weight}")
    
    def step(self, action):
        """
        환경 진행 (확장 버전)
        
        원본 환경의 step 메서드를 오버라이드하여
        인간 피드백 보상 모델을 통합합니다.
        """
        # 원본 환경의 step 함수 호출
        state, original_reward, terminated, truncated, info = super().step(action)
        
        # 통계 업데이트
        self._update_reward_stats("original", original_reward)
        
        # 나중에 정규화를 위해 저장
        self.original_rewards.append(original_reward)
        
        # 인간 피드백 보상 계산
        human_reward = self._compute_human_reward(state)
        
        # 결합된 보상 계산
        combined_reward = self._combine_rewards(original_reward, human_reward)
        
        # 추가 정보
        info.update({
            "original_reward": original_reward,
            "human_reward": human_reward,
            "reward_weights": {
                "original": self.original_reward_weight,
                "human": self.human_reward_weight
            }
        })
        
        # 로깅
        if self._step_counter % 100 == 0:
            logger.info(f"스텝 {self._step_counter}: 원본 보상={original_reward:.4f}, 인간 보상={human_reward:.4f}, 결합={combined_reward:.4f}")
        
        return state, combined_reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        """
        환경 초기화 (확장 버전)
        
        원본 환경의 reset 메서드를 오버라이드하여
        보상 모델이 로드되지 않았으면 로드합니다.
        """
        # 보상 모델이 로드되지 않았으면 다시 시도
        if not self.reward_model_initialized:
            self._initialize_reward_model()
        
        # 원본 환경의 reset 함수 호출
        state, info = super().reset(seed=seed, options=options)
        
        # 초기 인간 보상 계산
        if self.reward_model_initialized:
            human_reward = self._compute_human_reward(state)
            info["initial_human_reward"] = human_reward
            
            if self._step_counter == 0:
                logger.info(f"초기 상태 인간 보상: {human_reward:.4f}")
        
        return state, info
    
    def close(self):
        """
        환경 종료 (확장 버전)
        
        원본 환경의 close 메서드를 오버라이드하여
        추가 리소스 정리
        """
        # 원본 종료 처리
        super().close()
        
        # 추가 정리 작업
        self.reward_model = None
        self.reward_model_initialized = False
        
        logger.info("강화된 환경 종료 완료")
    
    def get_reward_stats(self):
        """보상 통계 정보 반환"""
        stats = {
            "original": {
                "min": self.reward_stats["original"]["min"] if self.reward_stats["original"]["min"] != float('inf') else 0.0,
                "max": self.reward_stats["original"]["max"] if self.reward_stats["original"]["max"] != float('-inf') else 0.0,
                "mean": np.mean(self.original_rewards) if self.original_rewards else 0.0,
                "std": np.std(self.original_rewards) if self.original_rewards else 0.0,
                "count": len(self.original_rewards)
            },
            "human": {
                "min": self.reward_stats["human"]["min"] if self.reward_stats["human"]["min"] != float('inf') else 0.0,
                "max": self.reward_stats["human"]["max"] if self.reward_stats["human"]["max"] != float('-inf') else 0.0,
                "mean": np.mean(self.human_rewards) if self.human_rewards else 0.0,
                "std": np.std(self.human_rewards) if self.human_rewards else 0.0,
                "count": len(self.human_rewards)
            },
            "combined": {
                "mean": np.mean(self.combined_rewards) if self.combined_rewards else 0.0,
                "std": np.std(self.combined_rewards) if self.combined_rewards else 0.0,
                "count": len(self.combined_rewards)
            },
            "weights": {
                "original": self.original_reward_weight,
                "human": self.human_reward_weight
            }
        }
        
        return stats

class EnhancedGrasshopperEnvFactory:
    """
    강화된 그래스호퍼 환경 팩토리 클래스
    
    표준 gym 환경 생성 인터페이스 제공
    """
    
    @staticmethod
    def create_env(
        compute_url,
        gh_definition_path,
        reward_model_path,
        state_output_param="CurrentState",
        reward_output_param="CalculatedReward",
        slider_info_param="SliderInfo",
        human_reward_weight=0.5,
        action_push_port=5556,
        device="auto"
    ):
        """
        환경 생성 함수
        
        Args:
            compute_url: Rhino.Compute 서버 URL
            gh_definition_path: Grasshopper 정의 파일 경로
            reward_model_path: 보상 모델 파일 경로
            state_output_param: 상태 출력 파라미터 이름
            reward_output_param: 보상 출력 파라미터 이름
            slider_info_param: 슬라이더 정보 파라미터 이름
            human_reward_weight: 인간 보상 가중치 (0.0 ~ 1.0)
            action_push_port: ZMQ 액션 전송 포트
            device: 보상 모델 장치 ('cpu', 'cuda', 'auto')
            
        Returns:
            EnhancedGrasshopperEnv: 환경 인스턴스
        """
        # 디바이스 설정
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 환경 생성
        env = EnhancedGrasshopperEnv(
            compute_url=compute_url,
            gh_definition_path=gh_definition_path,
            state_output_param_name=state_output_param,
            reward_output_param_name=reward_output_param,
            reward_model_path=reward_model_path,
            slider_info_param_name=slider_info_param,
            human_reward_weight=human_reward_weight,
            action_push_port=action_push_port,
            reward_model_device=device
        )
        
        logger.info(f"강화된 환경 생성 완료: {env}")
        return env

def parse_args():
    """
    커맨드 라인 인자 파싱
    """
    parser = argparse.ArgumentParser(description='강화된 그래스호퍼 환경 테스트')
    parser.add_argument('--gh-path', type=str, required=True,
                        help='Grasshopper 정의 파일 경로')
    parser.add_argument('--compute-url', type=str, default="http://localhost:6500/grasshopper",
                        help='Rhino.Compute 서버 URL')
    parser.add_argument('--reward-model', type=str, required=True,
                        help='보상 모델 파일 경로')
    parser.add_argument('--human-weight', type=float, default=0.5,
                        help='인간 보상 가중치 (0.0 ~ 1.0)')
    parser.add_argument('--port', type=int, default=5556,
                        help='ZMQ 액션 전송 포트')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='보상 모델 디바이스')
    parser.add_argument('--episodes', type=int, default=3,
                        help='테스트할 에피소드 수')
    parser.add_argument('--steps', type=int, default=10,
                        help='각 에피소드 스텝 수')
    
    return parser.parse_args()

def test_environment(env, episodes=3, steps_per_episode=10):
    """
    환경 테스트 함수
    
    Args:
        env: 테스트할 환경 인스턴스
        episodes: 테스트할 에피소드 수
        steps_per_episode: 각 에피소드의 스텝 수
    """
    logger.info("환경 테스트 시작...")
    
    episode_rewards = []
    
    for episode in range(episodes):
        logger.info(f"에피소드 {episode+1}/{episodes} 시작")
        
        # 환경 초기화
        state, info = env.reset()
        
        episode_reward = 0
        episode_human_reward = 0
        episode_original_reward = 0
        
        for step in range(steps_per_episode):
            # 랜덤 액션 샘플링
            action = env.action_space.sample()
            
            # 환경 진행
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 보상 누적
            episode_reward += reward
            episode_human_reward += info.get("human_reward", 0.0)
            episode_original_reward += info.get("original_reward", 0.0)
            
            # 디버그 출력
            logger.info(f"  스텝 {step+1}: 액션={action}, 보상={reward:.4f}")
            logger.info(f"    - 원본 보상: {info.get('original_reward', 0.0):.4f}")
            logger.info(f"    - 인간 보상: {info.get('human_reward', 0.0):.4f}")
            
            # 종료 처리
            if terminated or truncated:
                logger.info(f"  에피소드 조기 종료: terminated={terminated}, truncated={truncated}")
                break
        
        # 에피소드 통계
        episode_rewards.append({
            "total": episode_reward,
            "human": episode_human_reward,
            "original": episode_original_reward
        })
        
        logger.info(f"에피소드 {episode+1} 완료: 총 보상={episode_reward:.4f}")
        logger.info(f"  - 원본 보상 합계: {episode_original_reward:.4f}")
        logger.info(f"  - 인간 보상 합계: {episode_human_reward:.4f}")
    
    # 전체 통계
    avg_reward = np.mean([e["total"] for e in episode_rewards])
    avg_human = np.mean([e["human"] for e in episode_rewards])
    avg_original = np.mean([e["original"] for e in episode_rewards])
    
    logger.info("\n테스트 완료!")
    logger.info(f"평균 보상: {avg_reward:.4f}")
    logger.info(f"평균 원본 보상: {avg_original:.4f}")
    logger.info(f"평균 인간 보상: {avg_human:.4f}")
    
    # 보상 통계 출력
    reward_stats = env.get_reward_stats()
    logger.info("\n보상 통계:")
    for reward_type, stats in reward_stats.items():
        if reward_type != "weights":
            logger.info(f"  {reward_type}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value}")
    
    logger.info(f"보상 가중치: 원본={reward_stats['weights']['original']}, 인간={reward_stats['weights']['human']}")
    
    return episode_rewards

def main():
    """
    메인 함수
    """
    print("=" * 80)
    print("강화된 그래스호퍼 환경 테스트")
    print("=" * 80)
    
    args = parse_args()
    
    try:
        print(f"환경 생성 중...")
        print(f"  - Grasshopper 파일: {args.gh_path}")
        print(f"  - 보상 모델: {args.reward_model}")
        print(f"  - 인간 보상 가중치: {args.human_weight}")
        print(f"  - 디바이스: {args.device}")
        
        # 환경 생성
        env = EnhancedGrasshopperEnvFactory.create_env(
            compute_url=args.compute_url,
            gh_definition_path=args.gh_path,
            reward_model_path=args.reward_model,
            human_reward_weight=args.human_weight,
            action_push_port=args.port,
            device=args.device
        )
        
        # 환경 정보 출력
        print("\n환경 정보:")
        print(f"  - 액션 공간: {env.action_space}")
        print(f"  - 관측 공간: {env.observation_space}")
        
        # 환경 테스트
        print("\n환경 테스트 시작...")
        episode_rewards = test_environment(env, args.episodes, args.steps)
        
        print(f"\n✅ 테스트 완료! 평균 보상: {np.mean([e['total'] for e in episode_rewards]):.4f}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # 환경 종료
        if 'env' in locals():
            env.close()
        print("\n환경 리소스 정리 완료.")

if __name__ == "__main__":
    main()
        """
        초기화 함수
        
        Args:
            compute_url: Rhino.Compute 서버 URL
            gh_definition_path: Grasshopper 정의 파일 경로
            state_output_param_name: 상태 출력 파라미터 이름
            reward_output_param_name: 보상 출력 파라미터 이름
            reward_model_path: 보상 모델 파일 경로
            slider_info_param_name: 슬라이더 정보 파라미터 이름
            human_reward_weight: 인간 보상 가중치 (0.0 ~ 1.0)
            max_episode_steps: 에피소드 최대 스텝 수
            action_push_port: ZM
