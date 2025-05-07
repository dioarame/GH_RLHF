#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 보상 모델

이 모듈은 인간 피드백 선호도 쌍을 기반으로 보상 모델을 훈련합니다.
훈련된 모델은 환경의 보상 함수를 강화하는 데 사용됩니다.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pickle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reward_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """
    인간 선호도 데이터셋
    """
    def __init__(self, preference_file):
        """
        초기화 함수
        
        Args:
            preference_file: 선호도 쌍을 담은 JSON 파일 경로
        """
        self.data = []
        
        # 선호도 데이터 로드
        try:
            with open(preference_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                # 메타데이터 확인
                metadata = content.get("metadata", {})
                pairs_count = metadata.get("pairs_count", 0)
                
                logger.info(f"선호도 데이터 로드 중: {preference_file}, 쌍 수: {pairs_count}")
                
                # 선호도 쌍 파싱
                preference_pairs = content.get("preference_pairs", [])
                
                for pair in preference_pairs:
                    preferred = pair.get("preferred", {})
                    less_preferred = pair.get("less_preferred", {})
                    
                    # 필수 필드 확인
                    if not preferred.get("state") or not less_preferred.get("state"):
                        continue
                    
                    # 상태 벡터로 변환
                    preferred_state = preferred.get("state", [0.0])
                    less_preferred_state = less_preferred.get("state", [0.0])
                    
                    # float 변환
                    if isinstance(preferred_state, list):
                        preferred_state = np.array(preferred_state, dtype=np.float32)
                    else:
                        preferred_state = np.array([float(preferred_state)], dtype=np.float32)
                        
                    if isinstance(less_preferred_state, list):
                        less_preferred_state = np.array(less_preferred_state, dtype=np.float32)
                    else:
                        less_preferred_state = np.array([float(less_preferred_state)], dtype=np.float32)
                    
                    # 점수 차이 가져오기
                    score_diff = pair.get("score_diff", 1.0)
                    
                    # 데이터 추가
                    self.data.append({
                        "preferred_state": preferred_state,
                        "less_preferred_state": less_preferred_state,
                        "score_diff": score_diff
                    })
                
                logger.info(f"{len(self.data)}개 선호도 쌍 로드됨")
                
        except Exception as e:
            logger.error(f"선호도 데이터 로드 중 오류: {e}")
            self.data = []
        
        # 차원 일관성 확인 및 패딩
        if self.data:
            # 최대 차원 찾기
            max_dim = max(
                max(item["preferred_state"].shape[0] for item in self.data),
                max(item["less_preferred_state"].shape[0] for item in self.data)
            )
            
            # 패딩
            for item in self.data:
                preferred_dim = item["preferred_state"].shape[0]
                less_preferred_dim = item["less_preferred_state"].shape[0]
                
                if preferred_dim < max_dim:
                    padding = np.zeros(max_dim - preferred_dim, dtype=np.float32)
                    item["preferred_state"] = np.concatenate([item["preferred_state"], padding])
                
                if less_preferred_dim < max_dim:
                    padding = np.zeros(max_dim - less_preferred_dim, dtype=np.float32)
                    item["less_preferred_state"] = np.concatenate([item["less_preferred_state"], padding])
            
            self.input_dim = max_dim
            logger.info(f"입력 차원: {self.input_dim}")
        else:
            self.input_dim = 1
            logger.warning("데이터를 찾을 수 없습니다. 기본 입력 차원 1로 설정합니다.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 텐서로 변환
        preferred_state = torch.tensor(item["preferred_state"], dtype=torch.float32)
        less_preferred_state = torch.tensor(item["less_preferred_state"], dtype=torch.float32)
        score_diff = torch.tensor(item["score_diff"], dtype=torch.float32)
        
        return {
            "preferred_state": preferred_state,
            "less_preferred_state": less_preferred_state,
            "score_diff": score_diff
        }
    
    def get_input_dim(self):
        """입력 차원 반환"""
        return self.input_dim

class RewardModel(nn.Module):
    """
    인간 피드백 보상 모델
    """
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        """
        초기화 함수
        
        Args:
            input_dim: 입력 차원 수 (상태 벡터 크기)
            hidden_dims: 은닉층 차원 리스트
        """
        super(RewardModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 레이어 구성
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 최종 출력 레이어 (단일 보상 값)
        layers.append(nn.Linear(prev_dim, 1))
        
        # 순차 모델로 구성
        self.model = nn.Sequential(*layers)
        
        logger.info(f"보상 모델 초기화: 입력 차원={input_dim}, 은닉층={hidden_dims}")
    
    def forward(self, state):
        """
        순방향 연산
        
        Args:
            state: 상태 벡터
            
        Returns:
            torch.Tensor: 예측된 보상 값
        """
        return self.model(state)
    
    def save(self, path):
        """
        모델 저장
        
        Args:
            path: 저장 경로
        """
        # 상위 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 모델 구성 정보
        model_info = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "architecture": str(self)
        }
        
        # 모델 및 정보 저장
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_info": model_info
        }, path)
        
        logger.info(f"모델 저장됨: {path}")
    
    @classmethod
    def load(cls, path, device="cpu"):
        """
        모델 로드
        
        Args:
            path: 모델 파일 경로
            device: 텐서 디바이스
            
        Returns:
            RewardModel: 로드된 모델
        """
        try:
            # 저장된 모델 로드
            checkpoint = torch.load(path, map_location=device)
            
            # 모델 정보 추출
            model_info = checkpoint.get("model_info", {})
            input_dim = model_info.get("input_dim", 1)
            hidden_dims = model_info.get("hidden_dims", [64, 32])
            
            # 모델 인스턴스 생성
            model = cls(input_dim, hidden_dims)
            
            # 가중치 로드
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            logger.info(f"모델 로드됨: {path}")
            return model
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {e}")
            return None

class RewardModelTrainer:
    """
    인간 피드백 보상 모델 훈련기
    """
    def __init__(self, train_data_path, valid_data_path=None, 
                 output_dir=None, device=None, base_dir=None):
        """
        초기화 함수
        
        Args:
            train_data_path: 훈련 데이터 파일 경로
            valid_data_path: 검증 데이터 파일 경로
            output_dir: 출력 디렉토리
            device: 훈련 디바이스 ('cpu', 'cuda', 또는 None)
            base_dir: 프로젝트 기본 디렉토리
        """
        # 기본 디렉토리 설정
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = base_dir
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = os.path.join(base_dir, "data", "reward_models")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"훈련 디바이스: {self.device}")
        
        # 데이터셋 로드
        self.train_dataset = PreferenceDataset(train_data_path)
        
        if valid_data_path:
            self.valid_dataset = PreferenceDataset(valid_data_path)
        else:
            self.valid_dataset = None
        
        # 데이터 크기 확인
        if len(self.train_dataset) == 0:
            logger.error("훈련 데이터가 없습니다.")
            raise ValueError("훈련 데이터가 비어 있습니다.")
        
        # 입력 차원 가져오기
        self.input_dim = self.train_dataset.get_input_dim()
        
        # 훈련 진행 상황 저장
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')
    
    def _preference_loss(self, preferred_reward, less_preferred_reward, score_diff=None):
        """
        선호도 손실 함수
        
        Args:
            preferred_reward: 선호되는 디자인의 예측 보상
            less_preferred_reward: 덜 선호되는 디자인의 예측 보상
            score_diff: 점수 차이 (가중치 용도)
            
        Returns:
            torch.Tensor: 손실 값
        """
        # 기본 선호도 마진
        margin = 0.2
        
        # 점수 차이가 있으면 마진으로 사용
        if score_diff is not None:
            # 스케일링된 마진 (최소값 0.1)
            scaled_margin = torch.clamp(score_diff * 0.1, min=0.1)
        else:
            scaled_margin = margin
        
        # 로그 시그모이드 손실 (선호되는 항목이 더 높은 점수를 받도록)
        reward_diff = preferred_reward - less_preferred_reward
        loss = -torch.log(torch.sigmoid(reward_diff * scaled_margin))
        
        return loss.mean()
    
    def train(self, hidden_dims=[64, 32], batch_size=32, learning_rate=1e-3, 
              num_epochs=100, patience=10, min_delta=1e-4):
        """
        모델 훈련
        
        Args:
            hidden_dims: 은닉층 차원 리스트
            batch_size: 배치 크기
            learning_rate: 학습률
            num_epochs: 에폭 수
            patience: 조기 종료 인내심
            min_delta: 최소 개선 임계값
            
        Returns:
            RewardModel: 훈련된 모델
        """
        logger.info("모델 훈련 시작...")
        logger.info(f"설정: hidden_dims={hidden_dims}, batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
        
        # 모델 초기화
        model = RewardModel(self.input_dim, hidden_dims=hidden_dims)
        model.to(self.device)
        
        # 옵티마이저
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 데이터 로더
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device == "cuda"
        )
        
        # 검증 데이터 로더
        if self.valid_dataset and len(self.valid_dataset) > 0:
            valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device == "cuda"
            )
        else:
            valid_loader = None
        
        # 훈련 진행 상황 초기화
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')
        best_model_state = None
        no_improve_count = 0
        
        # 훈련 시작 시간
        start_time = time.time()
        
        # 에폭 반복
        for epoch in range(num_epochs):
            # 훈련 모드
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 배치 반복
            for batch in train_loader:
                preferred_state = batch["preferred_state"].to(self.device)
                less_preferred_state = batch["less_preferred_state"].to(self.device)
                score_diff = batch["score_diff"].to(self.device)
                
                # 순방향 계산
                preferred_reward = model(preferred_state)
                less_preferred_reward = model(less_preferred_state)
                
                # 손실 계산
                loss = self._preference_loss(preferred_reward, less_preferred_reward, score_diff)
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 손실 누적
                epoch_loss += loss.item()
                batch_count += 1
            
            # 에폭 평균 손실
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
            self.train_losses.append(avg_train_loss)
            
            # 검증
            if valid_loader:
                model.eval()
                valid_loss = 0.0
                valid_count = 0
                
                with torch.no_grad():
                    for batch in valid_loader:
                        preferred_state = batch["preferred_state"].to(self.device)
                        less_preferred_state = batch["less_preferred_state"].to(self.device)
                        score_diff = batch["score_diff"].to(self.device)
                        
                        # 순방향 계산
                        preferred_reward = model(preferred_state)
                        less_preferred_reward = model(less_preferred_state)
                        
                        # 손실 계산
                        loss = self._preference_loss(preferred_reward, less_preferred_reward, score_diff)
                        
                        # 손실 누적
                        valid_loss += loss.item()
                        valid_count += 1
                
                # 검증 평균 손실
                avg_valid_loss = valid_loss / valid_count if valid_count > 0 else 0
                self.valid_losses.append(avg_valid_loss)
                
                # 최고 모델 저장
                if avg_valid_loss < self.best_valid_loss - min_delta:
                    self.best_valid_loss = avg_valid_loss
                    best_model_state = model.state_dict().copy()
                    no_improve_count = 0
                    logger.info(f"에폭 {epoch+1}/{num_epochs}: 새로운 최고 검증 손실: {avg_valid_loss:.6f}")
                else:
                    no_improve_count += 1
                
                # 조기 종료
                if no_improve_count >= patience:
                    logger.info(f"에폭 {epoch+1}/{num_epochs}: {patience}번 동안 개선 없음. 조기 종료.")
                    break
                    
                # 진행 상황 로깅
                logger.info(f"에폭 {epoch+1}/{num_epochs}: 훈련 손실={avg_train_loss:.6f}, 검증 손실={avg_valid_loss:.6f}")
            else:
                # 검증 데이터가 없을 때 로깅
                logger.info(f"에폭 {epoch+1}/{num_epochs}: 훈련 손실={avg_train_loss:.6f}")
        
        # 훈련 완료
        elapsed_time = time.time() - start_time
        logger.info(f"훈련 완료. 소요 시간: {elapsed_time:.2f}초")
        
        # 최고 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"최고 모델 복원됨 (검증 손실: {self.best_valid_loss:.6f})")
        
        # 모델 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f"reward_model_{timestamp}.pt")
        model.save(model_path)
        
        # 훈련 곡선 저장
        self._save_training_curves()
        
        return model, model_path
    
    def _save_training_curves(self):
        """훈련 및 검증 손실 곡선 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"training_curves_{timestamp}.png")
        
        # 훈련 손실 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        
        # 검증 손실이 있으면 추가
        if self.valid_losses:
            plt.plot(self.valid_losses, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reward Model Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"훈련 곡선 저장됨: {plot_path}")
        
        # 훈련 데이터 저장
        data_path = os.path.join(self.output_dir, f"training_data_{timestamp}.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                "train_losses": self.train_losses,
                "valid_losses": self.valid_losses,
                "best_valid_loss": self.best_valid_loss
            }, f, indent=2)
        
        logger.info(f"훈련 데이터 저장됨: {data_path}")

class RewardModelInference:
    """
    학습된 보상 모델을 사용하여 추론
    """
    def __init__(self, model_path, device=None):
        """
        초기화 함수
        
        Args:
            model_path: 모델 파일 경로
            device: 추론 디바이스 ('cpu', 'cuda', 또는 None)
        """
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"추론 디바이스: {self.device}")
        
        # 모델 로드
        self.model = RewardModel.load(model_path, device=self.device)
        
        if self.model is None:
            raise ValueError(f"모델을 로드할 수 없습니다: {model_path}")
        
        logger.info(f"보상 모델 로드됨: {model_path}")
    
    def predict_reward(self, state):
        """
        상태 벡터에 대한 보상 예측
        
        Args:
            state: 상태 벡터 (numpy 배열 또는 텐서)
            
        Returns:
            float: 예측된 보상 값
        """
        self.model.eval()
        
        # numpy 배열을 텐서로 변환
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        # 차원 확인 및 수정
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 배치 차원 추가
        
        # 모델 입력 차원 확인
        model_input_dim = self.model.input_dim
        
        # 필요 시 패딩
        if state.shape[1] < model_input_dim:
            padding = torch.zeros(state.shape[0], model_input_dim - state.shape[1], dtype=torch.float32)
            state = torch.cat([state, padding], dim=1)
        # 필요 시 자르기
        elif state.shape[1] > model_input_dim:
            state = state[:, :model_input_dim]
        
        # 디바이스로 이동
        state = state.to(self.device)
        
        # 추론
        with torch.no_grad():
            reward = self.model(state)
        
        # 스칼라 값으로 변환
        return reward.item()
    
    def batch_predict_rewards(self, states):
        """
        상태 벡터 배치에 대한 보상 예측
        
        Args:
            states: 상태 벡터의 배치 (numpy 배열 또는 텐서)
            
        Returns:
            numpy.ndarray: 예측된 보상 값 배열
        """
        self.model.eval()
        
        # numpy 배열을 텐서로 변환
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        
        # 모델 입력 차원 확인
        model_input_dim = self.model.input_dim
        
        # 필요 시 패딩
        if states.shape[1] < model_input_dim:
            padding = torch.zeros(states.shape[0], model_input_dim - states.shape[1], dtype=torch.float32)
            states = torch.cat([states, padding], dim=1)
        # 필요 시 자르기
        elif states.shape[1] > model_input_dim:
            states = states[:, :model_input_dim]
        
        # 디바이스로 이동
        states = states.to(self.device)
        
        # 추론
        with torch.no_grad():
            rewards = self.model(states)
        
        # numpy 배열로 변환
        return rewards.cpu().numpy()
    
    def save_model_info(self, output_path):
        """
        모델 정보 저장
        
        Args:
            output_path: 출력 파일 경로
        """
        # 상위 디렉토리 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 모델 구조 및 설정 정보
        model_info = {
            "model_type": "RewardModel",
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "architecture": str(self.model),
            "device": self.device,
            "exported_at": datetime.now().isoformat()
        }
        
        # JSON으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"모델 정보 저장됨: {output_path}")
        
        return model_info

def main():
    """
    커맨드 라인 인터페이스
    """
    parser = argparse.ArgumentParser(description='인간 피드백 보상 모델 훈련 및 저장')
    parser.add_argument('--train-data', type=str, required=True,
                        help='훈련 데이터 JSON 파일 경로')
    parser.add_argument('--valid-data', type=str, default=None,
                        help='검증 데이터 JSON 파일 경로')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='모델 저장 디렉토리')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='훈련 디바이스 (기본값: auto)')
    parser.add_argument('--hidden-dims', type=str, default='64,32',
                        help='은닉층 차원 (쉼표로 구분)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='학습률 (기본값: 0.001)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='에폭 수 (기본값: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='조기 종료 인내심 (기본값: 10)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = None  # 자동 감지
    else:
        device = args.device
    
    # 은닉층 차원 파싱
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',') if dim.strip()]
    
    # 훈련기 초기화
    trainer = RewardModelTrainer(
        train_data_path=args.train_data,
        valid_data_path=args.valid_data,
        output_dir=args.output_dir,
        device=device
    )
    
    # 모델 훈련
    model, model_path = trainer.train(
        hidden_dims=hidden_dims,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # 모델 정보 저장
    inference = RewardModelInference(model_path, device=device)
    info_path = model_path.replace('.pt', '_info.json')
    model_info = inference.save_model_info(info_path)
    
    print(f"\n✅ 보상 모델 훈련 완료!")
    print(f"   - 모델 저장 경로: {model_path}")
    print(f"   - 모델 정보 경로: {info_path}")
    print(f"   - 입력 차원: {model_info['input_dim']}")
    print(f"   - 은닉층 구성: {model_info['hidden_dims']}")

if __name__ == "__main__":
    main()
