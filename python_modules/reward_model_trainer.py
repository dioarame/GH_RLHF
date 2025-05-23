#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 보상 모델 학습기

이 모듈은 인간 피드백 데이터를 사용하여 선호도 기반 보상 모델을 학습합니다.
Bradley-Terry 모델을 기반으로 한 쌍대비교 학습을 수행합니다.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging
from pathlib import Path
from datetime import datetime
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """인간 선호도 데이터셋"""
    
    def __init__(self, preference_pairs, scaler=None, is_training=True):
        """
        Args:
            preference_pairs: 선호도 쌍 데이터
            scaler: 데이터 정규화를 위한 스케일러
            is_training: 학습 모드 여부
        """
        self.pairs = preference_pairs
        self.is_training = is_training
        
        # 상태 데이터 추출
        preferred_states = [pair['preferred_state'] for pair in preference_pairs]
        rejected_states = [pair['rejected_state'] for pair in preference_pairs]
        
        self.preferred_states = np.array(preferred_states, dtype=np.float32)
        self.rejected_states = np.array(rejected_states, dtype=np.float32)
        
        # 데이터 정규화
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            # 모든 상태 데이터를 함께 정규화
            all_states = np.vstack([self.preferred_states, self.rejected_states])
            self.scaler.fit(all_states)
        else:
            self.scaler = scaler
        
        if self.scaler:
            self.preferred_states = self.scaler.transform(self.preferred_states)
            self.rejected_states = self.scaler.transform(self.rejected_states)
        
        # 추가 메타데이터
        self.timestamps = [pair.get('timestamp', 0) for pair in preference_pairs]
        self.session_info = [pair.get('session_info', {}) for pair in preference_pairs]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return {
            'preferred_state': torch.tensor(self.preferred_states[idx], dtype=torch.float32),
            'rejected_state': torch.tensor(self.rejected_states[idx], dtype=torch.float32),
            'timestamp': self.timestamps[idx],
            'session_info': self.session_info[idx]
        }

class RewardModel(nn.Module):
    """선호도 기반 보상 모델 (Bradley-Terry 모델)"""
    
    def __init__(self, state_dim, hidden_dims=[128, 64, 32]):
        """
        Args:
            state_dim: 상태 벡터 차원
            hidden_dims: 은닉층 차원들
        """
        super(RewardModel, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        # 네트워크 구성
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # 최종 출력층 (스칼라 보상값)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Args:
            state: 상태 벡터 [batch_size, state_dim]
        Returns:
            reward: 예측된 보상값 [batch_size, 1]
        """
        return self.network(state)
    
    def predict_preference(self, preferred_state, rejected_state):
        """
        두 상태 간의 선호도 확률을 예측
        
        Args:
            preferred_state: 선호된 상태
            rejected_state: 거부된 상태
        Returns:
            preference_prob: 선호 확률 (0-1)
        """
        preferred_reward = self.forward(preferred_state)
        rejected_reward = self.forward(rejected_state)
        
        # Bradley-Terry 모델: P(preferred > rejected) = sigmoid(r_preferred - r_rejected)
        logits = preferred_reward - rejected_reward
        return torch.sigmoid(logits)

class RewardModelTrainer:
    """보상 모델 학습기"""
    
    def __init__(self, state_dim, device='cpu', model_config=None):
        """
        Args:
            state_dim: 상태 벡터 차원
            device: 학습 디바이스
            model_config: 모델 설정
        """
        self.device = device
        self.state_dim = state_dim
        
        # 모델 설정
        self.config = model_config or {
            'hidden_dims': [128, 64, 32],
            'learning_rate': 3e-4,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'min_delta': 1e-4
        }
        
        # 모델 초기화
        self.model = RewardModel(
            state_dim=state_dim,
            hidden_dims=self.config['hidden_dims']
        ).to(device)
        
        # 최적화기 및 손실함수
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        
        # 학습 기록
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # 데이터 스케일러
        self.scaler = None
    
    def prepare_data(self, feedback_data_path, test_size=0.2, random_state=42):
        """
        피드백 데이터 준비
        
        Args:
            feedback_data_path: 피드백 데이터 파일 경로
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
        Returns:
            train_loader, val_loader: 데이터 로더들
        """
        # 피드백 데이터 로드
        with open(feedback_data_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
        
        preference_pairs = feedback_data.get('preference_pairs', [])
        
        if not preference_pairs:
            raise ValueError("선호도 쌍 데이터가 없습니다.")
        
        logger.info(f"로드된 선호도 쌍: {len(preference_pairs)}개")
        
        # 데이터 분할
        train_pairs, val_pairs = train_test_split(
            preference_pairs, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 데이터셋 생성
        train_dataset = PreferenceDataset(train_pairs, is_training=True)
        val_dataset = PreferenceDataset(val_pairs, scaler=train_dataset.scaler, is_training=False)
        
        # 스케일러 저장
        self.scaler = train_dataset.scaler
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        logger.info(f"학습 데이터: {len(train_dataset)}개")
        logger.info(f"검증 데이터: {len(val_dataset)}개")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            preferred_states = batch['preferred_state'].to(self.device)
            rejected_states = batch['rejected_state'].to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            
            # 선호도 확률 예측
            preference_probs = self.model.predict_preference(preferred_states, rejected_states)
            
            # 타겟은 항상 1 (preferred가 rejected보다 선호됨)
            targets = torch.ones_like(preference_probs)
            
            # 손실 계산
            loss = self.criterion(preference_probs.squeeze(), targets.squeeze())
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                preferred_states = batch['preferred_state'].to(self.device)
                rejected_states = batch['rejected_state'].to(self.device)
                
                # 선호도 확률 예측
                preference_probs = self.model.predict_preference(preferred_states, rejected_states)
                
                # 타겟은 항상 1
                batch_targets = torch.ones_like(preference_probs)
                
                # 손실 계산
                loss = self.criterion(preference_probs.squeeze(), batch_targets.squeeze())
                total_loss += loss.item()
                
                # 정확도 계산을 위한 예측값 저장
                predictions.extend((preference_probs.squeeze() > 0.5).cpu().numpy())
                targets.extend(batch_targets.squeeze().cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(targets, predictions)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """모델 학습"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("보상 모델 학습 시작...")
        
        for epoch in range(self.config['epochs']):
            # 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_accuracy = self.validate(val_loader)
            
            # 기록 저장
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # 로그 출력
            if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
                logger.info(
                    f"Epoch {epoch+1}/{self.config['epochs']} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
            
            # 조기 종료 체크
            if val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최고 모델로 복원
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("보상 모델 학습 완료!")
        return self.training_history
    
    def save_model(self, save_path):
        """모델 저장"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'state_dim': self.state_dim,
            'scaler_mean': self.scaler.mean_.tolist() if self.scaler else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.scaler else None,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"모델 저장됨: {save_path}")
    
    def load_model(self, load_path):
        """모델 로드"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 모델 상태 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.state_dim = checkpoint['state_dim']
        self.training_history = checkpoint.get('training_history', {})
        
        # 스케일러 복원
        if checkpoint.get('scaler_mean') and checkpoint.get('scaler_scale'):
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
            self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        logger.info(f"모델 로드됨: {load_path}")
    
    def predict_reward(self, state):
        """상태에 대한 보상 예측"""
        self.model.eval()
        
        # 입력 처리
        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 정규화 적용
        if self.scaler:
            state_numpy = state.cpu().numpy()
            state_normalized = self.scaler.transform(state_numpy)
            state = torch.tensor(state_normalized, dtype=torch.float32)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            reward = self.model(state)
        
        return reward.cpu().item()

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 보상 모델 학습')
    parser.add_argument('--feedback-data', type=str, required=True,
                       help='피드백 데이터 파일 경로')
    parser.add_argument('--output-dir', type=str, default='reward_models',
                       help='모델 저장 디렉토리')
    parser.add_argument('--state-dim', type=int, default=4,
                       help='상태 벡터 차원')
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='학습률')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='학습 디바이스')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 설정
    model_config = {
        'hidden_dims': [128, 64, 32],
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': 10,
        'min_delta': 1e-4
    }
    
    # 트레이너 초기화
    trainer = RewardModelTrainer(
        state_dim=args.state_dim,
        device=args.device,
        model_config=model_config
    )
    
    try:
        # 데이터 준비
        train_loader, val_loader = trainer.prepare_data(args.feedback_data)
        
        # 모델 학습
        history = trainer.train(train_loader, val_loader)
        
        # 모델 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(args.output_dir, f'reward_model_{timestamp}.pt')
        trainer.save_model(model_path)
        
        # 학습 결과 요약
        final_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else 0
        logger.info(f"최종 검증 정확도: {final_accuracy:.4f}")
        logger.info(f"모델 저장 경로: {model_path}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
