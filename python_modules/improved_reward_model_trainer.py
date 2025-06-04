#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 RLHF 보상 모델 학습기

주요 개선사항:
1. 더 나은 스케일링 방법 (MinMaxScaler + RobustScaler 옵션)
2. 출력 정규화 개선
3. 모델 아키텍처 단순화
4. 학습 안정성 향상
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from pathlib import Path
from datetime import datetime
import argparse

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedPreferenceDataset(Dataset):
    """개선된 선호도 데이터셋"""
    def __init__(self, preference_data, scalers=None, is_training=True, 
                 augmentation_mode='none', noise_level=0.02):
        self.preference_data = preference_data
        self.scalers = scalers
        self.is_training = is_training
        self.augmentation_mode = augmentation_mode
        self.noise_level = noise_level
        
        # Symmetry augmentation: 데이터 2배로 증강
        if augmentation_mode == 'symmetry' and is_training:
            augmented_data = []
            for pair in self.preference_data:
                # 원본 쌍
                augmented_data.append(pair)
                # 반전된 쌍
                augmented_data.append({
                    'preferred_state': pair['rejected_state'],
                    'rejected_state': pair['preferred_state']
                })
            self.preference_data = augmented_data
            logger.info(f"Symmetry augmentation applied: {len(preference_data)} → {len(self.preference_data)} pairs")
        
        if is_training and scalers is None:
            # 각 특성별로 적절한 스케일러 사용
            all_states = []
            for p_data in self.preference_data:
                all_states.append(p_data['preferred_state'])
                all_states.append(p_data['rejected_state'])
            
            all_states = np.array(all_states)
            
            # 특성별 스케일러 설정
            self.scalers = {
                'BCR': StandardScaler(),      # 0.3~0.7 범위
                'FAR': StandardScaler(),      # 2.0~5.0 범위  
                'WinterTime': MinMaxScaler(), # 30000~180000 범위 (매우 큼)
                'SVR': RobustScaler()         # 이상치 영향 최소화
            }
            
            # 각 스케일러 fit
            for i, (feature, scaler) in enumerate(self.scalers.items()):
                scaler.fit(all_states[:, i:i+1])
                
    def __len__(self):
        return len(self.preference_data)
    
    def __getitem__(self, idx):
        pair_data = self.preference_data[idx]
        
        preferred_state = np.array(pair_data['preferred_state'])
        rejected_state = np.array(pair_data['rejected_state'])
        
        # Noise augmentation (학습 시에만)
        if self.is_training and self.augmentation_mode == 'noise':
            # 각 특성의 스케일에 맞게 노이즈 추가
            noise_scales = [0.002, 0.05, 500, 0.005]  # BCR, FAR, WinterTime, SVR
            for i in range(len(preferred_state)):
                preferred_state[i] += np.random.normal(0, noise_scales[i] * self.noise_level)
                rejected_state[i] += np.random.normal(0, noise_scales[i] * self.noise_level)
        
        # 특성별 스케일링
        preferred_scaled = np.zeros_like(preferred_state)
        rejected_scaled = np.zeros_like(rejected_state)
        
        for i, (feature, scaler) in enumerate(self.scalers.items()):
            preferred_scaled[i] = scaler.transform(preferred_state[i:i+1].reshape(-1, 1))[0, 0]
            rejected_scaled[i] = scaler.transform(rejected_state[i:i+1].reshape(-1, 1))[0, 0]
        
        return {
            'preferred_state': torch.FloatTensor(preferred_scaled),
            'rejected_state': torch.FloatTensor(rejected_scaled)
        }

class SimplifiedRewardModel(nn.Module):
    """단순화된 보상 모델"""
    def __init__(self, state_dim=4, hidden_dim=64):
        super(SimplifiedRewardModel, self).__init__()
        
        # 단순한 MLP 구조
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier 초기화로 출력 분포 안정화
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                # 마지막 레이어의 bias를 음수로 초기화하여 균형 맞추기
                if module.out_features == 1:
                    nn.init.constant_(module.bias, -0.02)  # 더 작은 음수 bias
                else:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state):
        return self.network(state)

class ImprovedRewardModelTrainer:
    """개선된 보상 모델 학습기"""
    def __init__(self, state_dim=4, device='cpu', config=None):
        self.device = device
        self.state_dim = state_dim
        
        self.config = config or {}
        self.config.setdefault('hidden_dim', 64)
        self.config.setdefault('learning_rate', 5e-4)
        self.config.setdefault('batch_size', 64)
        self.config.setdefault('epochs', 200)
        self.config.setdefault('patience', 50)
        self.config.setdefault('weight_decay', 1e-4)
        
        # 모델 생성
        self.model = SimplifiedRewardModel(
            state_dim=state_dim,
            hidden_dim=self.config['hidden_dim']
        ).to(device)
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 학습률 스케줄러 - 더 보수적으로
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,  # 20에서 50으로 증가
            T_mult=2  # 정수 유지
        )
        
        # 손실 함수 - margin ranking loss 사용
        self.criterion = nn.MarginRankingLoss(margin=0.05)  # margin 줄임
        
        self.training_history = {
            'train_loss': [], 'val_loss': [], 
            'val_accuracy': [], 'reward_stats': []
        }
        
    def load_feedback_data(self, feedback_path):
        """피드백 데이터 로드"""
        preference_pairs = []
        path = Path(feedback_path)
        
        if path.is_file():
            logger.info(f"Loading single feedback file: {feedback_path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 파일이 리스트 형태인 경우 (train_ready_feedback.json)
                if isinstance(data, list):
                    for item in data:
                        if 'preferred_state' in item and 'rejected_state' in item:
                            preference_pairs.append({
                                'preferred_state': item['preferred_state'],
                                'rejected_state': item['rejected_state']
                            })
                # 파일이 단일 객체인 경우
                elif isinstance(data, dict):
                    if data['selected_design'] == data['design_a_id']:
                        preferred = data['design_a_state']
                        rejected = data['design_b_state']
                    else:
                        preferred = data['design_b_state']
                        rejected = data['design_a_state']
                    
                    preference_pairs.append({
                        'preferred_state': preferred,
                        'rejected_state': rejected
                    })
                    
            except Exception as e:
                logger.error(f"Error loading file: {e}")
                raise
                
        else:  # 디렉토리인 경우
            files = list(path.glob('*.json'))
            logger.info(f"Loading {len(files)} feedback files from directory...")
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 개별 파일 처리
                    if isinstance(data, list):
                        for item in data:
                            if 'preferred_state' in item and 'rejected_state' in item:
                                preference_pairs.append({
                                    'preferred_state': item['preferred_state'],
                                    'rejected_state': item['rejected_state']
                                })
                    elif isinstance(data, dict):
                        if 'preferred_state' in data and 'rejected_state' in data:
                            preference_pairs.append({
                                'preferred_state': data['preferred_state'],
                                'rejected_state': data['rejected_state']
                            })
                        elif 'selected_design' in data:
                            # 원본 형식 처리
                            if data['selected_design'] == data['design_a_id']:
                                preferred = data['design_a_state']
                                rejected = data['design_b_state']
                            else:
                                preferred = data['design_b_state']
                                rejected = data['design_a_state']
                            
                            preference_pairs.append({
                                'preferred_state': preferred,
                                'rejected_state': rejected
                            })
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        
        if not preference_pairs:
            raise ValueError("No valid preference pairs loaded!")
            
        logger.info(f"Loaded {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def prepare_data(self, feedback_path, test_size=0.2, random_state=42):
        """데이터 준비"""
        # 데이터 로드
        preference_pairs = self.load_feedback_data(feedback_path)
        
        # 학습/검증 분할
        train_data, val_data = train_test_split(
            preference_pairs, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 증강 모드 확인
        augmentation_mode = self.config.get('augmentation_mode', 'symmetry')
        noise_level = self.config.get('noise_level', 0.02)
        
        # 데이터셋 생성
        train_dataset = ImprovedPreferenceDataset(
            train_data, 
            is_training=True,
            augmentation_mode=augmentation_mode,
            noise_level=noise_level
        )
        val_dataset = ImprovedPreferenceDataset(
            val_data, 
            scalers=train_dataset.scalers, 
            is_training=False,
            augmentation_mode='none'  # 검증 데이터는 증강하지 않음
        )
        
        # 스케일러 저장
        self.scalers = train_dataset.scalers
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0  # Windows 호환성
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Augmentation mode: {augmentation_mode}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        all_rewards = []
        
        for batch in train_loader:
            preferred = batch['preferred_state'].to(self.device)
            rejected = batch['rejected_state'].to(self.device)
            
            # 보상 예측
            preferred_rewards = self.model(preferred)
            rejected_rewards = self.model(rejected)
            
            # 손실 계산 (preferred > rejected 여야 함)
            target = torch.ones(preferred_rewards.size(0), 1).to(self.device)
            loss = self.criterion(preferred_rewards, rejected_rewards, target)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 통계 수집
            total_loss += loss.item() * preferred.size(0)
            correct = (preferred_rewards > rejected_rewards).float().sum().item()
            total_correct += correct
            total_samples += preferred.size(0)
            
            # 보상 분포 수집
            all_rewards.extend(preferred_rewards.detach().cpu().numpy().flatten())
            all_rewards.extend(rejected_rewards.detach().cpu().numpy().flatten())
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        # 보상 통계
        reward_stats = {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'negative_ratio': np.sum(np.array(all_rewards) < 0) / len(all_rewards)
        }
        
        return avg_loss, accuracy, reward_stats
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        all_rewards = []
        
        with torch.no_grad():
            for batch in val_loader:
                preferred = batch['preferred_state'].to(self.device)
                rejected = batch['rejected_state'].to(self.device)
                
                preferred_rewards = self.model(preferred)
                rejected_rewards = self.model(rejected)
                
                target = torch.ones(preferred_rewards.size(0), 1).to(self.device)
                loss = self.criterion(preferred_rewards, rejected_rewards, target)
                
                total_loss += loss.item() * preferred.size(0)
                correct = (preferred_rewards > rejected_rewards).float().sum().item()
                total_correct += correct
                total_samples += preferred.size(0)
                
                all_rewards.extend(preferred_rewards.cpu().numpy().flatten())
                all_rewards.extend(rejected_rewards.cpu().numpy().flatten())
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        reward_stats = {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'negative_ratio': np.sum(np.array(all_rewards) < 0) / len(all_rewards)
        }
        
        return avg_loss, accuracy, reward_stats
    
    def train(self, train_loader, val_loader, output_dir):
        """전체 학습 과정"""
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience_counter = 0
        
        logger.info("Starting training...")
        
        for epoch in range(self.config['epochs']):
            # 학습
            train_loss, train_acc, train_rewards = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc, val_rewards = self.validate(val_loader)
            
            # 학습률 스케줄링
            self.scheduler.step()
            
            # 기록
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['reward_stats'].append({
                'train': train_rewards,
                'val': val_rewards
            })
            
            # 로깅
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config['epochs']} | "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f} | "
                    f"Val Reward Mean: {val_rewards['mean']:.3f}, Neg Ratio: {val_rewards['negative_ratio']:.3f}"
                )
            
            # 조기 종료 체크
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'reward_stats': val_rewards
                }
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience'] and epoch > 200:  # 최소 200 에폭은 무조건 학습
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 최고 성능 모델 복원
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            logger.info(f"Restored best model from epoch {self.best_model_state['epoch']+1}")
            logger.info(f"Best validation accuracy: {self.best_model_state['val_accuracy']:.3f}")
            logger.info(f"Best model reward stats: {self.best_model_state['reward_stats']}")
    
    def save_model(self, save_path):
        """모델 저장"""
        # 스케일러 정보를 딕셔너리로 변환
        scaler_info = {}
        for feature, scaler in self.scalers.items():
            if isinstance(scaler, StandardScaler):
                scaler_info[feature] = {
                    'type': 'StandardScaler',
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            elif isinstance(scaler, MinMaxScaler):
                scaler_info[feature] = {
                    'type': 'MinMaxScaler',
                    'min': scaler.min_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            elif isinstance(scaler, RobustScaler):
                scaler_info[feature] = {
                    'type': 'RobustScaler',
                    'center': scaler.center_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scalers': scaler_info,
            'training_history': self.training_history,
            'best_model_info': self.best_model_state if hasattr(self, 'best_model_state') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Improved Reward Model Trainer')
    parser.add_argument('--feedback-data', type=str, required=True,
                        help='Path to feedback data directory')
    parser.add_argument('--output-dir', type=str, default='improved_reward_models',
                        help='Output directory for models')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--augmentation-mode', type=str, default='symmetry',
                        choices=['none', 'noise', 'symmetry', 'both'],
                        help='Data augmentation mode')
    parser.add_argument('--noise-level', type=float, default=0.02,
                        help='Noise level for noise augmentation')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정
    config = {
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': 20,
        'weight_decay': 1e-4,
        'augmentation_mode': args.augmentation_mode,
        'noise_level': args.noise_level
    }
    
    # 학습기 생성
    trainer = ImprovedRewardModelTrainer(
        state_dim=4,
        device=device,
        config=config
    )
    
    # 데이터 준비
    train_loader, val_loader = trainer.prepare_data(args.feedback_data)
    
    # 학습
    trainer.train(train_loader, val_loader, output_dir)
    
    # 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f'improved_reward_model_{args.augmentation_mode}_{timestamp}.pt'
    trainer.save_model(model_path)
    
    print(f"\n✅ 학습 완료! 모델 저장 위치: {model_path}")
    print(f"증강 모드: {args.augmentation_mode}")
    if args.augmentation_mode == 'symmetry':
        print("→ 데이터가 2배로 증강되었습니다 (반전 쌍 추가)")
    elif args.augmentation_mode == 'noise':
        print(f"→ 노이즈 레벨 {args.noise_level}로 증강되었습니다")

if __name__ == "__main__":
    main()