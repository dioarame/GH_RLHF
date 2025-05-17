#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 데이터 통합 및 분석 도구

이 스크립트는 PPO 학습 결과와 ZMQ를 통해 수집된 상태/보상 데이터를 통합하고 분석합니다.
인간 피드백을 위한 기준점을 제공하고, 모델의 성능을 평가하는 데 활용할 수 있습니다.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import torch
from pathlib import Path
import re
import time
import shutil
import traceback

# 시각화 스타일 설정
plt.style.use('ggplot')
sns.set(style="whitegrid")

class RLHFDataAnalyzer:
    """
    강화학습 데이터와 인간 피드백 데이터를 통합 분석하는 클래스
    """
    
    def __init__(self, state_reward_log_path, model_dir=None, ppo_log_path=None, session_dir=None):
        """
        분석기 초기화
        
        Args:
            state_reward_log_path: ZMQ를 통해 수집된 상태/보상 로그 파일 경로
            model_dir: PPO 모델 디렉토리 경로 (선택적)
            ppo_log_path: PPO 학습 로그 파일 경로 (선택적)
            session_dir: 결과물이 저장될 세션 디렉토리 (선택적)
        """
        self.state_reward_log_path = state_reward_log_path
        self.model_dir = model_dir
        self.ppo_log_path = ppo_log_path
        self.session_dir = session_dir
        
        # 데이터 저장소
        self.zmq_data = None
        self.ppo_log_data = None
        self.model_info = None
        
        # 데이터 로드
        self._load_zmq_data()
        if ppo_log_path:
            self._load_ppo_log()
        if model_dir:
            self._load_model_info()
    
    def _load_zmq_data(self):
        """ZMQ 상태/보상 데이터 로드"""
        try:
            print(f"ZMQ 데이터 파일 로드 중: {self.state_reward_log_path}")
            
            # 먼저 바이너리 모드로 파일 읽기 시도
            with open(self.state_reward_log_path, 'rb') as f:
                binary_content = f.read()
                
            # 여러 인코딩 방식 시도
            encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            content = None
            
            for encoding in encodings_to_try:
                try:
                    content = binary_content.decode(encoding, errors='replace')
                    print(f"파일을 '{encoding}' 인코딩으로 성공적으로 로드했습니다.")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # 마지막 시도로 latin1 인코딩 사용 (항상 성공함)
                content = binary_content.decode('latin1')
                print(f"모든 인코딩 시도 실패. 'latin1'으로 대체하여 로드했습니다.")
            
            # 나머지 코드는 동일하게 유지
            # JSON 배열의 끝을 찾아서 적절하게 처리
            if not content.strip().endswith("]"):
                print("경고: JSON 파일이 올바르게 닫히지 않았습니다. 파일을 수정합니다.")
                
                # 마지막 쉼표 처리
                if content.strip().endswith(","):
                    content = content.rstrip().rstrip(",")
                
                # 배열 닫기 추가
                content += "\n]"
            
            # 수정된 내용으로 JSON 파싱 시도
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print("대체 방법으로 라인별 파싱을 시도합니다...")
                
                # 라인별 파싱 시도
                data = []
                with open(self.state_reward_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 첫 줄과 마지막 줄은 배열 시작/끝을 나타내므로 제외
                start_idx = 0
                end_idx = len(lines)
                
                if lines[0].strip() == "[":
                    start_idx = 1
                
                if lines[-1].strip() in ["]", "],", "}]"]:
                    end_idx = -1
                
                for line in lines[start_idx:end_idx]:
                    line = line.strip()
                    if not line or line in ["[", "]", "},", "}"]:
                        continue
                    
                    # 라인의 끝에 있는 쉼표 제거
                    if line.endswith(","):
                        line = line[:-1]
                    
                    # 객체가 완전하지 않은 경우 닫기 괄호 추가
                    if not line.endswith("}"):
                        if "{" in line and "}" not in line:
                            line += "}"
                    
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        print(f"경고: 다음 라인을 파싱할 수 없습니다: {line[:50]}...")
            
            # 데이터가 비어 있는지 확인
            if not data:
                print("경고: 로드된 데이터가 없습니다.")
                self.zmq_data = pd.DataFrame()
                self.zmq_data_filtered = pd.DataFrame()
                return
            
            # 데이터프레임으로 변환
            df_list = []
            for entry in data:
                try:
                    # state와 action을 단일 값 또는 리스트에 따라 처리
                    state_val = entry.get('state', [0.0])
                    if isinstance(state_val, list):
                        state_str = ','.join(map(str, state_val))
                    else:
                        state_str = str(state_val)
                    
                    action_val = entry.get('action', [0.0])
                    if isinstance(action_val, list):
                        action_str = ','.join(map(str, action_val))
                    else:
                        action_str = str(action_val)
                    
                    # 기본 정보 추출
                    row = {
                        'timestamp': entry.get('timestamp', 0),
                        'uptime_ms': entry.get('uptime_ms', 0),
                        'state': state_str,
                        'reward': entry.get('reward', 0.0),
                        'action': action_str,
                        'msg_id': entry.get('msg_id', 0),
                        'type': entry.get('type', 'data')
                    }
                    
                    # state 차원 확인 및 개별 열 추가
                    if isinstance(state_val, list):
                        for i, val in enumerate(state_val):
                            row[f'state_{i}'] = val
                    else:
                        row['state_0'] = state_val
                    
                    # action 차원 확인 및 개별 열 추가
                    if isinstance(action_val, list):
                        for i, val in enumerate(action_val):
                            row[f'action_{i}'] = val
                    else:
                        row['action_0'] = action_val
                    
                    df_list.append(row)
                except Exception as e:
                    print(f"항목 처리 중 오류 발생: {e}, 항목 건너뜀: {entry}")
            
            # 데이터프레임 생성
            if df_list:
                self.zmq_data = pd.DataFrame(df_list)
                
                # connection_test 타입 제외하고 데이터만 필터링
                self.zmq_data_filtered = self.zmq_data[self.zmq_data['type'] != 'connection_test'].copy()
                
                # 시간 기준 정렬
                self.zmq_data_filtered.sort_values('timestamp', inplace=True)
                
                # 시간 변환
                try:
                    self.zmq_data_filtered['datetime'] = pd.to_datetime(self.zmq_data_filtered['timestamp'], unit='ms')
                except Exception as e:
                    print(f"시간 변환 중 오류: {e}, datetime 열 생성을 건너뜁니다.")
                
                print(f"ZMQ 데이터 로드 완료: {len(self.zmq_data_filtered)} 개 데이터 포인트")
                
                # 상태 및 액션 차원 계산
                self._analyze_dimensions()
            else:
                print("유효한 데이터를 찾을 수 없습니다.")
                self.zmq_data = pd.DataFrame()
                self.zmq_data_filtered = pd.DataFrame()
            
        except Exception as e:
            print(f"ZMQ 데이터 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self.zmq_data = pd.DataFrame()
            self.zmq_data_filtered = pd.DataFrame()
    
    def _analyze_dimensions(self):
        """상태 및 액션 차원 분석"""
        if self.zmq_data_filtered.empty:
            self.state_dim = 0
            self.action_dim = 0
            return
        
        # 상태 차원 찾기
        state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
        self.state_dim = len(state_cols)
        
        # 액션 차원 찾기
        action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
        self.action_dim = len(action_cols)
        
        print(f"상태 차원: {self.state_dim}, 액션 차원: {self.action_dim}")

    def analyze_architecture_metrics(self):
        """건축 설계 최적화 관련 지표 분석"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return None, None

        # 상태 데이터 추출
        state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
        if len(state_cols) < 3:
            print("건축 지표 분석에 필요한 상태 차원이 부족합니다.")
            return None, None

        # 건축 지표 설정
        bcr_col = 'state_0'  # 건폐율
        far_col = 'state_1'  # 용적률
        sunlight_col = 'state_2'  # 일조량

        # 법적 제한
        bcr_limit = 0.6  # 60%
        far_limit = 4.0  # 400%

        # 지표 계산
        metrics = {
            'bcr_avg': self.zmq_data_filtered[bcr_col].mean(),
            'far_avg': self.zmq_data_filtered[far_col].mean(),
            'sunlight_avg': self.zmq_data_filtered[sunlight_col].mean(),
            'bcr_max': self.zmq_data_filtered[bcr_col].max(),
            'far_max': self.zmq_data_filtered[far_col].max(),
            'sunlight_max': self.zmq_data_filtered[sunlight_col].max(),
            'bcr_violations': (self.zmq_data_filtered[bcr_col] > bcr_limit).sum(),
            'bcr_violation_rate': (self.zmq_data_filtered[bcr_col] > bcr_limit).mean() * 100,
            'far_violations': (self.zmq_data_filtered[far_col] > far_limit).sum(),
            'far_violation_rate': (self.zmq_data_filtered[far_col] > far_limit).mean() * 100,
        }

        # 상관관계 분석
        corr_matrix = self.zmq_data_filtered[[bcr_col, far_col, sunlight_col, 'reward']].corr()

        # 결과 출력
        print("\n=== 건축 설계 지표 분석 ===")
        print(f"건폐율(BCR) 평균: {metrics['bcr_avg']*100:.2f}% (최대: {metrics['bcr_max']*100:.2f}%)")
        print(f"용적률(FAR) 평균: {metrics['far_avg']*100:.2f}% (최대: {metrics['far_max']*100:.2f}%)")
        print(f"일조량 평균: {metrics['sunlight_avg']:.3f} (최대: {metrics['sunlight_max']:.3f})")
        print(f"건폐율 법적 제한({bcr_limit*100:.0f}%) 위반: {metrics['bcr_violations']}회 ({metrics['bcr_violation_rate']:.1f}%)")
        print(f"용적률 법적 제한({far_limit*100:.0f}%) 위반: {metrics['far_violations']}회 ({metrics['far_violation_rate']:.1f}%)")

        print("\n상관관계 분석:")
        print(corr_matrix)

        # 시각화
        plt.figure(figsize=(12, 10))

        # 1. 건폐율과 용적률의 산점도 (일조량으로 색상 표시)
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(
            self.zmq_data_filtered[bcr_col] * 100,
            self.zmq_data_filtered[far_col] * 100,
            c=self.zmq_data_filtered[sunlight_col],
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='일조량')
        plt.axvline(x=bcr_limit*100, color='r', linestyle='--', label=f'BCR 제한({bcr_limit*100:.0f}%)')
        plt.axhline(y=far_limit*100, color='r', linestyle='--', label=f'FAR 제한({far_limit*100:.0f}%)')
        plt.xlabel('건폐율(BCR) %')
        plt.ylabel('용적률(FAR) %')
        plt.title('건폐율과 용적률의 관계')
        plt.legend()
        plt.grid(True)

        # 2. 건폐율과 보상의 관계
        plt.subplot(2, 2, 2)
        plt.scatter(self.zmq_data_filtered[bcr_col] * 100, self.zmq_data_filtered['reward'], alpha=0.5)
        plt.axvline(x=bcr_limit*100, color='r', linestyle='--', label=f'BCR 제한({bcr_limit*100:.0f}%)')
        plt.xlabel('건폐율(BCR) %')
        plt.ylabel('보상')
        plt.title('건폐율과 보상의 관계')
        plt.legend()
        plt.grid(True)

        # 3. 용적률과 보상의 관계
        plt.subplot(2, 2, 3)
        plt.scatter(self.zmq_data_filtered[far_col] * 100, self.zmq_data_filtered['reward'], alpha=0.5)
        plt.axvline(x=far_limit*100, color='r', linestyle='--', label=f'FAR 제한({far_limit*100:.0f}%)')
        plt.xlabel('용적률(FAR) %')
        plt.ylabel('보상')
        plt.title('용적률과 보상의 관계')
        plt.legend()
        plt.grid(True)

        # 4. 일조량과 보상의 관계
        plt.subplot(2, 2, 4)
        plt.scatter(self.zmq_data_filtered[sunlight_col], self.zmq_data_filtered['reward'], alpha=0.5)
        plt.xlabel('일조량')
        plt.ylabel('보상')
        plt.title('일조량과 보상의 관계')
        plt.grid(True)

        plt.tight_layout()

        # 저장 경로
        save_path = os.path.join(self.session_dir, "architecture_metrics_analysis.png") if self.session_dir else None

        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"건축 지표 분석 그래프 저장됨: {save_path}")

        return plt.gcf(), metrics

    
    def _load_ppo_log(self):
        """PPO 학습 로그 파일 로드 (CSV 형식)"""
        try:
            self.ppo_log_data = pd.read_csv(self.ppo_log_path)
            print(f"PPO 로그 데이터 로드 완료: {len(self.ppo_log_data)} 개 데이터 포인트")
        except Exception as e:
            print(f"PPO 로그 데이터 로드 중 오류 발생: {e}")
            self.ppo_log_data = pd.DataFrame()
    
    def _load_model_info(self):
        """PPO 모델 정보 로드"""
        try:
            model_info = {}
            
            # 버전 정보 확인
            version_file = os.path.join(self.model_dir, "_stable_baselines3_version")
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    model_info['sb3_version'] = f.read().strip()
            
            # 모델 메타데이터 확인
            data_file = os.path.join(self.model_dir, "data")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r') as f:
                        data_content = f.read()
                    
                    # 주요 정보 추출 (정규식 사용)
                    model_info['policy_class'] = re.search(r'"policy_class": {.*?"__module__": "([^"]+)"', data_content, re.DOTALL)
                    if model_info['policy_class']:
                        model_info['policy_class'] = model_info['policy_class'].group(1)
                    
                    # 학습 파라미터 추출
                    for param in ['learning_rate', 'gamma', 'gae_lambda', 'n_steps', 'ent_coef', 'batch_size', 'n_epochs']:
                        match = re.search(fr'"{param}": ([\d\.]+)', data_content)
                        if match:
                            model_info[param] = float(match.group(1))
                    
                    # observation_space 및 action_space 추출
                    for space in ['observation_space', 'action_space']:
                        shape_match = re.search(fr'"{space}".*?"_shape": \[([\d,\s]+)\]', data_content, re.DOTALL)
                        if shape_match:
                            shape_str = shape_match.group(1)
                            model_info[f'{space}_shape'] = [int(dim) for dim in shape_str.split(',') if dim.strip()]
                except:
                    print("데이터 파일 파싱 중 오류 발생")
            
            self.model_info = model_info
            print(f"모델 정보 로드 완료")
            
        except Exception as e:
            print(f"모델 정보 로드 중 오류 발생: {e}")
            self.model_info = {}
    
    def analyze_reward_distribution(self, save_path=None):
        """보상 분포 분석 및 시각화"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.zmq_data_filtered['reward'], bins=30, kde=True)
        plt.title('Reward Distribution')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        # 통계값 계산
        reward_stats = {
            'mean': self.zmq_data_filtered['reward'].mean(),
            'median': self.zmq_data_filtered['reward'].median(),
            'std': self.zmq_data_filtered['reward'].std(),
            'min': self.zmq_data_filtered['reward'].min(),
            'max': self.zmq_data_filtered['reward'].max(),
        }
        
        print("\n=== 보상 통계 ===")
        for key, value in reward_stats.items():
            print(f"{key}: {value:.4f}")
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"보상 분포 그래프 저장됨: {save_path}")
        
        return plt.gcf(), reward_stats
    
    def analyze_action_impact(self, save_path=None):
        """액션이 상태 및 보상에 미치는 영향 분석"""
        if self.zmq_data_filtered.empty or self.action_dim == 0:
            print("분석할 데이터가 없거나 액션 차원이 0입니다.")
            return None
        
        # 액션 별 분석을 위한 준비
        action_cols = [f'action_{i}' for i in range(self.action_dim)]
        
        # 각 액션 차원별 상관관계 계산
        correlation_data = []
        for i, action_col in enumerate(action_cols):
            if action_col not in self.zmq_data_filtered.columns:
                continue
                
            # 보상과의 상관관계
            reward_corr = self.zmq_data_filtered[[action_col, 'reward']].corr().iloc[0, 1]
            correlation_data.append({
                'action_dim': i,
                'reward_correlation': reward_corr,
            })
            
            # 각 상태 차원과의 상관관계
            for j in range(self.state_dim):
                state_col = f'state_{j}'
                if state_col in self.zmq_data_filtered.columns:
                    state_corr = self.zmq_data_filtered[[action_col, state_col]].corr().iloc[0, 1]
                    correlation_data[-1][f'state_{j}_correlation'] = state_corr
        
        # 상관관계 테이블 생성
        correlation_df = pd.DataFrame(correlation_data)
        
        # 각 액션 차원별 상관관계 시각화
        if not correlation_df.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_df.set_index('action_dim').T, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Between Actions and States/Rewards')
            plt.tight_layout()
            
            print("\n=== 액션 영향 분석 ===")
            print(correlation_df)
            
            # 그래프 저장
            if save_path:
                plt.savefig(save_path)
                print(f"액션 영향 그래프 저장됨: {save_path}")
            
            return plt.gcf(), correlation_df
        
        return None, None
    
    def analyze_state_trends(self, save_path=None):
        """상태 변화 추세 분석"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        # 상태 차원별 시계열 플롯
        plt.figure(figsize=(12, 4 * min(self.state_dim, 3)))
        
        for i in range(min(self.state_dim, 3)):  # 최대 3개 차원까지만 표시
            state_col = f'state_{i}'
            if state_col in self.zmq_data_filtered.columns:
                plt.subplot(min(self.state_dim, 3), 1, i+1)
                plt.plot(self.zmq_data_filtered['datetime'], self.zmq_data_filtered[state_col])
                plt.title(f'State Dimension {i} Trend')
                plt.xlabel('Time')
                plt.ylabel(f'State Value')
                plt.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"상태 변화 그래프 저장됨: {save_path}")
        
        return plt.gcf()
    
    def find_optimal_designs(self, top_n=5):
        """최적 디자인 찾기 (최고 보상 기준)"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 보상 기준 상위 N개 데이터 추출
        top_designs = self.zmq_data_filtered.sort_values('reward', ascending=False).head(top_n).copy()
        
        # 결과 출력용 칼럼 선택
        result_cols = ['msg_id', 'reward', 'datetime']
        
        # 상태와 액션 칼럼 추가
        state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
        action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
        
        result_cols += state_cols + action_cols
        
        # 결과 정리
        result_df = top_designs[result_cols].reset_index(drop=True)
        
        print("\n=== 최적 디자인 ===")
        print(f"상위 {top_n}개 디자인 (보상 기준):")
        print(result_df)
        
        return result_df
    
    def cluster_designs(self, n_clusters=3, save_path=None):
        """디자인 클러스터링 (K-means)"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return None, None
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 클러스터링을 위한 특징 선택
            state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
            if not state_cols:
                print("클러스터링을 위한 상태 데이터가 없습니다.")
                return None, None
            
            # 데이터 전처리
            X = self.zmq_data_filtered[state_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            
            if X.empty:
                print("전처리 후 데이터가 없습니다.")
                return None, None
            
            # 데이터 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # 결과 데이터프레임에 클러스터 ID 추가
            cluster_df = self.zmq_data_filtered.loc[X.index].copy()
            cluster_df['cluster'] = clusters
            
            # 클러스터별 통계
            cluster_stats = cluster_df.groupby('cluster').agg({
                'reward': ['mean', 'std', 'min', 'max', 'count']
            })
            
            print("\n=== 디자인 클러스터링 결과 ===")
            print(cluster_stats)
            
            # 클러스터별 최고 보상 인덱스 찾기
            best_designs = []
            for i in range(n_clusters):
                cluster_data = cluster_df[cluster_df['cluster'] == i]
                if not cluster_data.empty:
                    best_idx = cluster_data['reward'].idxmax()
                    best_designs.append(cluster_data.loc[best_idx])
            
            best_designs_df = pd.DataFrame(best_designs)
            
            # 클러스터별 시각화 (첫 두 개 상태 차원 사용)
            if len(state_cols) >= 2:
                plt.figure(figsize=(10, 8))
                
                if 'reward' in cluster_df.columns:
                    scatter = plt.scatter(
                        cluster_df[state_cols[0]], 
                        cluster_df[state_cols[1]], 
                        c=clusters, 
                        s=cluster_df['reward'] * 10,  # 보상 값에 따라 크기 조정
                        cmap='viridis', 
                        alpha=0.6
                    )
                else:
                    scatter = plt.scatter(
                        cluster_df[state_cols[0]], 
                        cluster_df[state_cols[1]], 
                        c=clusters, 
                        cmap='viridis', 
                        alpha=0.6
                    )
                
                plt.colorbar(scatter, label='Cluster')
                plt.xlabel(state_cols[0])
                plt.ylabel(state_cols[1])
                plt.title('Design Clusters in State Space')
                plt.grid(True)
                
                # 각 클러스터의 최적 디자인 표시
                for idx, row in best_designs_df.iterrows():
                    if state_cols[0] in row and state_cols[1] in row:
                        plt.scatter(
                            row[state_cols[0]], 
                            row[state_cols[1]], 
                            s=200, 
                            marker='*', 
                            color='red', 
                            edgecolor='black', 
                            label=f'Cluster {row["cluster"]} Best' if idx == 0 else ""
                        )
                
                plt.legend()
                plt.tight_layout()
                
                # 그래프 저장
                if save_path:
                    plt.savefig(save_path)
                    print(f"클러스터 그래프 저장됨: {save_path}")
                
                print("\n=== 클러스터별 최적 디자인 ===")
                print(best_designs_df[['cluster', 'reward'] + state_cols + [col for col in best_designs_df.columns if col.startswith('action_')]])
                
                return plt.gcf(), best_designs_df
            
            return None, best_designs_df
            
        except ImportError:
            print("클러스터링을 위해 scikit-learn 라이브러리가 필요합니다.")
            return None, None
        except Exception as e:
            print(f"클러스터링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_processed_data(self, output_path):
        """처리된 데이터 저장"""
        if self.zmq_data_filtered.empty:
            print("저장할 데이터가 없습니다.")
            return
        
        try:
            # 필요한 열만 선택
            cols_to_save = ['msg_id', 'timestamp', 'datetime', 'reward']
            
            # 상태와 액션 열 추가
            state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
            action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
            
            cols_to_save += state_cols + action_cols
            
            # 열이 존재하는지 확인
            existing_cols = [col for col in cols_to_save if col in self.zmq_data_filtered.columns]
            
            # 저장 폴더 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 데이터 저장
            self.zmq_data_filtered[existing_cols].to_csv(output_path, index=False)
            print(f"처리된 데이터가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")
    
    def generate_rlhf_reference_data(self, output_path):
        """RLHF 기준 데이터 생성"""
        if self.zmq_data_filtered.empty:
            print("기준 데이터 생성을 위한 데이터가 없습니다.")
            return
        
        try:
            # 상위 보상 기준 데이터 추출
            top_n = min(20, len(self.zmq_data_filtered))
            top_designs = self.zmq_data_filtered.sort_values('reward', ascending=False).head(top_n).copy()
            
            # 클러스터링 기반 다양한 디자인 추출
            diverse_designs = None
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # 상태 열 추출
                state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
                
                if state_cols:
                    # 데이터 전처리
                    X = self.zmq_data_filtered[state_cols].copy()
                    X = X.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not X.empty:
                        # 데이터 스케일링
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # K-means 클러스터링 (최대 5개 클러스터)
                        n_clusters = min(5, len(X) // 10) if len(X) >= 10 else 1
                        if n_clusters > 1:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(X_scaled)
                            
                            # 클러스터 정보 추가
                            cluster_df = self.zmq_data_filtered.loc[X.index].copy()
                            cluster_df['cluster'] = clusters
                            
                            # 각 클러스터에서 최고 보상 인덱스 찾기
                            diverse_designs = []
                            for i in range(n_clusters):
                                cluster_data = cluster_df[cluster_df['cluster'] == i]
                                if not cluster_data.empty:
                                    best_idx = cluster_data['reward'].idxmax()
                                    diverse_designs.append(cluster_data.loc[best_idx])
                            
                            diverse_designs = pd.DataFrame(diverse_designs)
            except:
                print("클러스터링 처리 중 오류가 발생했습니다. 클러스터링 없이 계속합니다.")
            
            # 기준 데이터 생성
            reference_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_samples': len(self.zmq_data_filtered),
                    'state_dimensions': self.state_dim,
                    'action_dimensions': self.action_dim,
                },
                'top_designs': [],
                'diverse_designs': []
            }
            
            # 상태와 액션 열 정의
            state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
            action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
            
            # 최고 보상 디자인 데이터 추가
            for _, row in top_designs.iterrows():
                design_data = {
                    'msg_id': int(row['msg_id']) if 'msg_id' in row else 0,
                    'reward': float(row['reward']) if 'reward' in row else 0.0,
                    'state': [float(row[col]) for col in state_cols if col in row],
                    'action': [float(row[col]) for col in action_cols if col in row],
                    'timestamp': int(row['timestamp']) if 'timestamp' in row else 0
                }
                reference_data['top_designs'].append(design_data)
            
            # 다양한 디자인 데이터 추가
            if diverse_designs is not None and not diverse_designs.empty:
                for _, row in diverse_designs.iterrows():
                    design_data = {
                        'msg_id': int(row['msg_id']) if 'msg_id' in row else 0,
                        'reward': float(row['reward']) if 'reward' in row else 0.0,
                        'state': [float(row[col]) for col in state_cols if col in row],
                        'action': [float(row[col]) for col in action_cols if col in row],
                        'timestamp': int(row['timestamp']) if 'timestamp' in row else 0,
                        'cluster': int(row['cluster']) if 'cluster' in row else 0
                    }
                    reference_data['diverse_designs'].append(design_data)
            
            # 저장 폴더 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON 파일로 저장
            with open(output_path, 'w') as f:
                json.dump(reference_data, f, indent=2)
            
            print(f"RLHF 기준 데이터가 {output_path}에 저장되었습니다.")
            print(f"- 최고 보상 디자인: {len(reference_data['top_designs'])}개")
            print(f"- 다양한 디자인: {len(reference_data['diverse_designs'])}개")
            
            return reference_data
            
        except Exception as e:
            print(f"RLHF 기준 데이터 생성 중 오류 발생: {e}")
            return None
    
    def interactive_design_browser(self):
        """대화형 디자인 브라우저 (IPython 환경에서 사용)"""
        try:
            from ipywidgets import interact, widgets
            import IPython.display as display
            
            # 데이터 존재 확인
            if self.zmq_data_filtered.empty:
                print("브라우징할 데이터가 없습니다.")
                return
            
            # 상태 및 액션 열 가져오기
            state_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('state_')]
            action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
            
            # 메시지 ID 목록
            message_ids = self.zmq_data_filtered['msg_id'].tolist()
            
            # 디자인 세부 정보 표시 함수
            def show_design_details(msg_id):
                # 선택한 메시지 ID에 해당하는 데이터 찾기
                design_data = self.zmq_data_filtered[self.zmq_data_filtered['msg_id'] == msg_id]
                
                if design_data.empty:
                    print(f"메시지 ID {msg_id}에 해당하는 데이터를 찾을 수 없습니다.")
                    return
                
                # 기본 정보 표시
                print(f"===== 디자인 세부 정보 (메시지 ID: {msg_id}) =====")
                print(f"타임스탬프: {design_data['timestamp'].values[0]}")
                if 'datetime' in design_data.columns:
                    print(f"날짜/시간: {design_data['datetime'].values[0]}")
                
                print(f"\n보상: {design_data['reward'].values[0]:.4f}")
                
                # 상태 값 표시
                print("\n상태 값:")
                for col in state_cols:
                    if col in design_data.columns:
                        print(f"  {col}: {design_data[col].values[0]:.4f}")
                
                # 액션 값 표시
                print("\n액션 값:")
                for col in action_cols:
                    if col in design_data.columns:
                        print(f"  {col}: {design_data[col].values[0]:.4f}")
            
            # 대화형 위젯 생성
            msg_id_dropdown = widgets.Dropdown(
                options=message_ids,
                description='메시지 ID:',
                style={'description_width': 'initial'}
            )
            
            # 대화형 함수 연결
            interact(show_design_details, msg_id=msg_id_dropdown)
            
        except ImportError:
            print("대화형 브라우저를 위해 ipywidgets 라이브러리가 필요합니다.")
            
            # 대체 메서드: 최고 보상 상위 5개 디자인 정보 출력
            top_designs = self.find_optimal_designs(5)
            for i, (_, row) in enumerate(top_designs.iterrows()):
                print(f"\n===== 디자인 {i+1} (메시지 ID: {row['msg_id']}) =====")
                print(f"보상: {row['reward']:.4f}")
                
                # 상태 값 표시
                print("\n상태 값:")
                for col in [c for c in row.index if c.startswith('state_')]:
                    print(f"  {col}: {row[col]:.4f}")
                
                # 액션 값 표시
                print("\n액션 값:")
                for col in [c for c in row.index if c.startswith('action_')]:
                    print(f"  {col}: {row[col]:.4f}")

def find_latest_file(directory, pattern, subdirectory=None):
    """
    지정된 폴더에서 패턴과 일치하는 가장 최신 파일을 찾습니다.
    
    Args:
        directory (str): 검색할 디렉토리
        pattern (str): 파일명 패턴 (glob 패턴)
        subdirectory (str, optional): 하위 디렉토리 (있는 경우)
    
    Returns:
        str or None: 찾은 파일의 경로 또는 파일이 없을 경우 None
    """
    import glob
    
    search_dir = os.path.join(directory, subdirectory) if subdirectory else directory
    files = glob.glob(os.path.join(search_dir, pattern))
    
    if not files:
        return None
    
    # 최신 파일 찾기 (수정 시간 기준)
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def find_latest_model(directory):
    """
    지정된 디렉토리에서 가장 최신 PPO 모델 파일(ZIP)을 찾습니다.
    중단된 모델(interrupted)과 일반 모델 모두 찾아서 가장 최신 파일을 반환합니다.
    
    Args:
        directory (str): 검색할 디렉토리
        
    Returns:
        tuple: (모델 파일 경로, 모델 디렉토리 경로)
    """
    # 일반 모델 파일 검색
    regular_model = find_latest_file(directory, "ppo_*model*.zip")
    
    # 중단된 모델 파일 검색
    interrupted_model = find_latest_file(directory, "ppo_*interrupted*.zip")
    
    # 체크포인트 모델 파일 검색
    checkpoint_model = find_latest_file(directory, "ppo_*checkpoint*.zip")
    
    # 존재하는 모델 중 가장 최신 파일 선택
    models = [m for m in [regular_model, interrupted_model, checkpoint_model] if m]
    
    if not models:
        return None, directory
    
    latest_model = max(models, key=os.path.getmtime)
    
    # 모델이 있는 디렉토리 반환 (압축 해제에 사용)
    model_dir = os.path.dirname(latest_model)
    if not model_dir:
        model_dir = "."
    
    return latest_model, model_dir

def create_session_directory(base_dir, session_name=None):
    """
    세션 디렉토리를 생성합니다.
    
    Args:
        base_dir (str): 기본 디렉토리
        session_name (str, optional): 세션 이름. 제공되지 않으면 타임스탬프 사용
        
    Returns:
        str: 생성된 세션 디렉토리 경로
    """
    # 타임스탬프 생성
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # 세션 이름이 제공되지 않으면 타임스탬프 사용
    if session_name:
        # 이름에 시간 정보 추가 (고유성 확보)
        dir_name = f"{session_name}_{timestamp}"
    else:
        dir_name = f"session_{timestamp}"
    
    # 디렉토리 경로 생성
    session_dir = os.path.join(base_dir, dir_name)
    
    # 디렉토리 생성
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 데이터 통합 및 분석 도구')
    parser.add_argument('--state-reward-log', type=str, default=None,
                        help='ZMQ를 통해 수집된 상태/보상 로그 파일 경로 (지정하지 않으면 최신 파일 자동 사용)')
    parser.add_argument('--model-file', type=str, default=None,
                        help='PPO 모델 파일 경로 (ZIP, 지정하지 않으면 최신 파일 자동 사용)')
    parser.add_argument('--model-dir', type=str, default='.',
                        help='PPO 모델 디렉토리 경로 (기본값: 현재 폴더)')
    parser.add_argument('--ppo-log', type=str, default=None,
                        help='PPO 학습 로그 파일 경로 (선택적)')
    parser.add_argument('--log-dir', type=str, default='zmq_logs',
                        help='ZMQ 로그 파일 디렉토리 (기본값: zmq_logs)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='결과 파일 저장 디렉토리 (기본값: analysis_results)')
    parser.add_argument('--session-name', type=str, default=None,
                        help='결과물을 저장할 세션 이름 (기본값: 타임스탬프 사용)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='분석만 수행하고 기준 데이터를 생성하지 않음')
    parser.add_argument('--fix-json', action='store_true',
                        help='손상된 JSON 파일 수정 시도')
    
    args = parser.parse_args()
    
    # 프로젝트 루트 디렉토리 경로 계산 (수정된 부분)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 출력 디렉토리 생성
    args.output_dir = os.path.join(project_root, 'data')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 세션 디렉토리 생성
    session_dir = create_session_directory(args.output_dir, args.session_name)
    print(f"\n🔹 세션 디렉토리 생성됨: {session_dir}")
    
    # 상태/보상 로그 파일 경로 결정 (수정된 부분)
    state_reward_log_path = args.state_reward_log
    if not state_reward_log_path:
        # 프로젝트 루트 기준 data/zmq_logs 디렉토리에서 먼저 찾기
        zmq_logs_dir = os.path.join(project_root, "data", "zmq_logs")
        state_reward_log_path = find_latest_file(zmq_logs_dir, "state_reward_log_*.json")
        
        if not state_reward_log_path:
            # args.log_dir에서 찾기
            state_reward_log_path = find_latest_file(args.log_dir, "state_reward_log_*.json")
            
        if not state_reward_log_path:
            print(f"경고: {zmq_logs_dir} 및 {args.log_dir} 폴더에서 state_reward_log 파일을 찾을 수 없습니다.")
            # 프로젝트 루트에서 찾기
            state_reward_log_path = find_latest_file(project_root, "state_reward_log_*.json")
            
        if not state_reward_log_path:
            print("오류: 상태/보상 로그 파일을 찾을 수 없습니다. --state-reward-log 인자로 직접 지정해주세요.")
            return
    
    print(f"🔹 사용할 상태/보상 로그 파일: {state_reward_log_path}")
    
    # 입력 파일 세션 디렉토리에 복사
    input_copy_path = os.path.join(session_dir, os.path.basename(state_reward_log_path))
    shutil.copy2(state_reward_log_path, input_copy_path)
    print(f"🔹 입력 파일이 세션 디렉토리에 복사됨: {input_copy_path}")
    
    # 손상된 JSON 파일 수정 시도 (--fix-json 옵션 사용 시)
    if args.fix_json:
        try:
            # 원본 파일 백업
            backup_file = os.path.join(session_dir, f"{os.path.basename(state_reward_log_path)}.bak")
            shutil.copy2(state_reward_log_path, backup_file)
            print(f"🔹 원본 파일이 {backup_file}에 백업되었습니다.")
            
            # 파일 내용 읽기
            with open(state_reward_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # JSON 수정 작업
            if not content.strip().endswith("]"):
                print("🔹 JSON 파일 수정 중...")
                
                # 마지막 쉼표 제거
                if content.strip().endswith(","):
                    content = content.rstrip().rstrip(",")
                
                # 배열 닫기 추가
                content += "\n]"
                
                # 수정된 내용 저장
                with open(state_reward_log_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("🔹 JSON 파일이 수정되었습니다.")
        except Exception as e:
            print(f"❌ JSON 파일 수정 중 오류 발생: {e}")
    
    # 모델 파일 및 디렉토리 경로 결정
    model_dir = args.model_dir
    if args.model_file:
        model_file = args.model_file
        # 모델 파일이 지정된 경우 해당 디렉토리를 모델 디렉토리로 사용
        model_dir = os.path.dirname(model_file)
        if not model_dir:
            model_dir = "."
    else:
        # 최신 모델 파일 찾기 (프로젝트 루트 기준 수정)
        python_modules_dir = os.path.join(project_root, "python_modules")
        model_file, model_dir = find_latest_model(python_modules_dir)
        if model_file:
            print(f"🔹 사용할 모델 파일: {model_file}")
        else:
            print(f"⚠️ 경고: {python_modules_dir} 폴더에서 PPO 모델 파일을 찾을 수 없습니다.")
    
    # 파일 경로 설정
    processed_data_file = os.path.join(session_dir, "processed_rlhf_data.csv")
    reference_data_file = os.path.join(session_dir, "rlhf_reference_data.json")
    reward_plot_path = os.path.join(session_dir, "reward_distribution.png")
    action_plot_path = os.path.join(session_dir, "action_impact.png")
    state_plot_path = os.path.join(session_dir, "state_trends.png")
    cluster_plot_path = os.path.join(session_dir, "design_clusters.png")
    summary_file = os.path.join(session_dir, "analysis_summary.txt")
    
    # 분석기 생성
    analyzer = RLHFDataAnalyzer(
        state_reward_log_path=state_reward_log_path,
        model_dir=model_dir,
        ppo_log_path=args.ppo_log,
        session_dir=session_dir
    )
    
    # 데이터가 있는지 확인
    if analyzer.zmq_data_filtered.empty:
        print("\n❌ 오류: 분석할 데이터가 없습니다.")
        print("다음 사항을 확인해 보세요:")
        print("1. JSON 파일이 올바른 형식인지 확인")
        print("2. --fix-json 옵션을 사용하여 손상된 JSON 파일 수정 시도")
        print("3. 수동으로 JSON 파일을 열어 형식을 확인 및 수정")
        
        # 오류 정보를 세션 디렉토리에 기록
        with open(os.path.join(session_dir, "error_log.txt"), 'w') as f:
            f.write("분석할 데이터가 없습니다.\n")
            f.write(f"로그 파일: {state_reward_log_path}\n")
            f.write(f"시간: {datetime.now().isoformat()}\n")
        
        return
    
    # 데이터 분석 수행
    print("\n1. 보상 분포 분석")
    reward_result = analyzer.analyze_reward_distribution(reward_plot_path)
    
    print("\n2. 액션 영향 분석")
    action_result = analyzer.analyze_action_impact(action_plot_path)
    
    print("\n3. 상태 변화 추세 분석")
    state_fig = analyzer.analyze_state_trends(state_plot_path)
    
    print("\n4. 최적 디자인 찾기")
    top_designs = analyzer.find_optimal_designs()
    
    print("\n5. 디자인 클러스터링")
    try:
        cluster_result = analyzer.cluster_designs(save_path=cluster_plot_path)
        if isinstance(cluster_result, tuple) and len(cluster_result) == 2:
            cluster_fig, cluster_designs = cluster_result
    except Exception as e:
        print(f"❌ 클러스터링 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        cluster_fig, cluster_designs = None, None
    
    # 처리된 데이터 저장
    analyzer.save_processed_data(processed_data_file)
    print(f"🔹 처리된 데이터 저장됨: {processed_data_file}")
    
    # 분석만 수행하는 경우 종료
    if args.analyze_only:
        print("\n🔹 분석 완료. 기준 데이터 생성을 건너뜁니다.")
    else:
        # RLHF 기준 데이터 생성
        print("\n6. RLHF 기준 데이터 생성")
        try:
            reference_data = analyzer.generate_rlhf_reference_data(reference_data_file)
            print(f"🔹 RLHF 기준 데이터 저장됨: {reference_data_file}")
        except Exception as e:
            print(f"❌ 기준 데이터 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
                        
    # 건축 설계 지표 분석 (기준 데이터 생성 후)
    if not args.analyze_only and reference_data is not None:
        print("\n6. 건축 설계 지표 분석")
        try:
            arch_fig, arch_metrics = analyzer.analyze_architecture_metrics()
            # 분석 요약에 결과 추가
            if arch_metrics:
                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write("\n\n=== 건축 설계 지표 분석 ===\n")
                    f.write(f"건폐율(BCR) 평균: {arch_metrics['bcr_avg']*100:.2f}% (최대: {arch_metrics['bcr_max']*100:.2f}%)\n")
                    f.write(f"용적률(FAR) 평균: {arch_metrics['far_avg']*100:.2f}% (최대: {arch_metrics['far_max']*100:.2f}%)\n")
                    f.write(f"일조량 평균: {arch_metrics['sunlight_avg']:.3f} (최대: {arch_metrics['sunlight_max']:.3f})\n")
                    f.write(f"건폐율 법적 제한(60%) 위반: {arch_metrics['bcr_violations']}회 ({arch_metrics['bcr_violation_rate']:.1f}%)\n")
                    f.write(f"용적률 법적 제한(400%) 위반: {arch_metrics['far_violations']}회 ({arch_metrics['far_violation_rate']:.1f}%)\n")
        except Exception as e:
            print(f"❌ 건축 지표 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 분석 요약 생성
    with open(summary_file, 'w') as f:
        f.write(f"RLHF 데이터 분석 요약\n")
        f.write("=" * 50 + "\n\n")
        
        # 세션 정보
        f.write(f"세션 디렉토리: {session_dir}\n")
        f.write(f"분석 시간: {datetime.now().isoformat()}\n\n")
        
        f.write(f"사용한 로그 파일: {state_reward_log_path}\n")
        f.write(f"사용한 모델 디렉토리: {model_dir}\n\n")
        
        # 분석 결과 요약
        if hasattr(analyzer, 'zmq_data_filtered') and not analyzer.zmq_data_filtered.empty:
            data = analyzer.zmq_data_filtered
            f.write(f"총 데이터 포인트: {len(data)}\n")
            f.write(f"상태 차원: {analyzer.state_dim}\n")
            f.write(f"액션 차원: {analyzer.action_dim}\n\n")
            
            # 보상 통계
            f.write("보상 통계:\n")
            f.write(f"  최소값: {data['reward'].min():.4f}\n")
            f.write(f"  최대값: {data['reward'].max():.4f}\n")
            f.write(f"  평균: {data['reward'].mean():.4f}\n")
            f.write(f"  중앙값: {data['reward'].median():.4f}\n")
            f.write(f"  표준편차: {data['reward'].std():.4f}\n\n")
            
            # 최적 디자인 요약
            if isinstance(top_designs, pd.DataFrame) and not top_designs.empty:
                f.write(f"최적 디자인 (상위 {len(top_designs)}개):\n")
                for idx, row in top_designs.iterrows():
                    f.write(f"  디자인 {idx+1} - 보상: {row['reward']:.4f}, ID: {row['msg_id']}\n")
                    
                    # 상태값도 추가
                    state_cols = [col for col in row.index if col.startswith('state_')]
                    f.write(f"    상태값: ")
                    for col in state_cols:
                        f.write(f"{col}={row[col]:.4f} ")
                    f.write("\n")
                    
                    # 액션값도 추가
                    action_cols = [col for col in row.index if col.startswith('action_')]
                    f.write(f"    액션값: ")
                    for col in action_cols:
                        f.write(f"{col}={row[col]:.4f} ")
                    f.write("\n\n")
            
            # 생성된 파일 목록
            f.write("세션 디렉토리 내 생성된 파일 목록:\n")
            f.write(f"  1. 처리된 데이터: processed_rlhf_data.csv\n")
            if not args.analyze_only:
                f.write(f"  2. 기준 데이터: rlhf_reference_data.json\n")
            f.write(f"  3. 보상 분포 그래프: reward_distribution.png\n")
            f.write(f"  4. 액션 영향 그래프: action_impact.png\n")
            f.write(f"  5. 상태 변화 그래프: state_trends.png\n")
            f.write(f"  6. 클러스터 그래프: design_clusters.png\n")
            f.write(f"  7. 분석 요약: analysis_summary.txt\n")
            f.write(f"  8. 입력 파일 복사본: {os.path.basename(state_reward_log_path)}\n")
        else:
            f.write("분석할 데이터가 없습니다.\n")
    
    print(f"\n🔹 분석 요약이 {summary_file}에 저장되었습니다.")
    print(f"\n✅ 모든 분석 결과가 {session_dir} 디렉토리에 저장되었습니다.")



if __name__ == "__main__":
    main()