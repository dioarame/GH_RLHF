#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 데이터 통합 및 분석 도구 (RLHF 쌍대비교용 업데이트)

이 스크립트는 PPO 학습 결과와 ZMQ를 통해 수집된 상태/보상 데이터를 통합하고 분석합니다.
쌍대비교를 위한 기준점을 제공하고, 웹 인터페이스에서 사용할 수 있는 형태로 데이터를 준비합니다.
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
import uuid

# 시각화 스타일 설정
plt.style.use('ggplot')
sns.set(style="whitegrid")

class RLHFDataAnalyzer:
    """
    강화학습 데이터와 인간 피드백 데이터를 통합 분석하는 클래스
    쌍대비교를 위한 데이터 준비에 특화
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
        
        # RLHF 시스템 설정 (확장된 상태 공간)
        self.state_labels = ["BCR", "FAR", "Winter_Sunlight", "SV_Ratio"]
        self.state_dimensions = len(self.state_labels)
        
        # 건축 설계 제약 조건
        self.design_constraints = {
            'bcr_limit': 0.70,  # 70% 미만
            'far_min': 2.0,     # 200% 이상
            'far_max': 5.0,     # 500% 이하  
            'sunlight_min': 80000,  # 80k kWh 이상
            'sunlight_max': 120000, # 120k kWh 목표
            'svr_optimal': 0.8,     # 0.8 최적
            'svr_range': [0.6, 1.0] # 허용 범위
        }
        
        # 데이터 로드
        self._load_zmq_data()
        if ppo_log_path:
            self._load_ppo_log()
        if model_dir:
            self._load_model_info()
    
    def _load_zmq_data(self):
        """ZMQ 상태/보상 데이터 로드 (RLHF용 개선)"""
        try:
            print(f"ZMQ 데이터 파일 로드 중: {self.state_reward_log_path}")
            
            # 파일 읽기 및 인코딩 처리
            with open(self.state_reward_log_path, 'rb') as f:
                binary_content = f.read()
                
            # 인코딩 시도
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
                content = binary_content.decode('latin1')
                print(f"모든 인코딩 시도 실패. 'latin1'으로 대체하여 로드했습니다.")
            
            # JSON 수정
            if not content.strip().endswith("]"):
                print("경고: JSON 파일이 올바르게 닫히지 않았습니다. 파일을 수정합니다.")
                if content.strip().endswith(","):
                    content = content.rstrip().rstrip(",")
                content += "\n]"
            
            # JSON 파싱
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print("대체 방법으로 라인별 파싱을 시도합니다...")
                data = self._parse_json_lines(content)
            
            if not data:
                print("경고: 로드된 데이터가 없습니다.")
                self.zmq_data = pd.DataFrame()
                self.zmq_data_filtered = pd.DataFrame()
                return
            
            # 데이터프레임 변환 (RLHF용 개선)
            df_list = []
            for entry in data:
                try:
                    # 상태 및 액션 처리
                    state_val = entry.get('state', [0.0] * self.state_dimensions)
                    action_val = entry.get('action', [0.0])
                    
                    # 상태 차원 검증 및 조정
                    if isinstance(state_val, list):
                        if len(state_val) < self.state_dimensions:
                            # 부족한 차원을 0으로 채움
                            state_val.extend([0.0] * (self.state_dimensions - len(state_val)))
                        elif len(state_val) > self.state_dimensions:
                            # 초과 차원 제거
                            state_val = state_val[:self.state_dimensions]
                        state_str = ','.join(map(str, state_val))
                    else:
                        state_val = [float(state_val)] + [0.0] * (self.state_dimensions - 1)
                        state_str = ','.join(map(str, state_val))
                    
                    # 액션 처리
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
                        'type': entry.get('type', 'data'),
                        'design_id': f"design_{entry.get('msg_id', 0)}_{int(entry.get('timestamp', 0))}"
                    }
                    
                    # 상태 차원별 개별 열 추가
                    for i, label in enumerate(self.state_labels):
                        if i < len(state_val):
                            row[f'state_{i}'] = state_val[i]
                            row[label.lower()] = state_val[i]  # 레이블명으로도 접근 가능
                        else:
                            row[f'state_{i}'] = 0.0
                            row[label.lower()] = 0.0
                    
                    # 액션 차원별 개별 열 추가
                    if isinstance(action_val, list):
                        for i, val in enumerate(action_val):
                            row[f'action_{i}'] = val
                    else:
                        row['action_0'] = action_val
                    
                    # 건축 설계 품질 지표 계산
                    row.update(self._calculate_design_quality_metrics(state_val))
                    
                    df_list.append(row)
                    
                except Exception as e:
                    print(f"항목 처리 중 오류 발생: {e}, 항목 건너뜀")
                    continue
            
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
            traceback.print_exc()
            self.zmq_data = pd.DataFrame()
            self.zmq_data_filtered = pd.DataFrame()
    
    def _parse_json_lines(self, content):
        """라인별 JSON 파싱"""
        data = []
        lines = content.split('\n')
        
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
            
            if line.endswith(","):
                line = line[:-1]
            
            if not line.endswith("}"):
                if "{" in line and "}" not in line:
                    line += "}"
            
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print(f"경고: 다음 라인을 파싱할 수 없습니다: {line[:50]}...")
        
        return data
    
    def _calculate_design_quality_metrics(self, state_values):
        """건축 설계 품질 지표 계산"""
        if len(state_values) < self.state_dimensions:
            return {
                'quality_score': 0.0,
                'constraint_violations': 4,
                'legal_compliance': False,
                'sustainability_score': 0.0
            }
        
        bcr = state_values[0]  # Building Coverage Ratio
        far = state_values[1]  # Floor Area Ratio  
        sunlight = state_values[2]  # Winter Sunlight
        svr = state_values[3]  # Surface to Volume Ratio
        
        # 제약 조건 위반 체크
        violations = 0
        if bcr > self.design_constraints['bcr_limit']:
            violations += 1
        if far < self.design_constraints['far_min'] or far > self.design_constraints['far_max']:
            violations += 1
        if sunlight < self.design_constraints['sunlight_min']:
            violations += 1
        if not (self.design_constraints['svr_range'][0] <= svr <= self.design_constraints['svr_range'][1]):
            violations += 1
        
        # 법적 준수 여부
        legal_compliance = (bcr <= self.design_constraints['bcr_limit'] and 
                          self.design_constraints['far_min'] <= far <= self.design_constraints['far_max'])
        
        # 지속가능성 점수 (일조량 + SV비율 기반)
        sunlight_score = min(1.0, max(0.0, (sunlight - self.design_constraints['sunlight_min']) / 
                                      (self.design_constraints['sunlight_max'] - self.design_constraints['sunlight_min'])))
        svr_score = 1.0 - abs(svr - self.design_constraints['svr_optimal']) / 0.4  # 0.4~1.2 범위에서 0.8이 최적
        svr_score = max(0.0, min(1.0, svr_score))
        
        sustainability_score = (sunlight_score + svr_score) / 2.0
        
        # 전체 품질 점수
        quality_score = (4 - violations) / 4.0 * 0.5 + sustainability_score * 0.5
        
        return {
            'quality_score': quality_score,
            'constraint_violations': violations,
            'legal_compliance': legal_compliance,
            'sustainability_score': sustainability_score
        }
    
    def _analyze_dimensions(self):
        """상태 및 액션 차원 분석"""
        if self.zmq_data_filtered.empty:
            self.state_dim = self.state_dimensions
            self.action_dim = 0
            return
        
        # 상태 차원 (고정)
        self.state_dim = self.state_dimensions
        
        # 액션 차원 찾기
        action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
        self.action_dim = len(action_cols)
        
        print(f"상태 차원: {self.state_dim} ({', '.join(self.state_labels)})")
        print(f"액션 차원: {self.action_dim}")

    def analyze_design_diversity_for_comparison(self, n_clusters=5, samples_per_cluster=5):
        """쌍대비교를 위한 다양한 디자인 분석 및 선별"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return None, None
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 클러스터링을 위한 특징 선택 (확장된 상태 공간)
            feature_cols = [f'state_{i}' for i in range(self.state_dimensions)]
            
            if not all(col in self.zmq_data_filtered.columns for col in feature_cols):
                print("클러스터링을 위한 상태 데이터가 완전하지 않습니다.")
                return None, None
            
            # 데이터 전처리
            X = self.zmq_data_filtered[feature_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            
            if X.empty:
                print("전처리 후 데이터가 없습니다.")
                return None, None
            
            # 품질 점수가 있는 데이터만 선택 (법적 준수 우선)
            valid_designs = self.zmq_data_filtered.loc[X.index].copy()
            legal_designs = valid_designs[valid_designs['legal_compliance'] == True]
            
            print(f"전체 디자인: {len(valid_designs)}, 법적 준수 디자인: {len(legal_designs)}")
            
            # 법적 준수 디자인이 충분하지 않으면 품질 점수 상위 50% 사용
            if len(legal_designs) < n_clusters * samples_per_cluster:
                print("법적 준수 디자인이 충분하지 않습니다. 품질 점수 기준으로 선별합니다.")
                quality_threshold = valid_designs['quality_score'].quantile(0.5)
                good_designs = valid_designs[valid_designs['quality_score'] >= quality_threshold]
                cluster_source = good_designs
            else:
                cluster_source = legal_designs
            
            # 클러스터링 대상 데이터 준비
            cluster_features = cluster_source[feature_cols]
            
            # 데이터 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_features)
            
            # K-means 클러스터링
            n_clusters = min(n_clusters, len(cluster_source) // 2)  # 클러스터 수 조정
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # 클러스터 정보 추가
            cluster_source = cluster_source.copy()
            cluster_source['cluster'] = clusters
            
            # 각 클러스터에서 다양한 샘플 선택
            diverse_designs = []
            cluster_stats = []
            
            for i in range(n_clusters):
                cluster_data = cluster_source[cluster_source['cluster'] == i]
                if cluster_data.empty:
                    continue
                
                # 클러스터 내에서 보상 점수 기준 정렬
                cluster_sorted = cluster_data.sort_values('reward', ascending=False)
                
                # 상위, 중위, 하위에서 균등하게 선택
                n_samples = min(samples_per_cluster, len(cluster_sorted))
                if n_samples >= 3:
                    # 상위 1/3, 중위 1/3, 하위 1/3에서 선택
                    indices = [
                        0,  # 최고
                        len(cluster_sorted) // 3,  # 중상위
                        len(cluster_sorted) * 2 // 3,  # 중하위
                    ]
                    if n_samples > 3:
                        # 추가 샘플들을 균등하게 분배
                        additional = np.linspace(0, len(cluster_sorted)-1, n_samples, dtype=int)
                        indices = sorted(set(list(additional)))
                else:
                    indices = list(range(n_samples))
                
                selected_designs = cluster_sorted.iloc[indices[:n_samples]]
                
                for _, design in selected_designs.iterrows():
                    diverse_designs.append(design)
                
                # 클러스터 통계
                cluster_stats.append({
                    'cluster': i,
                    'size': len(cluster_data),
                    'avg_reward': cluster_data['reward'].mean(),
                    'avg_quality': cluster_data['quality_score'].mean(),
                    'legal_compliance_rate': cluster_data['legal_compliance'].mean(),
                    'selected_count': len(selected_designs)
                })
            
            diverse_designs_df = pd.DataFrame(diverse_designs)
            cluster_stats_df = pd.DataFrame(cluster_stats)
            
            print(f"\n=== 쌍대비교용 다양한 디자인 선별 결과 ===")
            print(f"총 선별된 디자인: {len(diverse_designs_df)}개")
            print("클러스터별 통계:")
            print(cluster_stats_df.to_string(index=False))
            
            return diverse_designs_df, cluster_stats_df
            
        except ImportError:
            print("클러스터링을 위해 scikit-learn 라이브러리가 필요합니다.")
            return None, None
        except Exception as e:
            print(f"다양성 분석 중 오류 발생: {e}")
            traceback.print_exc()
            return None, None

    def find_optimal_designs_for_comparison(self, top_n=10, quality_weight=0.3, reward_weight=0.7):
        """쌍대비교를 위한 최적 디자인 찾기 (복합 점수 기준)"""
        if self.zmq_data_filtered.empty:
            print("분석할 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 복합 점수 계산 (보상 + 품질)
        reward_normalized = (self.zmq_data_filtered['reward'] - self.zmq_data_filtered['reward'].min()) / \
                           (self.zmq_data_filtered['reward'].max() - self.zmq_data_filtered['reward'].min())
        
        quality_normalized = self.zmq_data_filtered['quality_score']
        
        composite_score = reward_weight * reward_normalized + quality_weight * quality_normalized
        self.zmq_data_filtered = self.zmq_data_filtered.copy()
        self.zmq_data_filtered['composite_score'] = composite_score
        
        # 복합 점수 기준 상위 N개 선택
        top_designs = self.zmq_data_filtered.sort_values('composite_score', ascending=False).head(top_n).copy()
        
        # 결과 출력용 칼럼 선택
        result_cols = ['design_id', 'msg_id', 'reward', 'quality_score', 'composite_score', 
                      'legal_compliance', 'sustainability_score', 'datetime']
        
        # 상태 값들 추가
        for i, label in enumerate(self.state_labels):
            result_cols.append(f'state_{i}')
            result_cols.append(label.lower())
        
        # 액션 값들 추가
        action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
        result_cols.extend(action_cols)
        
        # 존재하는 칼럼만 선택
        available_cols = [col for col in result_cols if col in top_designs.columns]
        result_df = top_designs[available_cols].reset_index(drop=True)
        
        print(f"\n=== 쌍대비교용 최적 디자인 (상위 {len(result_df)}개) ===")
        print(f"복합 점수 구성: 보상({reward_weight:.1f}) + 품질({quality_weight:.1f})")
        
        for idx, row in result_df.iterrows():
            print(f"\n디자인 {idx+1}: ID={row.get('design_id', 'N/A')}")
            print(f"  복합점수: {row.get('composite_score', 0):.4f} (보상: {row.get('reward', 0):.4f}, 품질: {row.get('quality_score', 0):.4f})")
            print(f"  법적준수: {'예' if row.get('legal_compliance', False) else '아니오'}")
            print(f"  BCR: {row.get('bcr', 0)*100:.1f}%, FAR: {row.get('far', 0)*100:.1f}%")
            print(f"  일조량: {row.get('winter_sunlight', 0)/1000:.1f}k kWh, SV비율: {row.get('sv_ratio', 0):.3f}")
        
        return result_df

    def generate_rlhf_comparison_dataset(self, output_path, top_n=15, diverse_n=20):
        """RLHF 쌍대비교를 위한 기준 데이터셋 생성"""
        if self.zmq_data_filtered.empty:
            print("기준 데이터 생성을 위한 데이터가 없습니다.")
            return None
        
        try:
            # 1. 최적 디자인 선별
            print("1. 최적 디자인 선별 중...")
            top_designs = self.find_optimal_designs_for_comparison(top_n=top_n)
            
            # 2. 다양한 디자인 선별  
            print("2. 다양한 디자인 선별 중...")
            diverse_designs, cluster_stats = self.analyze_design_diversity_for_comparison(
                n_clusters=5, samples_per_cluster=diverse_n//5
            )
            
            if diverse_designs is None:
                diverse_designs = pd.DataFrame()
            
            # 3. 기준 데이터셋 구성
            reference_dataset = {
                'format_version': "2.0",
                'system_type': "RLHF_pairwise_comparison", 
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_samples': len(self.zmq_data_filtered),
                    'state_dimensions': self.state_dimensions,
                    'state_labels': self.state_labels,
                    'design_constraints': self.design_constraints,
                    'selection_criteria': {
                        'top_designs_count': len(top_designs),
                        'diverse_designs_count': len(diverse_designs),
                        'composite_score_weights': {'reward': 0.7, 'quality': 0.3}
                    }
                },
                'top_designs': [],
                'diverse_designs': [],
                'cluster_statistics': cluster_stats.to_dict('records') if cluster_stats is not None else []
            }
            
            # 4. 최고 성능 디자인 데이터 추가
            for idx, row in top_designs.iterrows():
                design_data = {
                    'id': row.get('design_id', f"top_{idx}_{int(time.time())}"),
                    'msg_id': int(row.get('msg_id', 0)),
                    'timestamp': int(row.get('timestamp', time.time() * 1000)),
                    'reward': float(row.get('reward', 0.0)),
                    'quality_score': float(row.get('quality_score', 0.0)),
                    'composite_score': float(row.get('composite_score', 0.0)),
                    'legal_compliance': bool(row.get('legal_compliance', False)),
                    'sustainability_score': float(row.get('sustainability_score', 0.0)),
                    'constraint_violations': int(row.get('constraint_violations', 0)),
                    'state': [float(row.get(f'state_{i}', 0.0)) for i in range(self.state_dimensions)],
                    'state_labels': dict(zip(self.state_labels, 
                                           [float(row.get(f'state_{i}', 0.0)) for i in range(self.state_dimensions)])),
                    'action': [float(row.get(f'action_{i}', 0.0)) for i in range(self.action_dim) 
                              if f'action_{i}' in row],
                    'type': 'top_performance'
                }
                reference_dataset['top_designs'].append(design_data)
            
            # 5. 다양한 디자인 데이터 추가
            if not diverse_designs.empty:
                for idx, row in diverse_designs.iterrows():
                    design_data = {
                        'id': row.get('design_id', f"diverse_{idx}_{int(time.time())}"),
                        'msg_id': int(row.get('msg_id', 0)),
                        'timestamp': int(row.get('timestamp', time.time() * 1000)),
                        'reward': float(row.get('reward', 0.0)),
                        'quality_score': float(row.get('quality_score', 0.0)),
                        'composite_score': float(row.get('composite_score', 0.0)) if 'composite_score' in row else 0.0,
                        'legal_compliance': bool(row.get('legal_compliance', False)),
                        'sustainability_score': float(row.get('sustainability_score', 0.0)),
                        'constraint_violations': int(row.get('constraint_violations', 0)),
                        'cluster': int(row.get('cluster', -1)),
                        'state': [float(row.get(f'state_{i}', 0.0)) for i in range(self.state_dimensions)],
                        'state_labels': dict(zip(self.state_labels, 
                                               [float(row.get(f'state_{i}', 0.0)) for i in range(self.state_dimensions)])),
                        'action': [float(row.get(f'action_{i}', 0.0)) for i in range(self.action_dim) 
                                  if f'action_{i}' in row],
                        'type': 'diverse_exploration'
                    }
                    reference_dataset['diverse_designs'].append(design_data)
            
            # 6. 저장 폴더 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 7. JSON 파일로 저장
            with open(output_path, 'w') as f:
                json.dump(reference_dataset, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ RLHF 쌍대비교 기준 데이터가 {output_path}에 저장되었습니다.")
            print(f"📊 최고 성능 디자인: {len(reference_dataset['top_designs'])}개")
            print(f"🎯 다양한 탐색 디자인: {len(reference_dataset['diverse_designs'])}개")
            print(f"📈 총 비교 가능한 디자인: {len(reference_dataset['top_designs']) + len(reference_dataset['diverse_designs'])}개")
            
            return reference_dataset
            
        except Exception as e:
            print(f"RLHF 기준 데이터 생성 중 오류 발생: {e}")
            traceback.print_exc()
            return None

    # 나머지 기존 메서드들은 유지...
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
            
            self.model_info = model_info
            print(f"모델 정보 로드 완료")
            
        except Exception as e:
            print(f"모델 정보 로드 중 오류 발생: {e}")
            self.model_info = {}

    def save_processed_data(self, output_path):
        """처리된 데이터 저장"""
        if self.zmq_data_filtered.empty:
            print("저장할 데이터가 없습니다.")
            return
        
        try:
            # 필요한 열만 선택
            cols_to_save = ['design_id', 'msg_id', 'timestamp', 'datetime', 'reward', 
                           'quality_score', 'legal_compliance', 'sustainability_score']
            
            # 상태 값들 추가
            for i in range(self.state_dimensions):
                cols_to_save.append(f'state_{i}')
            for label in self.state_labels:
                cols_to_save.append(label.lower())
            
            # 액션 값들 추가
            action_cols = [col for col in self.zmq_data_filtered.columns if col.startswith('action_')]
            cols_to_save.extend(action_cols)
            
            # 존재하는 열만 선택
            existing_cols = [col for col in cols_to_save if col in self.zmq_data_filtered.columns]
            
            # 저장 폴더 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 데이터 저장
            self.zmq_data_filtered[existing_cols].to_csv(output_path, index=False)
            print(f"처리된 데이터가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")


def find_latest_file(directory, pattern, subdirectory=None):
    """지정된 폴더에서 패턴과 일치하는 가장 최신 파일을 찾습니다."""
    import glob
    
    search_dir = os.path.join(directory, subdirectory) if subdirectory else directory
    files = glob.glob(os.path.join(search_dir, pattern))
    
    if not files:
        return None
    
    # 최신 파일 찾기 (수정 시간 기준)
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def create_session_directory(base_dir, session_name=None):
    """세션 디렉토리를 생성합니다."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if session_name:
        dir_name = f"{session_name}_{timestamp}"
    else:
        dir_name = f"rlhf_session_{timestamp}"
    
    session_dir = os.path.join(base_dir, dir_name)
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 쌍대비교용 데이터 분석 및 준비 도구')
    parser.add_argument('--state-reward-log', type=str, default=None,
                        help='ZMQ를 통해 수집된 상태/보상 로그 파일 경로')
    parser.add_argument('--model-dir', type=str, default='.',
                        help='PPO 모델 디렉토리 경로')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='결과 파일 저장 디렉토리')
    parser.add_argument('--session-name', type=str, default=None,
                        help='세션 이름')
    parser.add_argument('--top-designs', type=int, default=15,
                        help='선별할 최고 성능 디자인 수')
    parser.add_argument('--diverse-designs', type=int, default=20, 
                        help='선별할 다양한 디자인 수')
    
    args = parser.parse_args()
    
    # 프로젝트 루트 디렉토리 경로 계산
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 출력 디렉토리 설정
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 세션 디렉토리 생성
    session_dir = create_session_directory(args.output_dir, args.session_name)
    print(f"\n🔹 RLHF 분석 세션 디렉토리: {session_dir}")
    
    # 상태/보상 로그 파일 경로 결정
    state_reward_log_path = args.state_reward_log
    if not state_reward_log_path:
        # 프로젝트 루트 기준으로 최신 파일 찾기
        zmq_logs_dir = os.path.join(project_root, "data", "zmq_logs")
        state_reward_log_path = find_latest_file(zmq_logs_dir, "state_reward_log_*.json")
        
        if not state_reward_log_path:
            # 현재 디렉토리에서 찾기
            state_reward_log_path = find_latest_file(".", "state_reward_log_*.json")
            
        if not state_reward_log_path:
            print("❌ 상태/보상 로그 파일을 찾을 수 없습니다. --state-reward-log로 직접 지정해주세요.")
            return
    
    print(f"📂 사용할 상태/보상 로그 파일: {state_reward_log_path}")
    
    # 분석기 생성
    analyzer = RLHFDataAnalyzer(
        state_reward_log_path=state_reward_log_path,
        model_dir=args.model_dir,
        session_dir=session_dir
    )
    
    # 데이터 확인
    if analyzer.zmq_data_filtered.empty:
        print("\n❌ 분석할 데이터가 없습니다.")
        return
    
    print(f"\n📊 로드된 데이터: {len(analyzer.zmq_data_filtered)}개 디자인")
    print(f"📏 상태 차원: {analyzer.state_dimensions} ({', '.join(analyzer.state_labels)})")
    print(f"🎯 액션 차원: {analyzer.action_dim}")
    
    # 파일 경로 설정
    processed_data_file = os.path.join(session_dir, "processed_rlhf_data.csv")
    reference_data_file = os.path.join(session_dir, "rlhf_reference_data.json")
    
    # 1. 처리된 데이터 저장
    print("\n1️⃣ 처리된 데이터 저장 중...")
    analyzer.save_processed_data(processed_data_file)
    
    # 2. RLHF 쌍대비교용 기준 데이터 생성
    print("\n2️⃣ RLHF 쌍대비교용 기준 데이터 생성 중...")
    reference_data = analyzer.generate_rlhf_comparison_dataset(
        reference_data_file, 
        top_n=args.top_designs, 
        diverse_n=args.diverse_designs
    )
    
    if reference_data:
        print(f"\n✅ RLHF 쌍대비교 데이터 준비 완료!")
        print(f"📁 세션 디렉토리: {session_dir}")
        print(f"📄 기준 데이터: {reference_data_file}")
        print(f"📈 처리된 데이터: {processed_data_file}")
        print(f"\n🔄 다음 단계: design_regenerator.py 실행하여 3D 메시 생성")
    else:
        print(f"\n❌ 기준 데이터 생성에 실패했습니다.")

if __name__ == "__main__":
    main()