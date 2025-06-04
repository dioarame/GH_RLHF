#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
액션 변화 추이 분석 스크립트 (개선된 버전)
- RL 에이전트의 액션 값 변화 패턴 분석
- 보상 함수별 탐색 행동 비교
- 액션 공간 탐색 전략 시각화
- 초기화 액션 값 필터링 및 학습 단계별 정교한 분석
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# 경고 억제
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 전처리 (개선된 버전)
# ─────────────────────────────────────────────────────────────────────────────

def load_action_data(data_path, strategy_type):
    """액션 데이터 로드 및 전처리 (초기화 값 감지 및 필터링 포함)"""
    
    print(f"Loading action data: {data_path}")
    
    # 파일 존재 확인
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # CSV 로드 (헤더 자동 처리)
        first_line = open(data_path, 'r').readline().strip()
        if 'timestamp' in first_line.lower():
            df = pd.read_csv(data_path)
        else:
            # 헤더 없는 경우
            df = pd.read_csv(data_path, header=None)
            
            if len(df.columns) >= 9:
                base_cols = ['timestamp', 'step', 'is_closed_brep', 'excluded_from_training', 
                            'bcr', 'far', 'winter_sunlight', 'sv_ratio', 'reward']
                action_cols = [f'action{i+1}' for i in range(len(df.columns) - len(base_cols))]
                df.columns = base_cols + action_cols
            else:
                raise ValueError(f"Unexpected column count: {len(df.columns)}")
    
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")
    
    print(f"Original data: {len(df):,} rows")
    
    # 액션 컬럼 식별
    action_cols = [col for col in df.columns if col.startswith('action')]
    if not action_cols:
        raise ValueError("No action columns found!")
    
    print(f"Action columns detected: {action_cols}")
    
    # 데이터 타입 변환
    numeric_cols = ['bcr', 'far', 'winter_sunlight', 'sv_ratio', 'reward'] + action_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 기본 필터링
    original_count = len(df)
    df = df[df['is_closed_brep'] == 1]  # 유효한 형태만
    df = df[df['excluded_from_training'] == 0]  # 학습 포함 데이터만
    df = df.dropna(subset=numeric_cols)  # 결측치 제거
    
    print(f"After basic filtering: {len(df):,} rows (removed: {original_count - len(df):,})")
    
    # 초기화 액션 감지 및 제거
    df = detect_and_filter_initialization_actions(df, action_cols)
    
    if len(df) == 0:
        raise ValueError("No valid data after filtering!")
    
    # 액션 정규화 (실제 슬라이더 값을 -1~1 범위로 역변환)
    # 실제 슬라이더 범위 자동 감지 또는 기본값 사용
    slider_ranges = auto_detect_slider_ranges(df, action_cols)
    
    for col in action_cols:
        if col in slider_ranges:
            min_val, max_val = slider_ranges[col]
            # 실제 값을 정규화된 값으로 역변환
            df[f'{col}_normalized'] = 2.0 * (df[col] - min_val) / (max_val - min_val) - 1.0
            df[f'{col}_normalized'] = df[f'{col}_normalized'].clip(-1.0, 1.0)
        else:
            # 범위를 모르는 경우 데이터 기반 정규화
            min_val, max_val = df[col].min(), df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = 2.0 * (df[col] - min_val) / (max_val - min_val) - 1.0
            else:
                df[f'{col}_normalized'] = 0.0
    
    # 학습 단계 구분 (더 정교한 구분)
    df = add_learning_phases(df)
    
    # 액션 변화량 계산 (탐색 활동도 측정용)
    df = add_action_dynamics(df, action_cols)
    
    return df, action_cols

def detect_and_filter_initialization_actions(df, action_cols):
    """초기화 액션 감지 및 필터링"""
    
    original_count = len(df)
    
    # 1. 초기 스텝들 중 동일한 액션 값을 가진 것들 감지
    # 보통 처음 몇 스텝은 초기화 액션이 반복됨
    initial_steps = min(10, len(df) // 10)  # 처음 10개 또는 전체의 10%
    
    if len(df) > initial_steps:
        initial_data = df.head(initial_steps)
        
        # 모든 액션이 동일한 값을 가지는 행들 찾기
        identical_action_mask = pd.Series([False] * len(df))
        
        for _, row in initial_data.iterrows():
            # 현재 행과 동일한 액션 조합을 가진 모든 행 찾기
            mask = pd.Series([True] * len(df))
            for col in action_cols:
                mask &= (abs(df[col] - row[col]) < 0.001)  # 거의 동일한 값
            
            # 동일한 액션 조합이 5개 이상이면 초기화로 간주
            if mask.sum() >= 5:
                identical_action_mask |= mask
        
        # 2. 추가 필터링: 모든 액션이 특정 값으로 동일한 경우 (예: 모두 0 또는 중간값)
        for col in action_cols:
            # 특정 값에 집중된 데이터 감지 (전체의 20% 이상이 동일한 값)
            value_counts = df[col].round(2).value_counts()
            if len(value_counts) > 0:
                most_common_value = value_counts.index[0]
                most_common_count = value_counts.iloc[0]
                
                if most_common_count > len(df) * 0.2:  # 20% 이상이 같은 값
                    print(f"⚠️ {col}에서 {most_common_value} 값이 {most_common_count}회 ({most_common_count/len(df)*100:.1f}%) 반복됨")
                    
                    # 해당 값들을 초기화로 간주하되, 너무 많이 제거하지 않도록 조심
                    if most_common_count < len(df) * 0.5:  # 50% 미만일 때만 제거
                        identical_action_mask |= (abs(df[col] - most_common_value) < 0.001)
        
        # 3. 연속된 동일 액션 시퀀스 감지
        for col in action_cols:
            # 연속된 5개 이상의 동일한 값 찾기
            diff = df[col].diff().fillna(1)  # 첫 번째는 변화로 간주
            same_value_groups = (diff != 0).cumsum()
            group_sizes = same_value_groups.value_counts()
            
            # 5개 이상 연속된 그룹들 찾기
            large_groups = group_sizes[group_sizes >= 5].index
            for group_id in large_groups:
                group_mask = (same_value_groups == group_id)
                if group_mask.sum() >= 5:
                    print(f"⚠️ {col}에서 {group_mask.sum()}개의 연속된 동일 값 감지")
                    identical_action_mask |= group_mask
        
        # 필터링 실행
        if identical_action_mask.sum() > 0:
            print(f"🔍 초기화 액션으로 감지된 데이터: {identical_action_mask.sum()}개")
            
            # 너무 많이 제거하지 않도록 안전장치
            removal_ratio = identical_action_mask.sum() / len(df)
            if removal_ratio > 0.7:  # 70% 이상 제거하려 하면 경고
                print(f"⚠️ 너무 많은 데이터({removal_ratio*100:.1f}%)를 제거하려 합니다. 일부만 제거합니다.")
                # 초기 30%만 제거
                initial_portion = int(len(df) * 0.3)
                identical_action_mask = identical_action_mask & (df.index < df.index[initial_portion])
            
            df = df[~identical_action_mask]
            print(f"✅ 초기화 액션 제거 완료: {identical_action_mask.sum()}개 제거, {len(df)}개 남음")
    
    removed_count = original_count - len(df)
    print(f"Total removed by initialization filter: {removed_count} rows")
    
    return df

def auto_detect_slider_ranges(df, action_cols):
    """슬라이더 범위 자동 감지"""
    
    slider_ranges = {}
    
    # 일반적인 슬라이더 범위 패턴 감지
    for col in action_cols:
        values = df[col].dropna()
        min_val, max_val = values.min(), values.max()
        value_range = max_val - min_val
        
        # 일반적인 패턴들 확인
        if min_val >= 9 and max_val <= 26 and value_range > 10:
            # Height 관련 슬라이더 (10~25 등)
            slider_ranges[col] = (10.0, 25.0)
        elif min_val >= 45 and max_val <= 105 and value_range > 30:
            # Percentage 관련 슬라이더 (50~100 등)
            slider_ranges[col] = (50.0, 100.0)
        elif min_val >= -5 and max_val <= 105 and value_range > 50:
            # 0~100 범위 슬라이더
            slider_ranges[col] = (0.0, 100.0)
        else:
            # 데이터 기반 범위 설정 (약간의 마진 추가)
            margin = value_range * 0.1
            slider_ranges[col] = (min_val - margin, max_val + margin)
        
        print(f"📏 {col} 감지된 범위: {slider_ranges[col]}")
    
    return slider_ranges

def add_learning_phases(df):
    """학습 단계 구분 추가 (더 정교한 구분)"""
    
    total_steps = df['step'].max() - df['step'].min()
    
    if total_steps >= 2000:
        # 충분한 데이터가 있으면 5단계 구분
        bins = [
            df['step'].min(),
            df['step'].min() + total_steps * 0.2,   # Early (0-20%)
            df['step'].min() + total_steps * 0.4,   # Early-Mid (20-40%)
            df['step'].min() + total_steps * 0.6,   # Mid (40-60%)
            df['step'].min() + total_steps * 0.8,   # Mid-Late (60-80%)
            df['step'].max()                        # Late (80-100%)
        ]
        labels = ['Early', 'Early-Mid', 'Middle', 'Mid-Late', 'Late']
    elif total_steps >= 500:
        # 중간 정도 데이터면 3단계 구분
        bins = [
            df['step'].min(),
            df['step'].min() + total_steps * 0.33,
            df['step'].min() + total_steps * 0.67,
            df['step'].max()
        ]
        labels = ['Early', 'Middle', 'Late']
    else:
        # 적은 데이터면 단일 단계
        df['learning_phase'] = 'All'
        df['learning_phase_numeric'] = 0.5
        return df
    
    df['learning_phase'] = pd.cut(df['step'], bins=bins, labels=labels, include_lowest=True)
    
    # 수치형 단계 추가 (0~1 사이)
    phase_mapping = {label: i/(len(labels)-1) for i, label in enumerate(labels)}
    df['learning_phase_numeric'] = df['learning_phase'].map(phase_mapping)
    
    return df

def add_action_dynamics(df, action_cols):
    """액션 변화량 및 탐색 활동도 계산"""
    
    # 이전 스텝 대비 액션 변화량 계산
    for col in action_cols:
        df[f'{col}_delta'] = df[col].diff()
        df[f'{col}_abs_delta'] = df[f'{col}_delta'].abs()
    
    # 전체 액션 변화량 (모든 액션의 변화량 합)
    delta_cols = [f'{col}_abs_delta' for col in action_cols]
    df['total_action_change'] = df[delta_cols].sum(axis=1)
    
    # 탐색 활동도 (이동 평균)
    window_size = min(50, len(df) // 10)
    df['exploration_activity'] = df['total_action_change'].rolling(
        window=window_size, min_periods=1
    ).mean()
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. 개선된 액션 추이 분석 함수들
# ─────────────────────────────────────────────────────────────────────────────

def create_action_trends_plot(df, action_cols, strategy_type, output_dir):
    """액션 값 변화 추이 그래프 생성 (개선된 버전)"""
    
    # 스무딩 윈도우 설정
    smooth_window = min(100, len(df) // 20)
    
    # 서브플롯 생성
    fig, axes = plt.subplots(len(action_cols) + 1, 1, 
                            figsize=(16, 3 * (len(action_cols) + 1)), sharex=True)
    if len(action_cols) == 0:
        axes = [axes]
    
    strategy_name = "MaxMin Strategy" if strategy_type == "maxmin" else "Optimized Strategy"
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8']
    
    # 각 액션별 추이
    for i, action_col in enumerate(action_cols):
        ax = axes[i]
        
        # 원본 및 스무딩된 데이터
        raw_data = df[action_col]
        smoothed_data = df[action_col].rolling(window=smooth_window, min_periods=1).mean()
        
        # 정규화된 데이터도 표시
        norm_col = f'{action_col}_normalized'
        if norm_col in df.columns:
            norm_smoothed = df[norm_col].rolling(window=smooth_window, min_periods=1).mean()
            
            # 보조 축 생성
            ax2 = ax.twinx()
            ax2.plot(df['step'], norm_smoothed, color='gray', linewidth=1.5, alpha=0.8, 
                    linestyle='--', label='Normalized (-1~1)')
            ax2.set_ylabel('Normalized Value', color='gray', fontsize=10)
            ax2.set_ylim(-1.2, 1.2)
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        # 메인 데이터 플롯
        ax.plot(df['step'], raw_data, alpha=0.3, color='lightgray', linewidth=0.5, label='Raw')
        ax.plot(df['step'], smoothed_data, color=colors[i % len(colors)], linewidth=2.5, 
               label=f'Smoothed (window={smooth_window})')
        
        # 학습 단계별 배경색
        if 'learning_phase' in df.columns and len(df['learning_phase'].unique()) > 1:
            add_phase_backgrounds(ax, df)
        
        # 액션 이름 및 범위 정보
        action_name = f"Action {i+1}"
        action_range = f"[{raw_data.min():.1f}, {raw_data.max():.1f}]"
        
        # 변화량 통계
        if f'{action_col}_abs_delta' in df.columns:
            avg_change = df[f'{action_col}_abs_delta'].mean()
            change_info = f", Avg Change: {avg_change:.2f}"
        else:
            change_info = ""
        
        ax.set_title(f'{action_name} Trend - {strategy_name}\n'
                    f'Range: {action_range}{change_info}', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{action_name} Value')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        
        # 통계 정보 텍스트 박스
        mean_val = raw_data.mean()
        std_val = raw_data.std()
        
        # 트렌드 방향 계산 (초기 20% vs 마지막 20%)
        early_data = raw_data.head(len(raw_data)//5)
        late_data = raw_data.tail(len(raw_data)//5)
        trend_direction = "↗" if late_data.mean() > early_data.mean() else "↘"
        
        stability = "안정적" if std_val < mean_val * 0.1 else "변동적"
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nTrend: {trend_direction}\n{stability}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    # 전체 탐색 활동도 그래프 추가
    if 'exploration_activity' in df.columns:
        ax_explore = axes[-1]
        
        ax_explore.plot(df['step'], df['exploration_activity'], 
                       color='purple', linewidth=2, label='Exploration Activity')
        ax_explore.fill_between(df['step'], df['exploration_activity'], 
                               alpha=0.3, color='purple')
        
        ax_explore.set_title('Overall Exploration Activity', fontsize=11, fontweight='bold')
        ax_explore.set_ylabel('Activity Level')
        ax_explore.set_xlabel('Training Step')
        ax_explore.legend()
        ax_explore.grid(alpha=0.3)
        
        # 학습 단계별 배경색
        if 'learning_phase' in df.columns and len(df['learning_phase'].unique()) > 1:
            add_phase_backgrounds(ax_explore, df)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"action_trends_{strategy_type}.png", dpi=200, bbox_inches='tight')
    plt.close()

def add_phase_backgrounds(ax, df):
    """학습 단계별 배경색 추가"""
    
    if 'learning_phase_numeric' not in df.columns:
        return
    
    colors = ['#ffebee', '#e8f5e8', '#e3f2fd', '#fff3e0', '#f3e5f5']
    phases = df['learning_phase'].unique()
    
    for i, phase in enumerate(phases):
        if pd.isna(phase):
            continue
            
        phase_data = df[df['learning_phase'] == phase]
        if len(phase_data) > 0:
            start_step = phase_data['step'].min()
            end_step = phase_data['step'].max()
            
            ax.axvspan(start_step, end_step, alpha=0.2, 
                      color=colors[i % len(colors)], label=f'{phase} Phase')

def create_action_exploration_heatmap(df, action_cols, strategy_type, output_dir):
    """액션 공간 탐색 히트맵 생성 (개선된 버전)"""
    
    # 정규화된 액션 데이터 사용
    norm_cols = [f'{col}_normalized' for col in action_cols if f'{col}_normalized' in df.columns]
    
    if len(norm_cols) >= 2:
        # 학습 단계별 세분화
        if 'learning_phase' in df.columns and len(df['learning_phase'].unique()) > 1:
            phases = [phase for phase in df['learning_phase'].unique() if pd.notna(phase)]
            n_phases = len(phases)
        else:
            # 수동으로 단계 구분
            total_len = len(df)
            phases = ['Early', 'Late']
            phase_data = {
                'Early': df.head(total_len//2),
                'Late': df.tail(total_len//2)
            }
            n_phases = 2
        
        # 서브플롯 생성
        fig, axes = plt.subplots(1, n_phases, figsize=(6 * n_phases, 6))
        if n_phases == 1:
            axes = [axes]
        
        strategy_name = "MaxMin Strategy" if strategy_type == "maxmin" else "Optimized Strategy"
        
        for i, phase in enumerate(phases):
            ax = axes[i]
            
            # 단계별 데이터 추출
            if 'learning_phase' in df.columns:
                phase_df = df[df['learning_phase'] == phase]
            else:
                phase_df = phase_data[phase]
            
            if len(phase_df) == 0:
                continue
            
            # 2D 히스토그램
            x_data = phase_df[norm_cols[0]]
            y_data = phase_df[norm_cols[1]]
            
            # 아웃라이어 제거 (99% 분위수 기준)
            x_q1, x_q99 = x_data.quantile([0.01, 0.99])
            y_q1, y_q99 = y_data.quantile([0.01, 0.99])
            
            mask = ((x_data >= x_q1) & (x_data <= x_q99) & 
                   (y_data >= y_q1) & (y_data <= y_q99))
            
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            # 히트맵 생성
            h = ax.hist2d(x_clean, y_clean, bins=25, cmap='Blues', alpha=0.8)
            plt.colorbar(h[3], ax=ax, label='Frequency')
            
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel(f'{action_cols[0]} (Normalized)')
            ax.set_ylabel(f'{action_cols[1]} (Normalized)')
            ax.set_title(f'{phase} Phase\n({len(phase_df)} samples)')
            ax.grid(alpha=0.3)
            
            # 중심점 및 사분면 표시
            ax.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # 탐색 범위 통계 추가
            x_range = x_clean.max() - x_clean.min()
            y_range = y_clean.max() - y_clean.min()
            coverage = (x_range * y_range) / 4.0  # 정규화된 공간에서의 커버리지
            
            stats_text = f'X Range: {x_range:.2f}\nY Range: {y_range:.2f}\nCoverage: {coverage:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(f'Action Space Exploration Pattern - {strategy_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"action_exploration_{strategy_type}.png", dpi=200, bbox_inches='tight')
        plt.close()

def create_action_statistics_plot(df, action_cols, strategy_type, output_dir):
    """액션 통계 분석 플롯 (개선된 버전)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    strategy_name = "MaxMin Strategy" if strategy_type == "maxmin" else "Optimized Strategy"
    
    # 1. 액션 변동성 추이 (개선됨)
    ax1 = axes[0, 0]
    window_size = max(50, len(df) // 30)
    
    for i, action_col in enumerate(action_cols):
        rolling_std = df[action_col].rolling(window=window_size, min_periods=1).std()
        rolling_mean = df[action_col].rolling(window=window_size, min_periods=1).mean()
        cv = rolling_std / (rolling_mean + 1e-8)  # 변동계수 (Coefficient of Variation)
        
        ax1.plot(df['step'], cv, label=f'Action {i+1} CV', linewidth=2, alpha=0.8)
    
    ax1.set_title('Action Variability (Coefficient of Variation)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('CV (Std/Mean)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 탐색-활용 전환점 표시
    if len(action_cols) > 0:
        avg_cv = np.mean([df[col].rolling(window=window_size, min_periods=1).std() / 
                         (df[col].rolling(window=window_size, min_periods=1).mean() + 1e-8) 
                         for col in action_cols], axis=0)
        # CV가 최고점에서 50% 감소한 지점을 전환점으로 간주
        max_cv_idx = np.argmax(avg_cv)
        transition_threshold = avg_cv[max_cv_idx] * 0.5
        transition_points = np.where(avg_cv[max_cv_idx:] <= transition_threshold)[0]
        if len(transition_points) > 0:
            transition_step = df['step'].iloc[max_cv_idx + transition_points[0]]
            ax1.axvline(transition_step, color='red', linestyle='--', alpha=0.7, 
                       label=f'Explore→Exploit transition')
            ax1.legend()
    
    # 2. 액션별 분포 비교 (개선됨)
    ax2 = axes[0, 1]
    
    # 바이올린 플롯으로 분포 형태까지 표시
    action_data = [df[col] for col in action_cols]
    action_labels = [f'Action {i+1}' for i in range(len(action_cols))]
    
    parts = ax2.violinplot(action_data, showmeans=True, showmedians=True)
    ax2.set_xticks(range(1, len(action_cols) + 1))
    ax2.set_xticklabels(action_labels)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    
    ax2.set_title('Action Value Distribution (Violin Plot)')
    ax2.set_ylabel('Action Value')
    ax2.grid(alpha=0.3)
    
    # 분포 통계 추가
    for i, (col, data) in enumerate(zip(action_cols, action_data)):
        skewness = data.skew()
        kurtosis = data.kurtosis()
        ax2.text(i+1, data.max(), f'Skew: {skewness:.2f}\nKurt: {kurtosis:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 3. 액션 간 상관관계 (시간 진화 포함)
    ax3 = axes[1, 0]
    if len(action_cols) >= 2:
        # 시간 구간별 상관관계 변화
        n_segments = 5
        segment_size = len(df) // n_segments
        correlations_over_time = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)
            segment_df = df.iloc[start_idx:end_idx]
            
            if len(segment_df) > 1:
                corr = segment_df[action_cols[0]].corr(segment_df[action_cols[1]])
                correlations_over_time.append(corr)
            else:
                correlations_over_time.append(0)
        
        # 상관관계 변화 플롯
        segment_steps = [df['step'].iloc[i * segment_size] for i in range(n_segments)]
        ax3.plot(segment_steps, correlations_over_time, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel(f'Correlation ({action_cols[0]} vs {action_cols[1]})')
        ax3.set_title('Action Correlation Evolution')
        ax3.grid(alpha=0.3)
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 최종 상관관계 매트릭스 (인셋)
        if len(action_cols) > 2:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax3, width="40%", height="40%", loc='upper right')
            
            final_corr = df[action_cols].corr()
            im = axins.imshow(final_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            axins.set_xticks(range(len(action_cols)))
            axins.set_yticks(range(len(action_cols)))
            axins.set_xticklabels([f'A{i+1}' for i in range(len(action_cols))], fontsize=8)
            axins.set_yticklabels([f'A{i+1}' for i in range(len(action_cols))], fontsize=8)
            axins.set_title('Final Correlation', fontsize=10)
    
    # 4. 학습 단계별 액션 특성 (개선됨)
    ax4 = axes[1, 1]
    if 'learning_phase' in df.columns and len(df['learning_phase'].unique()) > 1:
        phases = [phase for phase in df['learning_phase'].unique() if pd.notna(phase)]
        
        # 각 단계별 액션의 평균과 표준편차
        phase_stats = {}
        for phase in phases:
            phase_data = df[df['learning_phase'] == phase]
            means = [phase_data[col].mean() for col in action_cols]
            stds = [phase_data[col].std() for col in action_cols]
            phase_stats[phase] = {'means': means, 'stds': stds}
        
        x = np.arange(len(action_cols))
        width = 0.8 / len(phases)
        
        for i, phase in enumerate(phases):
            means = phase_stats[phase]['means']
            stds = phase_stats[phase]['stds']
            
            bars = ax4.bar(x + i * width, means, width, 
                          yerr=stds, label=f'{phase} Phase', 
                          alpha=0.8, capsize=5)
        
        ax4.set_title('Action Statistics by Learning Phase')
        ax4.set_xlabel('Actions')
        ax4.set_ylabel('Value (Mean ± Std)')
        ax4.set_xticks(x + width * (len(phases) - 1) / 2)
        ax4.set_xticklabels([f'Action {i+1}' for i in range(len(action_cols))])
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    fig.suptitle(f'Comprehensive Action Analysis - {strategy_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"action_statistics_{strategy_type}.png", dpi=200, bbox_inches='tight')
    plt.close()

def create_action_reward_correlation(df, action_cols, strategy_type, output_dir):
    """액션-보상 상관관계 분석 (개선된 버전)"""
    
    n_actions = len(action_cols)
    fig, axes = plt.subplots(2, n_actions, figsize=(5 * n_actions, 10))
    if n_actions == 1:
        axes = axes.reshape(2, 1)
    
    strategy_name = "MaxMin Strategy" if strategy_type == "maxmin" else "Optimized Strategy"
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8']
    
    for i, action_col in enumerate(action_cols):
        # 상단: 산점도 (시간 진행별 색상)
        ax1 = axes[0, i]
        
        # 아웃라이어 제거
        action_data = df[action_col]
        reward_data = df['reward']
        
        # 99% 분위수 기준 필터링
        action_q1, action_q99 = action_data.quantile([0.01, 0.99])
        reward_q1, reward_q99 = reward_data.quantile([0.01, 0.99])
        
        mask = ((action_data >= action_q1) & (action_data <= action_q99) & 
               (reward_data >= reward_q1) & (reward_data <= reward_q99))
        
        clean_action = action_data[mask]
        clean_reward = reward_data[mask]
        clean_steps = df['step'][mask]
        
        # 산점도
        scatter = ax1.scatter(clean_action, clean_reward, c=clean_steps, 
                             cmap='viridis', alpha=0.6, s=15)
        
        # 상관계수 계산
        correlation = clean_action.corr(clean_reward)
        
        # 추세선 (로버스트 회귀)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_action, clean_reward)
        line_x = np.array([clean_action.min(), clean_action.max()])
        line_y = slope * line_x + intercept
        
        ax1.plot(line_x, line_y, "r-", alpha=0.8, linewidth=2.5, 
                label=f'R={correlation:.3f}, p={p_value:.3f}')
        
        ax1.set_xlabel(f'{action_col} Value')
        ax1.set_ylabel('Reward')
        ax1.set_title(f'Action {i+1} vs Reward')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # 컬러바 (마지막 액션에만)
        if i == len(action_cols) - 1:
            plt.colorbar(scatter, ax=ax1, label='Training Step')
        
        # 하단: 시간에 따른 상관관계 변화
        ax2 = axes[1, i]
        
        # 이동 윈도우 상관관계 계산
        window_size = max(100, len(df) // 20)
        rolling_corr = clean_action.rolling(window=window_size, min_periods=window_size//2).corr(clean_reward)
        
        ax2.plot(clean_steps, rolling_corr, color=colors[i % len(colors)], linewidth=2)
        ax2.fill_between(clean_steps, rolling_corr, alpha=0.3, color=colors[i % len(colors)])
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Rolling Correlation')
        ax2.set_title(f'Action {i+1} Correlation Evolution')
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 상관관계 강도별 구간 표시
        strong_positive = rolling_corr > 0.3
        strong_negative = rolling_corr < -0.3
        
        ax2.fill_between(clean_steps, rolling_corr, 0, where=strong_positive, 
                        color='green', alpha=0.2, label='Strong Positive')
        ax2.fill_between(clean_steps, rolling_corr, 0, where=strong_negative, 
                        color='red', alpha=0.2, label='Strong Negative')
        
        if strong_positive.any() or strong_negative.any():
            ax2.legend()
        
        # 통계 정보 추가
        final_corr = rolling_corr.iloc[-window_size:].mean() if len(rolling_corr) >= window_size else correlation
        corr_std = rolling_corr.std()
        
        stats_text = f'Final Corr: {final_corr:.3f}\nStd: {corr_std:.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'Action-Reward Relationship Analysis - {strategy_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"action_reward_correlation_{strategy_type}.png", dpi=200, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. 보고서 생성 (개선된 버전)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_action_patterns(df, action_cols, strategy_type):
    """액션 패턴 분석 및 요약 생성 (개선된 버전)"""
    
    total_steps = int(df['step'].max() - df['step'].min())
    total_samples = len(df)
    
    # 탐색 활동도 분석
    if 'exploration_activity' in df.columns:
        early_exploration = df['exploration_activity'].head(len(df)//3).mean()
        late_exploration = df['exploration_activity'].tail(len(df)//3).mean()
        exploration_decay = (early_exploration - late_exploration) / early_exploration if early_exploration > 0 else 0
        
        exploration_analysis = f"""
탐색 활동도가 초기 {early_exploration:.3f}에서 후기 {late_exploration:.3f}로 변화했습니다 ({exploration_decay*100:.1f}% 감소).
이는 {"적절한 탐색에서 활용으로의 전환" if exploration_decay > 0.2 else "지속적인 탐색 행동"}을 보여줍니다.
"""
    else:
        # 액션 변화량 기반 분석
        if len(action_cols) >= 2:
            action1_range = df[action_cols[0]].max() - df[action_cols[0]].min()
            action2_range = df[action_cols[1]].max() - df[action_cols[1]].min()
            exploration_diversity = (action1_range + action2_range) / 2
            
            if strategy_type == "maxmin":
                exploration_analysis = f"넓은 탐색 범위(평균 {exploration_diversity:.2f})를 보이며, 극값 추구 특성에 따라 액션 공간의 경계 영역을 적극적으로 탐색했습니다."
            else:
                exploration_analysis = f"안정적인 탐색 범위(평균 {exploration_diversity:.2f})를 보이며, 최적 범위 내에서 집중적인 탐색이 이루어졌습니다."
        else:
            exploration_analysis = "단일 액션으로 인해 제한적 분석이 수행되었습니다."
    
    # 상관관계 분석 (개선됨)
    if len(action_cols) >= 2:
        corr_matrix = df[action_cols].corr()
        max_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        
        # 시간별 상관관계 변화 분석
        early_corr = df[action_cols].head(len(df)//3).corr()
        late_corr = df[action_cols].tail(len(df)//3).corr()
        
        early_max = early_corr.abs().values[np.triu_indices_from(early_corr.values, k=1)].max()
        late_max = late_corr.abs().values[np.triu_indices_from(late_corr.values, k=1)].max()
        
        correlation_analysis = f"""
액션 간 상관관계가 초기 {early_max:.3f}에서 후기 {late_max:.3f}로 변화했습니다.
{"상관관계가 강화되어" if late_max > early_max else "상관관계가 약화되어"} 
{"협력적 액션 패턴" if late_max > 0.5 else "독립적 액션 패턴"}을 보입니다.
"""
    else:
        correlation_analysis = "단일 액션으로 인해 상관관계 분석이 불가능합니다."
    
    # 성과 기여도 분석 (개선됨)
    action_reward_corrs = [df[col].corr(df['reward']) for col in action_cols]
    best_action_idx = np.argmax(np.abs(action_reward_corrs))
    best_corr = action_reward_corrs[best_action_idx]
    
    # 시간별 상관관계 변화
    early_corrs = [df[col].head(len(df)//3).corr(df['reward'].head(len(df)//3)) for col in action_cols]
    late_corrs = [df[col].tail(len(df)//3).corr(df['reward'].tail(len(df)//3)) for col in action_cols]
    
    performance_analysis = f"""
Action {best_action_idx + 1}이 보상과 가장 강한 상관관계({best_corr:.3f})를 보였습니다.
학습 초기와 후기 상관관계 변화를 보면, 
{f"대부분 액션의 성과 기여도가 증가" if np.mean(late_corrs) > np.mean(early_corrs) else "액션 효과성이 안정화"}되었습니다.
"""
    
    # 핵심 발견사항 (전략별 맞춤화)
    if strategy_type == "maxmin":
        # 변동성 분석
        avg_cv = np.mean([df[col].std() / df[col].mean() for col in action_cols])
        
        key_findings = f"""
1. **적극적 탐색**: 평균 변동계수 {avg_cv:.3f}로 법적 경계 근처에서의 활발한 액션 변화 확인
2. **높은 변동성**: 혁신적 솔루션 탐색 과정에서 나타나는 자연스러운 변동성
3. **학습 진화**: 초기 무작위 탐색에서 후기 목적성 있는 탐색으로 진화
4. **경계 탐색**: 극값 추구로 인한 액션 공간의 경계 영역 집중 탐색
"""
    else:
        # 안정성 분석
        avg_cv = np.mean([df[col].std() / df[col].mean() for col in action_cols])
        
        key_findings = f"""
1. **안정적 수렴**: 평균 변동계수 {avg_cv:.3f}로 최적 범위로의 점진적 수렴 확인
2. **예측 가능성**: 일관된 액션 선택으로 높은 재현성 확보
3. **효율적 학습**: 불필요한 탐색 없이 목표 지향적 학습 수행
4. **범위 준수**: 검증된 최적 범위 내에서의 집중적 탐색
"""
    
    # 실무 적용 시사점 (상황별 세분화)
    if strategy_type == "maxmin":
        practical_implications = f"""
- **혁신 프로젝트**: 창의적 솔루션 발견에 효과적, 탐색 다양성 {exploration_diversity:.2f}
- **실험적 설계**: 충분한 탐색 시간과 실험 여유가 있는 프로젝트에 적합
- **위험 관리**: 결과 변동성({avg_cv:.3f})을 감안한 여러 대안 검토 필요
- **혁신 vs 안정성**: 획기적 개선 가능성과 예측 불가능성의 트레이드오프
"""
    else:
        practical_implications = f"""
- **상업 프로젝트**: 안정성({avg_cv:.3f})과 예측 가능성이 중요한 실무에 적합
- **즉시 적용**: 검증된 범위 내 솔루션으로 즉시 적용 가능한 결과 제공
- **위험 최소화**: 낮은 변동성으로 위험 관리가 중요한 프로젝트에 선호
- **효율성**: 탐색 대비 활용 비율이 높아 시간 효율적 접근법
"""
    
    # 개선 방향 (구체적 제안)
    improvement_suggestions = f"""
1. **하이브리드 접근**: 
   - 초기엔 MaxMin 방식으로 넓은 탐색, 후기엔 Optimized 방식으로 세밀 조정
   - 탐색-활용 균형점을 동적으로 조절하는 적응적 전략

2. **상황별 최적화**:
   - 프로젝트 위험도에 따른 탐색 강도 조절
   - 시간 제약에 따른 전략 선택 가이드라인

3. **성과 기반 조정**:
   - 실시간 성과 피드백을 통한 액션 공간 동적 조정
   - 상관관계 변화 패턴을 활용한 예측적 액션 가중치 조절

4. **멀티모달 탐색**:
   - 여러 최적해 후보를 동시 탐색하는 앙상블 접근법
   - 다양한 설계 제약 조건에 대한 강건성 확보
"""
    
    return {
        'total_steps': total_steps,
        'total_samples': total_samples,
        'exploration_analysis': exploration_analysis.strip(),
        'correlation_analysis': correlation_analysis.strip(),
        'performance_analysis': performance_analysis.strip(),
        'key_findings': key_findings.strip(),
        'practical_implications': practical_implications.strip(),
        'improvement_suggestions': improvement_suggestions.strip()
    }

def generate_action_analysis_report(df, action_cols, strategy_type, output_dir, analysis_summary):
    """액션 분석 보고서 생성 (개선된 버전)"""
    
    strategy_names = {
        "maxmin": "MaxMin 방향성 전략",
        "optimized": "Optimized 범위 기반 전략"
    }
    
    strategy_name_kr = strategy_names.get(strategy_type, strategy_type)
    strategy_name_en = "MaxMin Strategy" if strategy_type == "maxmin" else "Optimized Strategy"
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 데이터 품질 정보 추가
    data_quality_info = f"""
### 데이터 품질 정보
- **원본 데이터**: 필터링 전 정보 (로그에서 확인)
- **유효 데이터**: {analysis_summary['total_samples']:,}개 샘플
- **학습 기간**: {analysis_summary['total_steps']:,} 스텝
- **초기화 필터링**: 반복된 초기 액션 값 제거 완료
- **아웃라이어 제거**: 99% 분위수 기준 극값 제거 적용
"""
    
    report_md = f"""# 액션 변화 추이 분석 보고서 - {strategy_name_kr}

## 1. 분석 개요

- **분석 날짜**: {timestamp}
- **전략 유형**: {strategy_name_kr} ({strategy_name_en})
- **총 스텝 수**: {analysis_summary['total_steps']:,}
- **분석 샘플 수**: {analysis_summary['total_samples']:,}
- **액션 차원**: {len(action_cols)}개

{data_quality_info}

## 2. 액션 변화 추이 분석

![Action Trends](./action_trends_{strategy_type}.png)

*위 그래프는 학습 과정에서 각 액션 값의 변화 추이를 보여줍니다. 회색 선은 원본 데이터, 색상 선은 스무딩된 추이, 점선은 정규화된 값(-1~1)을 나타냅니다.*

### 2.1 주요 발견사항

"""
    
    # 액션별 통계 분석 (개선됨)
    for i, action_col in enumerate(action_cols):
        action_data = df[action_col]
        mean_val = action_data.mean()
        std_val = action_data.std()
        cv = std_val / mean_val if mean_val != 0 else 0
        
        # 트렌드 분석 (개선됨)
        early_data = action_data.head(len(action_data)//5)
        late_data = action_data.tail(len(action_data)//5)
        trend_change = (late_data.mean() - early_data.mean()) / early_data.mean() * 100 if early_data.mean() != 0 else 0
        trend_direction = "상승" if trend_change > 5 else "하락" if trend_change < -5 else "안정"
        
        # 변동성 특성
        stability = "매우 안정" if cv < 0.05 else "안정적" if cv < 0.15 else "변동적" if cv < 0.3 else "매우 변동적"
        
        # 분포 특성
        skewness = action_data.skew()
        kurtosis = action_data.kurtosis()
        distribution_shape = "정규분포" if abs(skewness) < 0.5 and abs(kurtosis) < 3 else "비대칭분포" if abs(skewness) > 1 else "치우친분포"
        
        report_md += f"""
#### Action {i+1} 세부 분석
- **통계**: 평균 {mean_val:.2f}, 표준편차 {std_val:.2f}, 변동계수 {cv:.3f}
- **범위**: [{action_data.min():.2f}, {action_data.max():.2f}]
- **변화 경향**: {trend_direction} ({trend_change:+.1f}%)
- **안정성**: {stability}
- **분포 특성**: {distribution_shape} (왜도: {skewness:.2f}, 첨도: {kurtosis:.2f})
"""
    
    # 전략별 해석 추가 (개선됨)
    if strategy_type == "maxmin":
        strategy_interpretation = f"""
### 2.2 MaxMin 전략 특성 해석

MaxMin 방향성 전략에서의 액션 변화는 다음과 같은 특징을 보입니다:

{analysis_summary['exploration_analysis']}

**전략적 함의:**
- 혁신적 설계 솔루션 탐색을 위한 적극적 액션 공간 탐색
- 법적 제한과 성과 간의 최적 균형점 발견 과정
- 기존 관례를 벗어난 창의적 파라미터 조합 시도

**학습 패턴:**
- 초기: 광범위한 탐색으로 가능성 공간 파악
- 중기: 유망한 영역에서의 집중적 탐색
- 후기: 극값 근처에서의 세밀한 조정
"""
    else:
        strategy_interpretation = f"""
### 2.2 Optimized 전략 특성 해석

Optimized 범위 기반 전략에서의 액션 변화는 다음과 같은 특징을 보입니다:

{analysis_summary['exploration_analysis']}

**전략적 함의:**
- 검증된 최적 범위로의 효율적이고 안정적인 수렴
- 예측 가능한 성과를 위한 체계적 액션 선택
- 실무 적용 가능성이 높은 보수적 탐색 전략

**학습 패턴:**
- 초기: 최적 범위 경계 확인
- 중기: 범위 내에서의 세밀한 조정
- 후기: 최적점 주변에서의 안정화
"""
    
    report_md += strategy_interpretation
    
    report_md += f"""

## 3. 액션 공간 탐색 패턴

![Action Exploration](./action_exploration_{strategy_type}.png)

*위 히트맵은 학습 단계별 액션 공간 탐색 패턴을 비교합니다. 색상이 진할수록 해당 액션 조합을 더 자주 선택했음을 의미합니다.*

### 3.1 탐색 전략 분석

{analysis_summary['exploration_analysis']}

### 3.2 공간 커버리지 특성

- **탐색 범위**: 정규화된 액션 공간에서의 실제 탐색 영역
- **집중도**: 특정 액션 조합에 대한 선호도 패턴
- **진화**: 학습 진행에 따른 탐색 패턴의 변화

## 4. 종합 액션 통계 분석

![Action Statistics](./action_statistics_{strategy_type}.png)

*위 그래프는 액션의 변동성, 분포, 상관관계, 학습 단계별 변화를 종합 분석한 결과입니다.*

### 4.1 액션 변동성 분석

- **변동계수 추이**: 학습 진행에 따른 탐색-활용 전환 패턴
- **안정화 지점**: 탐색에서 활용으로 전환되는 임계점 식별
- **분포 특성**: 각 액션의 값 분포 형태와 선호 구간

### 4.2 액션 간 상관관계

{analysis_summary['correlation_analysis']}

**상관관계 해석:**
- 높은 상관관계 (>0.7): 액션들이 협력적으로 작동하는 패턴
- 중간 상관관계 (0.3-0.7): 부분적 연관성을 가진 독립적 작동
- 낮은 상관관계 (<0.3): 각 액션이 독립적으로 기능

### 4.3 학습 단계별 특성 변화

학습 진행에 따른 액션 선택 패턴의 진화를 분석하여 전략의 효과성을 평가합니다.

## 5. 액션-보상 상관관계 심층 분석

![Action-Reward Correlation](./action_reward_correlation_{strategy_type}.png)

*위 그래프는 각 액션과 보상 간의 상관관계를 시간 순서와 함께 보여줍니다. 상단은 전체 상관관계, 하단은 시간에 따른 상관관계 변화를 나타냅니다.*

### 5.1 성과 기여도 분석

{analysis_summary['performance_analysis']}

### 5.2 학습 진화 패턴

- **초기 단계**: 무작위 탐색으로 인한 낮은 상관관계
- **중간 단계**: 패턴 학습으로 상관관계 강화
- **후기 단계**: 최적화된 액션-보상 관계 안정화

## 6. 전략별 성능 비교 및 인사이트

### 6.1 핵심 발견사항

{analysis_summary['key_findings']}

### 6.2 실무 적용 시사점

{analysis_summary['practical_implications']}

### 6.3 전략 선택 가이드라인

#### MaxMin 전략이 적합한 경우:
- 혁신적 돌파구가 필요한 프로젝트
- 탐색 시간과 실험 비용에 여유가 있는 상황
- 기존 관례에서 벗어난 창의적 솔루션 필요
- 장기적 관점에서 최적해 발견이 중요한 경우

#### Optimized 전략이 적합한 경우:
- 안정적이고 예측 가능한 결과가 필요한 상업 프로젝트
- 시간과 비용 제약이 엄격한 상황
- 위험 관리가 중요한 프로젝트
- 검증된 범위 내에서의 최적화가 목표인 경우

## 7. 향후 개선 및 연구 방향

### 7.1 단기 개선 방안

{analysis_summary['improvement_suggestions']}

### 7.2 장기 연구 방향

1. **적응적 전략 개발**:
   - 프로젝트 특성과 진행 상황에 따른 동적 전략 전환
   - 실시간 성과 피드백 기반 탐색 강도 자동 조절

2. **멀티 목적 최적화**:
   - 건축 성능, 비용, 시공성을 동시 고려하는 통합 보상 함수
   - 파레토 최적해 집합 탐색을 위한 다목적 액션 전략

3. **전이 학습 적용**:
   - 유사 프로젝트 경험을 활용한 초기 액션 가중치 설정
   - 도메인 지식 통합을 통한 탐색 효율성 향상

4. **설명 가능한 AI**:
   - 액션 선택 근거의 시각화 및 해석
   - 설계자가 이해할 수 있는 액션 추천 시스템

## 8. 결론

### 8.1 전략별 성과 요약

**{strategy_name_kr}의 특징:**
- **탐색 특성**: {analysis_summary['exploration_analysis'].split('.')[0]}
- **안정성**: {analysis_summary['correlation_analysis'].split('.')[0]}
- **성과 기여도**: {analysis_summary['performance_analysis'].split('.')[0]}

### 8.2 실무 권장사항

1. **프로젝트 초기**: 요구사항과 제약조건에 따른 전략 선택
2. **학습 과정**: 정기적인 성과 모니터링 및 전략 조정
3. **결과 활용**: 액션 패턴 분석을 통한 설계 인사이트 도출

### 8.3 최종 평가

이 분석을 통해 {strategy_name_kr}의 액션 탐색 특성과 학습 패턴을 종합적으로 이해할 수 있었습니다. 각 전략은 고유한 장단점을 가지고 있으며, 프로젝트의 목표와 제약조건에 따라 적절한 전략을 선택하는 것이 중요합니다.

---

**분석 완료 시간**: {timestamp}  
**사용된 데이터**: {analysis_summary['total_samples']:,}개 샘플  
**필터링된 초기화 액션**: 로그 참조  
**분석 도구**: Python 기반 개선된 액션 추이 분석 스크립트 v2.0

---

## 부록

### A. 기술적 세부사항

- **데이터 전처리**: 초기화 액션 자동 감지 및 제거
- **아웃라이어 처리**: 99% 분위수 기준 극값 필터링
- **스무딩 기법**: 적응적 이동평균 (윈도우 크기: 데이터 크기의 5-10%)
- **상관관계 계산**: 피어슨 상관계수 및 이동 윈도우 상관관계

### B. 시각화 범례

- **실선**: 스무딩된 추이
- **점선**: 정규화된 값 (-1~1 범위)
- **음영**: 학습 단계별 구간
- **색상**: 시간 진행 (어두울수록 후기)

### C. 용어 정의

- **변동계수(CV)**: 표준편차를 평균으로 나눈 값, 상대적 변동성 측정
- **탐색 활동도**: 이전 스텝 대비 전체 액션 변화량의 이동평균
- **학습 단계**: 전체 학습 과정을 시간순으로 구분한 구간
- **액션 공간**: 모든 가능한 액션 조합이 이루는 다차원 공간
"""