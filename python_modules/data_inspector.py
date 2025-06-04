#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV 데이터 검사 스크립트
"""

import pandas as pd
import numpy as np

# CSV 파일 로드
csv_path = r"C:\Users\valen\Desktop\Dev\6. RLHF\data\zmq_logs\architecture_metrics_20250523_162526.csv"

csv_columns = [
    'timestamp', 'step', 'is_closed_brep', 'excluded_from_training',
    'bcr', 'far', 'winter_sunlight', 'sv_ratio', 'reward',
    'action1', 'action2', 'action3', 'action4'
]

df = pd.read_csv(csv_path, header=None, names=csv_columns)

print(f"전체 데이터: {len(df)}개")
print(f"Closed brep: {df['is_closed_brep'].sum()}개")
print(f"학습 포함: {(~df['excluded_from_training']).sum()}개")

# 필터링된 데이터
filtered_df = df[(df['is_closed_brep'] == True) & (df['excluded_from_training'] == False)]
print(f"필터링된 데이터: {len(filtered_df)}개")

if len(filtered_df) > 0:
    print("\n=== 데이터 범위 확인 ===")
    
    # 상태값 범위
    print(f"BCR 범위: {filtered_df['bcr'].min():.4f} ~ {filtered_df['bcr'].max():.4f}")
    print(f"FAR 범위: {filtered_df['far'].min():.4f} ~ {filtered_df['far'].max():.4f}")
    print(f"일조량 범위: {filtered_df['winter_sunlight'].min():.0f} ~ {filtered_df['winter_sunlight'].max():.0f}")
    print(f"SV비율 범위: {filtered_df['sv_ratio'].min():.4f} ~ {filtered_df['sv_ratio'].max():.4f}")
    
    # 액션값 범위
    print(f"Action1 범위: {filtered_df['action1'].min():.4f} ~ {filtered_df['action1'].max():.4f}")
    print(f"Action2 범위: {filtered_df['action2'].min():.4f} ~ {filtered_df['action2'].max():.4f}")
    print(f"Action3 범위: {filtered_df['action3'].min():.4f} ~ {filtered_df['action3'].max():.4f}")
    print(f"Action4 범위: {filtered_df['action4'].min():.4f} ~ {filtered_df['action4'].max():.4f}")
    
    # 보상값 범위
    print(f"보상 범위: {filtered_df['reward'].min():.4f} ~ {filtered_df['reward'].max():.4f}")
    
    # 0값 개수 확인
    print("\n=== 0값 개수 확인 ===")
    print(f"Action1=0: {(filtered_df['action1'] == 0).sum()}개")
    print(f"Action2=0: {(filtered_df['action2'] == 0).sum()}개")
    print(f"Action3=0: {(filtered_df['action3'] == 0).sum()}개")
    print(f"Action4=0: {(filtered_df['action4'] == 0).sum()}개")
    
    # 극단값 개수 확인
    print("\n=== 극단값 개수 확인 (절댓값 > 0.95) ===")
    print(f"Action1>0.95: {(np.abs(filtered_df['action1']) > 0.95).sum()}개")
    print(f"Action2>0.95: {(np.abs(filtered_df['action2']) > 0.95).sum()}개")
    print(f"Action3>0.95: {(np.abs(filtered_df['action3']) > 0.95).sum()}개")
    print(f"Action4>0.95: {(np.abs(filtered_df['action4']) > 0.95).sum()}개")
    
    # 샘플 데이터 출력
    print("\n=== 샘플 데이터 (처음 5개) ===")
    print(filtered_df.head().to_string())
    
    # NaN 확인
    print("\n=== NaN 개수 ===")
    print(filtered_df.isnull().sum())
    
else:
    print("필터링된 데이터가 없습니다!")
    print("\n=== 전체 데이터 샘플 ===")
    print(df.head())