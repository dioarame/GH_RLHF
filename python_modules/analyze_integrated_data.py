#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 데이터 분석 및 쌍대비교용 기준 데이터 생성 (CSV + JSON 결합 버전)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RLHFDataAnalyzer:
    def __init__(self, csv_path, json_path=None, session_dir=None):
        self.csv_path = csv_path
        self.json_path = json_path
        self.session_dir = session_dir
        
        # 데이터 저장소
        self.csv_data = None
        self.json_data = None
        self.combined_data = None
        
        # 분석 매개변수
        self.bcr_limit = 0.7  # 70% 건폐율 제한
        self.far_min = 2.0    # 200% 최소 용적률
        self.far_max = 5.0    # 500% 최대 용적률
        self.sunlight_min = 80000  # 최소 일조량
        self.svr_optimal = 0.8     # 최적 SV 비율
        
        self._load_data()
    
    def _load_data(self):
        """CSV와 JSON 데이터 로드 및 결합"""
        try:
            print(f"🔍 CSV 데이터 파일 로드 중: {self.csv_path}")
            
            # CSV 파일 로드 (헤더 없음)
            csv_columns = [
                'timestamp', 'step', 'is_closed_brep', 'excluded_from_training',
                'bcr', 'far', 'winter_sunlight', 'sv_ratio', 'reward',
                'action1', 'action2', 'action3', 'action4'
            ]
            
            self.csv_data = pd.read_csv(self.csv_path, header=None, names=csv_columns)
            print(f"✅ CSV 데이터 로드 완료: {len(self.csv_data)} 개 레코드")
            
            # 데이터 필터링 (학습에서 제외되지 않고, closed brep인 것만)
            valid_mask = (
                (self.csv_data['is_closed_brep'] == True) & 
                (self.csv_data['excluded_from_training'] == False)
            )
            
            self.csv_data_filtered = self.csv_data[valid_mask].copy()
            print(f"📊 유효한 데이터: {len(self.csv_data_filtered)} 개 (closed brep & 학습 포함)")
            
            # 상태 및 액션 차원
            self.state_dim = 4  # bcr, far, winter_sunlight, sv_ratio
            self.action_dim = 4  # action1, action2, action3, action4
            
            print(f"📏 상태 차원: {self.state_dim} (BCR, FAR, Winter_Sunlight, SV_Ratio)")
            print(f"🎯 액션 차원: {self.action_dim}")
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류: {e}")
            self.csv_data = pd.DataFrame()
            self.csv_data_filtered = pd.DataFrame()
    
    def is_valid_design(self, design):
        """디자인 유효성 검사"""
        try:
            # 상태 값 체크
            state_values = [
                design['bcr'], design['far'], 
                design['winter_sunlight'], design['sv_ratio']
            ]
            action_values = [
                design['action1'], design['action2'], 
                design['action3'], design['action4']
            ]
            
            # 액션값 범위 체크 (실제 슬라이더 범위에 맞춤)
            # Action1: 10-25 범위
            if not (10 <= design['action1'] <= 25):
                return False
            # Action2: 50-100 범위  
            if not (50 <= design['action2'] <= 100):
                return False
            # Action3, Action4: 0-100 범위 (0값 허용)
            if not (0 <= design['action3'] <= 100):
                return False
            if not (0 <= design['action4'] <= 100):
                return False
            
            # 상태값 범위 체크 (관대하게 조정)
            if not (0.0001 < design['bcr'] < 0.5):  # BCR: 0.01%-50%
                return False
            if not (2.0 < design['far'] < 8.0):  # FAR: 200%-800%
                return False
            if not (50000 < design['winter_sunlight'] < 200000):  # 일조량: 50k-200k
                return False
            if not (0.5 < design['sv_ratio'] < 1.0):  # SV비율: 0.5-1.0
                return False
            
            # NaN, inf 체크
            for val in state_values + action_values + [design['reward']]:
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    return False
            
            # 보상값이 정상 범위인지 체크
            reward = design['reward']
            if reward < -10 or reward > 10:  # 비정상적으로 큰 보상값 제외
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def calculate_quality_score(self, design):
        """디자인 품질 점수 계산"""
        try:
            bcr = design['bcr'] * 100  # 건폐율 (%)
            far = design['far'] * 100  # 용적률 (%)
            sunlight = design['winter_sunlight']   # 일조량
            svr = design['sv_ratio']        # SV 비율
            
            # 각 지표별 점수 계산 (0-1 범위)
            # BCR 점수: 70% 이하면 1.0, 초과하면 감점
            bcr_score = 1.0 if bcr <= 70 else max(0, 1.0 - (bcr - 70) / 30)
            
            # FAR 점수: 200-500% 범위에서 최적
            if 200 <= far <= 500:
                far_score = 1.0
            elif far < 200:
                far_score = far / 200
            else:
                far_score = max(0, 1.0 - (far - 500) / 200)
            
            # 일조량 점수: 80k 이상에서 최적
            sunlight_score = min(1.0, sunlight / 100000) if sunlight > 0 else 0
            
            # SV 비율 점수: 0.8 근처에서 최적
            svr_score = 1.0 - abs(svr - 0.8) / 0.3 if svr > 0 else 0
            svr_score = max(0, min(1.0, svr_score))
            
            # 가중 평균 (건폐율과 용적률이 중요)
            quality_score = (
                bcr_score * 0.3 +
                far_score * 0.3 +
                sunlight_score * 0.2 +
                svr_score * 0.2
            )
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def calculate_composite_score(self, design):
        """복합 점수 계산"""
        try:
            reward = design['reward']
            quality = self.calculate_quality_score(design)
            
            # 보상과 품질을 결합
            return reward * 0.7 + quality * 0.3
                
        except Exception as e:
            return 0.0
    
    def check_legal_compliance(self, design):
        """법적 준수 여부 확인"""
        try:
            bcr = design['bcr'] * 100
            far = design['far'] * 100
            
            return bcr <= 70 and 200 <= far <= 500
        except:
            return False
    
    def generate_rlhf_reference_data(self, output_path):
        """RLHF 쌍대비교용 기준 데이터 생성 (30개 버전)"""
        if self.csv_data_filtered.empty:
            print("❌ 기준 데이터 생성을 위한 데이터가 없습니다.")
            return None
        
        print("2️⃣ RLHF 쌍대비교용 기준 데이터 생성 중...")
        
        # 1. 유효한 디자인만 필터링
        print("1. 유효한 디자인 필터링 중...")
        valid_designs = []
        invalid_count = 0
        
        for idx, design in self.csv_data_filtered.iterrows():
            if self.is_valid_design(design):
                quality_score = self.calculate_quality_score(design)
                composite_score = self.calculate_composite_score(design)
                legal_compliance = self.check_legal_compliance(design)
                
                design_info = {
                    'id': f"design_{design['step']}_{design['timestamp']}",
                    'step': design['step'],
                    'timestamp': design['timestamp'],
                    'state': [design['bcr'], design['far'], design['winter_sunlight'], design['sv_ratio']],
                    'action': [design['action1'], design['action2'], design['action3'], design['action4']],
                    'reward': design['reward'],
                    'quality_score': quality_score,
                    'composite_score': composite_score,
                    'legal_compliance': legal_compliance,
                    'raw_data': {
                        'bcr': design['bcr'],
                        'far': design['far'],
                        'winter_sunlight': design['winter_sunlight'],
                        'sv_ratio': design['sv_ratio']
                    }
                }
                valid_designs.append(design_info)
            else:
                invalid_count += 1
        
        print(f"   유효한 디자인: {len(valid_designs)}개")
        print(f"   필터링된 디자인: {invalid_count}개 (극단값, NaN, 비정상 범위 등)")
        
        if len(valid_designs) < 30:
            print(f"⚠️ 경고: 유효한 디자인이 {len(valid_designs)}개뿐입니다. 가능한 개수로 진행합니다.")
        
        # 2. 상위 점수 20개 선별
        print("2. 상위 점수 20개 디자인 선별 중...")
        top_count = min(20, len(valid_designs))
        top_designs = sorted(valid_designs, key=lambda x: x['composite_score'], reverse=True)[:top_count]
        
        print(f"\n=== 쌍대비교용 상위 {top_count}개 디자인 ===")
        print("복합 점수 구성: 실제보상(0.7) + 품질점수(0.3)")
        
        for i, design in enumerate(top_designs[:10], 1):  # 상위 10개만 출력
            bcr = design['state'][0] * 100
            far = design['state'][1] * 100
            sunlight = design['state'][2] / 1000
            svr = design['state'][3]
            
            print(f"\n디자인 {i}: ID={design['id']}")
            print(f"  복합점수: {design['composite_score']:.4f} (보상: {design['reward']:.4f}, 품질: {design['quality_score']:.4f})")
            print(f"  법적준수: {'예' if design['legal_compliance'] else '아니오'}")
            print(f"  BCR: {bcr:.1f}%, FAR: {far:.1f}%")
            print(f"  일조량: {sunlight:.1f}k kWh, SV비율: {svr:.3f}")
        
        # 3. 랜덤 10개 선별
        print("3. 추가 랜덤 10개 디자인 선별 중...")
        used_ids = set([d['id'] for d in top_designs])
        remaining_designs = [d for d in valid_designs if d['id'] not in used_ids]
        
        random_count = min(10, len(remaining_designs))
        random_designs = []
        
        if len(remaining_designs) > 0:
            # 법적 준수 디자인 우선 선별
            legal_designs = [d for d in remaining_designs if d['legal_compliance']]
            
            if len(legal_designs) >= random_count:
                random_designs = np.random.choice(legal_designs, random_count, replace=False).tolist()
            elif len(remaining_designs) >= random_count:
                # 법적 준수 디자인이 부족하면 전체에서 선별
                random_designs = np.random.choice(remaining_designs, random_count, replace=False).tolist()
            else:
                # 남은 디자인이 부족하면 모두 선택
                random_designs = remaining_designs
        
        # 4. 기준 데이터 구성
        reference_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples': len(self.csv_data),
                'valid_samples': len(self.csv_data_filtered),
                'filtered_samples': len(valid_designs),
                'invalid_samples': invalid_count,
                'state_dimensions': self.state_dim,
                'action_dimensions': self.action_dim,
                'data_sources': {
                    'csv_file': os.path.basename(self.csv_path),
                    'json_file': os.path.basename(self.json_path) if self.json_path else None
                },
                'analysis_parameters': {
                    'bcr_limit': self.bcr_limit,
                    'far_range': [self.far_min, self.far_max],
                    'sunlight_min': self.sunlight_min,
                    'svr_optimal': self.svr_optimal
                },
                'selection_criteria': {
                    'top_designs': top_count,
                    'random_designs': len(random_designs),
                    'total_target': top_count + len(random_designs)
                }
            },
            'top_designs': self.format_designs_for_reference(top_designs),
            'random_designs': self.format_designs_for_reference(random_designs)
        }
        
        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        total_designs = len(top_designs) + len(random_designs)
        possible_pairs = total_designs * (total_designs - 1) // 2
        
        print(f"\n✅ RLHF 쌍대비교 기준 데이터가 {output_path}에 저장되었습니다.")
        print(f"📊 상위 성능 디자인: {len(top_designs)}개")
        print(f"🎲 랜덤 선별 디자인: {len(random_designs)}개")
        print(f"📈 총 비교 대상 디자인: {total_designs}개")
        print(f"🔢 가능한 비교 쌍 수: {possible_pairs:,}개")
        print(f"💡 권장 피드백 수집: {min(400, possible_pairs)}건")
        
        # 보상 분포 통계
        if valid_designs:  # valid_designs가 비어있지 않은 경우에만
            rewards = [d['reward'] for d in valid_designs]
            print(f"\n📈 보상 통계:")
            print(f"   평균: {np.mean(rewards):.4f}")
            print(f"   최소: {np.min(rewards):.4f}")
            print(f"   최대: {np.max(rewards):.4f}")
            print(f"   표준편차: {np.std(rewards):.4f}")
        else:
            print(f"\n📈 보상 통계: 유효한 디자인이 없어 통계를 계산할 수 없습니다.")
        
        return reference_data
    
    def format_designs_for_reference(self, designs):
        """기준 데이터 형식으로 변환"""
        formatted_designs = []
        for design in designs:
            formatted_design = {
                'step': int(design['step']),  # numpy int를 Python int로 변환
                'reward': float(design['reward']),  # numpy float를 Python float로 변환
                'quality_score': float(design['quality_score']),
                'composite_score': float(design['composite_score']),
                'state': [float(x) for x in design['state']],  # numpy array를 Python list로 변환
                'action': [float(x) for x in design['action']],
                'timestamp': int(design['timestamp']),
                'legal_compliance': bool(design['legal_compliance'])  # numpy bool을 Python bool로 변환
            }
            formatted_designs.append(formatted_design)
        
        return formatted_designs

def create_session_directory(base_dir, session_name=None):
    """세션 디렉토리 생성"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if session_name:
        dir_name = f"{session_name}_{timestamp}"
    else:
        dir_name = f"rlhf_session_{timestamp}"
    
    session_dir = os.path.join(base_dir, dir_name)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def find_latest_file(directory, pattern):
    """최신 파일 찾기"""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 데이터 분석 및 쌍대비교 기준 데이터 생성')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='PPO 학습 CSV 파일 경로')
    parser.add_argument('--json-file', type=str, default=None,
                        help='ZMQ 통신 JSON 파일 경로 (선택적)')
    parser.add_argument('--session-name', type=str, default=None,
                        help='세션 이름')
    
    args = parser.parse_args()
    
    # 프로젝트 루트 디렉토리
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # 세션 디렉토리 생성
    session_dir = create_session_directory(data_dir, args.session_name)
    print(f"🔹 RLHF 분석 세션 디렉토리: {session_dir}")
    
    # CSV 파일 찾기
    csv_path = args.csv_file
    if not csv_path:
        zmq_logs_dir = os.path.join(data_dir, "zmq_logs")
        csv_path = find_latest_file(zmq_logs_dir, "architecture_metrics_*.csv")
        
        if not csv_path:
            print("❌ 오류: PPO 학습 CSV 파일을 찾을 수 없습니다.")
            return
    
    print(f"📂 사용할 CSV 파일: {csv_path}")
    
    # JSON 파일 찾기 (선택적)
    json_path = args.json_file
    if not json_path:
        zmq_logs_dir = os.path.join(data_dir, "zmq_logs")
        json_path = find_latest_file(zmq_logs_dir, "state_reward_log_*.json")
        if json_path:
            print(f"📂 참조용 JSON 파일: {json_path}")
        else:
            print("📂 JSON 파일 없음 (CSV 파일만 사용)")
    
    # 분석기 생성
    analyzer = RLHFDataAnalyzer(csv_path, json_path, session_dir)
    
    if analyzer.csv_data_filtered.empty:
        print("❌ 오류: 분석할 데이터가 없습니다.")
        return
    
    print(f"📊 로드된 데이터: {len(analyzer.csv_data_filtered)}개 유효한 디자인")
    
    # 1. 처리된 데이터 저장
    print("1️⃣ 처리된 데이터 저장 중...")
    processed_data_file = os.path.join(session_dir, "processed_rlhf_data.csv")
    analyzer.csv_data_filtered.to_csv(processed_data_file, index=False)
    print(f"처리된 데이터가 {processed_data_file}에 저장되었습니다.")
    
    # 2. RLHF 기준 데이터 생성
    reference_data_file = os.path.join(session_dir, "rlhf_reference_data.json")
    reference_data = analyzer.generate_rlhf_reference_data(reference_data_file)
    
    if reference_data:
        print(f"\n✅ RLHF 쌍대비교 데이터 준비 완료!")
        print(f"📁 세션 디렉토리: {session_dir}")
        print(f"📄 기준 데이터: {reference_data_file}")
        print(f"📈 처리된 데이터: {processed_data_file}")

if __name__ == "__main__":
    main()