#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
피드백 데이터 전처리 및 중복 제거

목적:
1. 중복된 피드백 쌍 제거
2. 데이터 품질 향상
3. 전처리된 데이터 저장
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import hashlib

class FeedbackDataPreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_feedback = []
        self.unique_feedback = []
        self.duplicate_info = []
        
    def create_pair_hash(self, design_a_id, design_b_id, design_a_state, design_b_state):
        """디자인 쌍의 고유 해시 생성"""
        # ID와 상태를 조합하여 고유성 확보
        pair_data = {
            'ids': tuple(sorted([design_a_id, design_b_id])),
            'states': tuple(sorted([tuple(design_a_state), tuple(design_b_state)]))
        }
        
        # 해시 생성
        pair_str = json.dumps(pair_data, sort_keys=True)
        return hashlib.md5(pair_str.encode()).hexdigest()
    
    def load_and_process_feedback(self):
        """피드백 데이터 로드 및 처리"""
        json_files = list(self.input_dir.glob('*.json'))
        print(f"📂 {len(json_files)}개의 피드백 파일 발견")
        
        seen_hashes = {}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 해시 생성
                pair_hash = self.create_pair_hash(
                    data['design_a_id'],
                    data['design_b_id'],
                    data['design_a_state'],
                    data['design_b_state']
                )
                
                # 중복 체크
                if pair_hash in seen_hashes:
                    # 중복된 경우 - 동일한 선택인지 확인
                    original = seen_hashes[pair_hash]
                    if original['selected_design'] != data['selected_design']:
                        print(f"⚠️ 충돌하는 선택 발견: {json_file.name}")
                        print(f"   원본: {original['selected_design']}")
                        print(f"   현재: {data['selected_design']}")
                    
                    self.duplicate_info.append({
                        'file': json_file.name,
                        'original_file': original['file'],
                        'hash': pair_hash
                    })
                else:
                    # 새로운 피드백
                    seen_hashes[pair_hash] = {
                        'data': data,
                        'file': json_file.name,
                        'selected_design': data['selected_design']
                    }
                    self.unique_feedback.append(data)
                
                self.all_feedback.append(data)
                
            except Exception as e:
                print(f"❌ 파일 로드 실패: {json_file.name} - {e}")
        
        print(f"\n📊 처리 결과:")
        print(f"   전체 피드백: {len(self.all_feedback)}개")
        print(f"   고유한 피드백: {len(self.unique_feedback)}개")
        print(f"   중복 제거됨: {len(self.duplicate_info)}개")
    
    def analyze_unique_data(self):
        """고유 데이터 분석"""
        print("\n🔍 고유 데이터 분석")
        print("="*60)
        
        # 선호/거부 상태 추출
        preferred_states = []
        rejected_states = []
        
        for feedback in self.unique_feedback:
            if feedback['selected_design'] == feedback['design_a_id']:
                preferred_states.append(feedback['design_a_state'])
                rejected_states.append(feedback['design_b_state'])
            else:
                preferred_states.append(feedback['design_b_state'])
                rejected_states.append(feedback['design_a_state'])
        
        preferred_states = np.array(preferred_states)
        rejected_states = np.array(rejected_states)
        
        # 특성별 차이 분석
        feature_names = ['BCR', 'FAR', 'WinterTime', 'SVR']
        state_diffs = preferred_states - rejected_states
        
        print("\n선호-거부 차이 분석 (중복 제거 후):")
        for i, feature in enumerate(feature_names):
            diff_values = state_diffs[:, i]
            print(f"\n{feature}:")
            print(f"  평균 차이: {diff_values.mean():.4f} (±{diff_values.std():.4f})")
            print(f"  선호 방향: 높음 {np.sum(diff_values > 0)}개 ({np.sum(diff_values > 0)/len(diff_values)*100:.1f}%)")
            
            # 효과 크기 (Cohen's d)
            cohens_d = diff_values.mean() / diff_values.std() if diff_values.std() > 0 else 0
            print(f"  효과 크기 (Cohen's d): {cohens_d:.3f}")
    
    def save_processed_data(self):
        """처리된 데이터 저장"""
        # 1. 고유 피드백 저장 (개별 파일)
        unique_dir = self.output_dir / 'unique_feedback'
        unique_dir.mkdir(exist_ok=True)
        
        for i, feedback in enumerate(self.unique_feedback):
            output_file = unique_dir / f"unique_feedback_{i:04d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 {len(self.unique_feedback)}개의 고유 피드백 저장됨: {unique_dir}")
        
        # 2. 통합 파일 저장 (학습용)
        consolidated_file = self.output_dir / 'consolidated_feedback.json'
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(self.unique_feedback, f, indent=2, ensure_ascii=False)
        
        print(f"💾 통합 피드백 파일 저장됨: {consolidated_file}")
        
        # 3. 처리 요약 저장
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_original': len(self.all_feedback),
            'total_unique': len(self.unique_feedback),
            'duplicates_removed': len(self.duplicate_info),
            'duplicate_ratio': len(self.duplicate_info) / len(self.all_feedback) if self.all_feedback else 0,
            'duplicate_details': self.duplicate_info[:10]  # 처음 10개만
        }
        
        summary_file = self.output_dir / 'preprocessing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 전처리 요약 저장됨: {summary_file}")
    
    def create_train_ready_format(self):
        """학습 준비 형식으로 변환"""
        train_ready_data = []
        
        for feedback in self.unique_feedback:
            # 선호/거부 상태 결정
            if feedback['selected_design'] == feedback['design_a_id']:
                preferred = feedback['design_a_state']
                rejected = feedback['design_b_state']
            else:
                preferred = feedback['design_b_state']
                rejected = feedback['design_a_state']
            
            train_ready_data.append({
                'preferred_state': preferred,
                'rejected_state': rejected,
                'original_id': feedback['id']
            })
        
        # 학습용 형식으로 저장
        train_file = self.output_dir / 'train_ready_feedback.json'
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_ready_data, f, indent=2, ensure_ascii=False)
        
        print(f"🎯 학습 준비 데이터 저장됨: {train_file}")
        
        return train_file

def main():
    # 경로 설정
    input_dir = r"C:\Users\valen\Desktop\Dev\6. RLHF\data\feedback"
    output_dir = r"C:\Users\valen\Desktop\Dev\6. RLHF\data\processed_feedback"
    
    print("🔧 피드백 데이터 전처리 시작")
    print("="*80)
    
    # 전처리기 생성
    preprocessor = FeedbackDataPreprocessor(input_dir, output_dir)
    
    # 데이터 로드 및 처리
    preprocessor.load_and_process_feedback()
    
    # 고유 데이터 분석
    preprocessor.analyze_unique_data()
    
    # 처리된 데이터 저장
    preprocessor.save_processed_data()
    
    # 학습 준비 형식 생성
    train_file = preprocessor.create_train_ready_format()
    
    print("\n✅ 전처리 완료!")
    print(f"\n다음 명령어로 개선된 모델 학습을 시작하세요:")
    print(f'python improved_reward_model_trainer.py --feedback-data "{train_file}" --output-dir "improved_models"')

if __name__ == "__main__":
    main()