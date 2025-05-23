#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
개선된 피드백 데이터 처리기

웹 인터페이스에서 수집된 인간 피드백 데이터를 
RLHF 학습에 적합한 형식으로 변환하고 전처리합니다.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackDataProcessor:
    """피드백 데이터 처리 클래스"""
    
    def __init__(self, data_dir='data', output_dir='processed_feedback'):
        """
        Args:
            data_dir: 원본 데이터 디렉토리
            output_dir: 처리된 데이터 출력 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장소
        self.raw_feedback_data = []
        self.processed_pairs = []
        self.session_stats = {}
        
        logger.info(f"피드백 데이터 처리기 초기화 완료")
        logger.info(f"입력 디렉토리: {self.data_dir}")
        logger.info(f"출력 디렉토리: {self.output_dir}")
    
    def collect_feedback_files(self) -> List[Path]:
        """피드백 데이터 파일들을 수집"""
        feedback_files = []
        
        # 여러 가능한 위치에서 피드백 파일 검색
        search_paths = [
            self.data_dir / 'feedback',
            self.data_dir,
            Path.cwd() / 'data' / 'feedback',
            Path.cwd() / 'server' / 'data' / 'feedback'
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                # JSON 파일 패턴 검색
                patterns = [
                    '*feedback*.json',
                    'rlhf_feedback_data_*.json',
                    'session_*.json'
                ]
                
                for pattern in patterns:
                    files = list(search_path.glob(pattern))
                    feedback_files.extend(files)
        
        # 중복 제거
        feedback_files = list(set(feedback_files))
        
        logger.info(f"발견된 피드백 파일: {len(feedback_files)}개")
        for file in feedback_files:
            logger.info(f"  - {file}")
        
        return feedback_files
    
    def load_feedback_data(self, feedback_files: List[Path]) -> List[Dict]:
        """피드백 파일들을 로드하고 통합"""
        all_feedback_data = []
        
        for file_path in feedback_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 형식 확인 및 변환
                if isinstance(data, dict):
                    # 새로운 형식 (preference_pairs 포함)
                    if 'preference_pairs' in data:
                        pairs = data['preference_pairs']
                        metadata = data.get('metadata', {})
                        
                        for pair in pairs:
                            pair['source_file'] = str(file_path)
                            pair['session_metadata'] = metadata
                        
                        all_feedback_data.extend(pairs)
                    
                    # 개별 피드백 데이터
                    elif 'design_a_id' in data and 'design_b_id' in data:
                        data['source_file'] = str(file_path)
                        all_feedback_data.append(data)
                
                elif isinstance(data, list):
                    # 리스트 형식의 피드백 데이터
                    for item in data:
                        if isinstance(item, dict):
                            item['source_file'] = str(file_path)
                            all_feedback_data.append(item)
                
                logger.info(f"파일 로드 성공: {file_path.name}")
                
            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")
                continue
        
        logger.info(f"총 로드된 피드백 데이터: {len(all_feedback_data)}개")
        return all_feedback_data
    
    def validate_feedback_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """피드백 데이터 유효성 검사 및 정리"""
        valid_data = []
        
        for i, item in enumerate(feedback_data):
            try:
                # 필수 필드 확인
                required_fields = self.get_required_fields(item)
                
                if not all(field in item for field in required_fields):
                    logger.warning(f"데이터 {i}: 필수 필드 누락")
                    continue
                
                # 상태 데이터 유효성 검사
                if not self.validate_state_data(item):
                    logger.warning(f"데이터 {i}: 상태 데이터 무효")
                    continue
                
                valid_data.append(item)
                
            except Exception as e:
                logger.error(f"데이터 {i} 유효성 검사 오류: {e}")
                continue
        
        logger.info(f"유효한 피드백 데이터: {len(valid_data)}개")
        return valid_data
    
    def get_required_fields(self, item: Dict) -> List[str]:
        """데이터 형식에 따른 필수 필드 반환"""
        if 'preferred_state' in item and 'rejected_state' in item:
            # 새로운 preference pair 형식
            return ['preferred_state', 'rejected_state']
        else:
            # 기존 비교 형식
            return ['design_a_state', 'design_b_state', 'selected_design']
    
    def validate_state_data(self, item: Dict) -> bool:
        """상태 데이터 유효성 검사"""
        try:
            if 'preferred_state' in item:
                # 새로운 형식
                preferred = item['preferred_state']
                rejected = item['rejected_state']
                
                if not (isinstance(preferred, list) and isinstance(rejected, list)):
                    return False
                
                if len(preferred) < 4 or len(rejected) < 4:
                    return False
                
                # NaN 및 무한값 검사
                if any(not np.isfinite(x) for x in preferred + rejected):
                    return False
            
            else:
                # 기존 형식
                state_a = item.get('design_a_state', [])
                state_b = item.get('design_b_state', [])
                
                if not (isinstance(state_a, list) and isinstance(state_b, list)):
                    return False
                
                if len(state_a) < 4 or len(state_b) < 4:
                    return False
                
                if any(not np.isfinite(x) for x in state_a + state_b):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def convert_to_preference_pairs(self, feedback_data: List[Dict]) -> List[Dict]:
        """피드백 데이터를 선호도 쌍으로 변환"""
        preference_pairs = []
        
        for item in feedback_data:
            try:
                if 'preferred_state' in item:
                    # 이미 preference pair 형식
                    pair = {
                        'preferred_state': item['preferred_state'][:4],
                        'rejected_state': item['rejected_state'][:4],
                        'timestamp': item.get('timestamp', 0),
                        'session_info': item.get('session_info', {}),
                        'source_file': item.get('source_file', ''),
                        'metadata': item.get('session_metadata', {})
                    }
                
                else:
                    # 기존 비교 형식에서 변환
                    state_a = item.get('design_a_state', [])[:4]
                    state_b = item.get('design_b_state', [])[:4]
                    selected = item.get('selected_design', '')
                    
                    if selected == item.get('design_a_id', ''):
                        # A가 선택됨
                        preferred_state = state_a
                        rejected_state = state_b
                    else:
                        # B가 선택됨
                        preferred_state = state_b
                        rejected_state = state_a
                    
                    pair = {
                        'preferred_state': preferred_state,
                        'rejected_state': rejected_state,
                        'timestamp': item.get('comparison_time', 0),
                        'session_info': {
                            'session_id': item.get('session_id', ''),
                            'design_a_id': item.get('design_a_id', ''),
                            'design_b_id': item.get('design_b_id', ''),
                            'selected_design': selected
                        },
                        'source_file': item.get('source_file', ''),
                        'metadata': item.get('session_metadata', {})
                    }
                
                preference_pairs.append(pair)
                
            except Exception as e:
                logger.error(f"선호도 쌍 변환 오류: {e}")
                continue
        
        logger.info(f"변환된 선호도 쌍: {len(preference_pairs)}개")
        return preference_pairs
    
    def apply_quality_filters(self, preference_pairs: List[Dict]) -> List[Dict]:
        """데이터 품질 필터 적용"""
        filtered_pairs = []
        
        for pair in preference_pairs:
            try:
                preferred = np.array(pair['preferred_state'])
                rejected = np.array(pair['rejected_state'])
                
                # 1. 상태 차이가 너무 작은 경우 제외
                state_diff = np.abs(preferred - rejected)
                if np.sum(state_diff) < 0.01:  # 임계값 조정 가능
                    continue
                
                # 2. 극단적인 값 제외
                all_values = np.concatenate([preferred, rejected])
                if np.any(np.abs(all_values) > 100):  # 임계값 조정 가능
                    continue
                
                # 3. 건축적 제약 조건 확인
                if not self.check_architectural_constraints(preferred, rejected):
                    continue
                
                filtered_pairs.append(pair)
                
            except Exception as e:
                logger.error(f"품질 필터 적용 오류: {e}")
                continue
        
        logger.info(f"품질 필터 적용 후: {len(filtered_pairs)}개")
        return filtered_pairs
    
    def check_architectural_constraints(self, preferred: np.ndarray, rejected: np.ndarray) -> bool:
        """건축적 제약 조건 확인"""
        try:
            # BCR (건폐율) 체크 - 0~1 범위
            if not (0 <= preferred[0] <= 1 and 0 <= rejected[0] <= 1):
                return False
            
            # FAR (용적률) 체크 - 0~10 범위 (일반적)
            if not (0 <= preferred[1] <= 10 and 0 <= rejected[1] <= 10):
                return False
            
            # 일조량 체크 - 양수
            if preferred[2] < 0 or rejected[2] < 0:
                return False
            
            # SV Ratio 체크 - 0.1~2.0 범위 (일반적)
            if not (0.1 <= preferred[3] <= 2.0 and 0.1 <= rejected[3] <= 2.0):
                return False
            
            return True
            
        except Exception:
            return False
    
    def analyze_session_statistics(self, preference_pairs: List[Dict]):
        """세션별 통계 분석"""
        session_stats = defaultdict(list)
        
        for pair in preference_pairs:
            session_info = pair.get('session_info', {})
            session_id = session_info.get('session_id', 'unknown')
            session_stats[session_id].append(pair)
        
        # 통계 계산
        self.session_stats = {}
        for session_id, pairs in session_stats.items():
            preferred_states = [pair['preferred_state'] for pair in pairs]
            rejected_states = [pair['rejected_state'] for pair in pairs]
            
            all_states = np.array(preferred_states + rejected_states)
            
            self.session_stats[session_id] = {
                'total_pairs': len(pairs),
                'state_mean': np.mean(all_states, axis=0).tolist(),
                'state_std': np.std(all_states, axis=0).tolist(),
                'timestamps': [pair.get('timestamp', 0) for pair in pairs]
            }
        
        logger.info(f"분석된 세션: {len(self.session_stats)}개")
        for session_id, stats in self.session_stats.items():
            logger.info(f"  {session_id}: {stats['total_pairs']}개 선호도 쌍")
    
    def create_training_dataset(self, preference_pairs: List[Dict]) -> Dict:
        """학습용 데이터셋 생성"""
        # 상태 데이터 추출
        preferred_states = [pair['preferred_state'] for pair in preference_pairs]
        rejected_states = [pair['rejected_state'] for pair in preference_pairs]
        
        # 데이터셋 메타데이터
        metadata = {
            'format_version': '2.0',
            'data_format': 'preference_pairs',
            'created_at': datetime.now().isoformat(),
            'total_pairs': len(preference_pairs),
            'state_dimensions': 4,
            'state_labels': ['BCR', 'FAR', 'Sunlight', 'SV_Ratio'],
            'sessions': len(self.session_stats),
            'session_stats': self.session_stats
        }
        
        # 데이터 통계
        all_states = np.array(preferred_states + rejected_states)
        
        statistics = {
            'state_mean': np.mean(all_states, axis=0).tolist(),
            'state_std': np.std(all_states, axis=0).tolist(),
            'state_min': np.min(all_states, axis=0).tolist(),
            'state_max': np.max(all_states, axis=0).tolist(),
            'state_percentiles': {
                '25': np.percentile(all_states, 25, axis=0).tolist(),
                '50': np.percentile(all_states, 50, axis=0).tolist(),
                '75': np.percentile(all_states, 75, axis=0).tolist()
            }
        }
        
        # 최종 데이터셋
        dataset = {
            'metadata': metadata,
            'statistics': statistics,
            'preference_pairs': preference_pairs
        }
        
        return dataset
    
    def save_processed_data(self, dataset: Dict, filename: str = None):
        """처리된 데이터 저장"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'processed_feedback_data_{timestamp}.json'
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"처리된 데이터 저장됨: {output_path}")
        
        # 요약 정보도 저장
        summary_path = self.output_dir / f'summary_{filename}'
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_preference_pairs': dataset['metadata']['total_pairs'],
            'sessions': dataset['metadata']['sessions'],
            'state_statistics': dataset['statistics'],
            'output_file': str(output_path)
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def process_all_feedback_data(self) -> str:
        """전체 피드백 데이터 처리 파이프라인"""
        logger.info("피드백 데이터 처리 시작")
        
        try:
            # 1. 피드백 파일 수집
            feedback_files = self.collect_feedback_files()
            
            if not feedback_files:
                logger.warning("처리할 피드백 파일이 없습니다.")
                return None
            
            # 2. 데이터 로드
            raw_data = self.load_feedback_data(feedback_files)
            
            # 3. 데이터 유효성 검사
            valid_data = self.validate_feedback_data(raw_data)
            
            # 4. 선호도 쌍으로 변환
            preference_pairs = self.convert_to_preference_pairs(valid_data)
            
            # 5. 품질 필터 적용
            filtered_pairs = self.apply_quality_filters(preference_pairs)
            
            # 6. 세션 통계 분석
            self.analyze_session_statistics(filtered_pairs)
            
            # 7. 학습용 데이터셋 생성
            dataset = self.create_training_dataset(filtered_pairs)
            
            # 8. 데이터 저장
            output_path = self.save_processed_data(dataset)
            
            logger.info("피드백 데이터 처리 완료!")
            logger.info(f"최종 데이터셋: {len(filtered_pairs)}개 선호도 쌍")
            logger.info(f"출력 파일: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"피드백 데이터 처리 중 오류 발생: {e}")
            raise
    
    def generate_training_report(self, dataset: Dict) -> str:
        """학습 데이터셋 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("RLHF 피드백 데이터 처리 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 기본 정보
        metadata = dataset['metadata']
        report.append(f"생성 시간: {metadata['created_at']}")
        report.append(f"데이터 형식: {metadata['data_format']} v{metadata['format_version']}")
        report.append(f"총 선호도 쌍: {metadata['total_pairs']}개")
        report.append(f"세션 수: {metadata['sessions']}개")
        report.append("")
        
        # 상태 통계
        stats = dataset['statistics']
        report.append("상태 벡터 통계:")
        labels = metadata['state_labels']
        
        for i, label in enumerate(labels):
            report.append(f"  {label}:")
            report.append(f"    평균: {stats['state_mean'][i]:.4f}")
            report.append(f"    표준편차: {stats['state_std'][i]:.4f}")
            report.append(f"    범위: [{stats['state_min'][i]:.4f}, {stats['state_max'][i]:.4f}]")
            report.append(f"    중앙값: {stats['state_percentiles']['50'][i]:.4f}")
        
        report.append("")
        
        # 세션별 통계
        report.append("세션별 통계:")
        for session_id, session_stats in metadata['session_stats'].items():
            report.append(f"  {session_id}: {session_stats['total_pairs']}개 쌍")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLHF 피드백 데이터 처리')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='입력 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='processed_feedback',
                       help='출력 디렉토리')
    parser.add_argument('--output-file', type=str, default=None,
                       help='출력 파일명 (기본: 자동 생성)')
    parser.add_argument('--generate-report', action='store_true',
                       help='처리 리포트 생성')
    parser.add_argument('--min-pairs', type=int, default=5,
                       help='최소 선호도 쌍 수')
    
    args = parser.parse_args()
    
    try:
        # 피드백 데이터 처리기 생성
        processor = FeedbackDataProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # 전체 처리 파이프라인 실행
        output_path = processor.process_all_feedback_data()
        
        if output_path is None:
            logger.error("처리할 피드백 데이터가 없습니다.")
            return 1
        
        # 최소 데이터 수 체크
        with open(output_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if dataset['metadata']['total_pairs'] < args.min_pairs:
            logger.warning(f"선호도 쌍이 너무 적습니다: {dataset['metadata']['total_pairs']} < {args.min_pairs}")
            logger.warning("더 많은 피드백 데이터를 수집해주세요.")
        
        # 리포트 생성
        if args.generate_report:
            report = processor.generate_training_report(dataset)
            report_path = Path(args.output_dir) / f"report_{Path(output_path).stem}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(report)
            logger.info(f"리포트 저장됨: {report_path}")
        
        logger.info(f"피드백 데이터 처리 완료: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        return 1

if __name__ == "__main__":
    exit(main())