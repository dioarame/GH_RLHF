#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF 피드백 프로세서

이 모듈은 서버에서 수집된 인간 피드백 데이터를 처리하여
보상 모델 훈련을 위한 선호도 쌍을 생성합니다.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
import logging
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedback_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """
    인간 피드백 처리 클래스
    
    이 클래스는 다음 작업을 수행합니다:
    1. 피드백 파일 로드
    2. 디자인 데이터와 연결
    3. 선호도 쌍 생성
    4. 훈련 데이터셋 준비
    """
    
    def __init__(self, feedback_dir=None, designs_dir=None, output_dir=None, base_dir=None):
        """
        초기화 함수
        
        Args:
            feedback_dir: 피드백 JSON 파일 디렉토리
            designs_dir: 디자인 JSON 파일 디렉토리
            output_dir: 처리된 데이터 저장 디렉토리
            base_dir: 프로젝트 기본 디렉토리 (지정하지 않으면 자동 탐지)
        """
        # 기본 디렉토리 설정
        if base_dir is None:
            # 현재 모듈이 있는 디렉토리의 부모 디렉토리를 프로젝트 루트로 사용
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 디렉토리 설정
        self.base_dir = base_dir
        self.feedback_dir = feedback_dir or os.path.join(base_dir, "data", "feedback")
        self.designs_dir = designs_dir or os.path.join(base_dir, "data", "designs")
        self.output_dir = output_dir or os.path.join(base_dir, "data", "processed_feedback")
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 데이터 저장소
        self.feedback_data = []
        self.designs_data = {}
        self.preference_pairs = []
        
        # 평가 차원 가중치 (기본값)
        self.rating_weights = {
            "aesthetic": 1.0,
            "functionality": 1.0, 
            "innovation": 1.0,
            "feasibility": 1.0,
            "overall": 2.0  # overall에 더 높은 가중치 부여
        }
        
        logger.info(f"피드백 프로세서 초기화 완료. 경로: feedback={self.feedback_dir}, designs={self.designs_dir}, output={self.output_dir}")
    
    def set_rating_weights(self, weights_dict):
        """
        평가 차원별 가중치 설정
        
        Args:
            weights_dict: 차원별 가중치 딕셔너리 (예: {"aesthetic": 2.0, "overall": 3.0})
        """
        for key, value in weights_dict.items():
            if key in self.rating_weights:
                self.rating_weights[key] = float(value)
        
        logger.info(f"평가 가중치 업데이트됨: {self.rating_weights}")
    
    def load_feedback_data(self):
        """
        모든 피드백 파일 로드
        """
        feedback_files = glob.glob(os.path.join(self.feedback_dir, "*.json"))
        logger.info(f"{len(feedback_files)}개 피드백 파일 발견")
        
        self.feedback_data = []
        
        for file_path in feedback_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    feedback = json.load(f)
                    
                    # 필수 필드 확인
                    if "design_id" not in feedback or "ratings" not in feedback:
                        logger.warning(f"필수 필드 누락: {file_path}")
                        continue
                    
                    # 유효한 피드백인 경우 추가
                    self.feedback_data.append(feedback)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류 ({file_path}): {e}")
            except Exception as e:
                logger.error(f"파일 로드 중 오류 ({file_path}): {e}")
        
        logger.info(f"{len(self.feedback_data)}개 유효한 피드백 로드됨")
        return len(self.feedback_data)
    
    def load_designs_data(self):
        """
        피드백에서 참조된 디자인 파일 로드
        """
        # 참조된 디자인 ID 수집
        design_ids = {f["design_id"] for f in self.feedback_data if "design_id" in f}
        logger.info(f"{len(design_ids)}개 고유 디자인 ID 발견")
        
        # 디자인 파일 로드
        for design_id in design_ids:
            design_file = os.path.join(self.designs_dir, f"{design_id}.json")
            
            if not os.path.exists(design_file):
                logger.warning(f"디자인 파일 없음: {design_id}")
                continue
            
            try:
                with open(design_file, 'r', encoding='utf-8') as f:
                    design_data = json.load(f)
                    
                    # NaN 값 처리
                    design_data = self._handle_nan_values(design_data)
                    
                    # 필수 필드 확인
                    if "state" not in design_data or "action" not in design_data:
                        logger.warning(f"디자인 파일에 필수 필드 누락: {design_id}")
                        continue
                    
                    # 유효한 디자인 데이터 저장
                    self.designs_data[design_id] = design_data
                    
            except json.JSONDecodeError as e:
                logger.error(f"디자인 JSON 파싱 오류 ({design_id}): {e}")
            except Exception as e:
                logger.error(f"디자인 파일 로드 중 오류 ({design_id}): {e}")
        
        logger.info(f"{len(self.designs_data)}개 디자인 데이터 로드됨")
        return len(self.designs_data)
    
    def _handle_nan_values(self, data):
        """
        JSON 직렬화 전에 NaN 값을 None으로 변환하는 재귀 함수
        """
        if isinstance(data, dict):
            return {k: self._handle_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._handle_nan_values(item) for item in data]
        elif isinstance(data, float) and np.isnan(data):
            return None
        else:
            return data
    
    def _calculate_weighted_score(self, ratings):
        """
        가중치를 적용한 종합 점수 계산
        
        Args:
            ratings: 평가 딕셔너리
            
        Returns:
            float: 가중치가 적용된 종합 점수
        """
        total_weight = 0
        weighted_sum = 0
        
        for dimension, weight in self.rating_weights.items():
            if dimension in ratings:
                weighted_sum += float(ratings[dimension]) * weight
                total_weight += weight
        
        # 분모가 0이면 0 반환
        if total_weight == 0:
            return 0
            
        return weighted_sum / total_weight
    
    def generate_preference_pairs(self, min_score_diff=0.5, max_pairs_per_design=5):
        """
        피드백 데이터로부터 선호도 쌍 생성
        
        Args:
            min_score_diff: 선호도 쌍으로 간주할 최소 점수 차이
            max_pairs_per_design: 각 디자인에 대해 생성할 최대 쌍 수
            
        Returns:
            int: 생성된 선호도 쌍 수
        """
        if not self.feedback_data or not self.designs_data:
            logger.error("선호도 쌍 생성을 위한 데이터가 부족합니다")
            return 0
        
        # 디자인별 피드백 그룹화
        design_feedbacks = {}
        
        for feedback in self.feedback_data:
            design_id = feedback["design_id"]
            
            if design_id not in self.designs_data:
                continue
                
            if design_id not in design_feedbacks:
                design_feedbacks[design_id] = []
                
            # 가중치를 적용한 종합 점수 계산
            feedback["weighted_score"] = self._calculate_weighted_score(feedback["ratings"])
            design_feedbacks[design_id].append(feedback)
        
        # 디자인 쌍 생성
        all_designs = list(self.designs_data.keys())
        self.preference_pairs = []
        
        # 각 디자인에 대해
        for idx, design_id in enumerate(all_designs):
            if design_id not in design_feedbacks:
                continue
                
            # 현재 디자인의 피드백 목록
            curr_feedbacks = design_feedbacks[design_id]
            
            # 현재 디자인의 평균 점수 계산
            curr_scores = [f["weighted_score"] for f in curr_feedbacks]
            curr_avg_score = sum(curr_scores) / len(curr_scores) if curr_scores else 0
            
            # 비교할 디자인 쌍 생성
            pairs_for_design = 0
            
            # 다른 모든 디자인과 비교
            for other_id in all_designs:
                if other_id == design_id or other_id not in design_feedbacks:
                    continue
                    
                # 비교 디자인의 피드백 목록
                other_feedbacks = design_feedbacks[other_id]
                
                # 비교 디자인의 평균 점수 계산
                other_scores = [f["weighted_score"] for f in other_feedbacks]
                other_avg_score = sum(other_scores) / len(other_scores) if other_scores else 0
                
                # 점수 차이 계산
                score_diff = curr_avg_score - other_avg_score
                
                # 점수 차이가 임계값을 넘으면 선호도 쌍 생성
                if abs(score_diff) >= min_score_diff:
                    # 선호되는 디자인과 덜 선호되는 디자인 결정
                    if score_diff > 0:
                        preferred_id, less_preferred_id = design_id, other_id
                    else:
                        preferred_id, less_preferred_id = other_id, design_id
                    
                    # 선호도 쌍 추가
                    pair = {
                        "preferred": {
                            "design_id": preferred_id,
                            "state": self.designs_data[preferred_id]["state"],
                            "action": self.designs_data[preferred_id]["action"],
                            "score": max(curr_avg_score, other_avg_score)
                        },
                        "less_preferred": {
                            "design_id": less_preferred_id,
                            "state": self.designs_data[less_preferred_id]["state"],
                            "action": self.designs_data[less_preferred_id]["action"],
                            "score": min(curr_avg_score, other_avg_score)
                        },
                        "score_diff": abs(score_diff)
                    }
                    
                    self.preference_pairs.append(pair)
                    pairs_for_design += 1
                    
                    # 최대 쌍 수 제한
                    if pairs_for_design >= max_pairs_per_design:
                        break
        
        # 점수 차이를 기준으로 쌍 정렬 (차이가 큰 것이 더 중요)
        self.preference_pairs.sort(key=lambda p: p["score_diff"], reverse=True)
        
        logger.info(f"{len(self.preference_pairs)}개 선호도 쌍 생성됨")
        return len(self.preference_pairs)
    
    def export_preference_dataset(self, train_ratio=0.8, add_timestamp=True):
        """
        선호도 데이터셋 생성 및 내보내기
        
        Args:
            train_ratio: 훈련/검증 분할 비율
            add_timestamp: 파일명에 타임스탬프 추가 여부
            
        Returns:
            tuple: (훈련 데이터 파일 경로, 검증 데이터 파일 경로)
        """
        if not self.preference_pairs:
            logger.error("내보낼 선호도 쌍이 없습니다")
            return None, None
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
        
        # 파일 경로 생성
        train_file = os.path.join(self.output_dir, f"preference_train_{timestamp}.json")
        valid_file = os.path.join(self.output_dir, f"preference_valid_{timestamp}.json")
        
        # 데이터 셔플 및 분할
        random.shuffle(self.preference_pairs)
        split_idx = int(len(self.preference_pairs) * train_ratio)
        
        train_pairs = self.preference_pairs[:split_idx]
        valid_pairs = self.preference_pairs[split_idx:]
        
        # 훈련 데이터 저장
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "pairs_count": len(train_pairs),
                    "rating_weights": self.rating_weights
                },
                "preference_pairs": train_pairs
            }, f, indent=2)
        
        # 검증 데이터 저장
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "pairs_count": len(valid_pairs),
                    "rating_weights": self.rating_weights
                },
                "preference_pairs": valid_pairs
            }, f, indent=2)
        
        logger.info(f"선호도 데이터셋 내보내기 완료: 훈련={len(train_pairs)}개, 검증={len(valid_pairs)}개")
        logger.info(f"파일 저장됨: {train_file}, {valid_file}")
        
        return train_file, valid_file
    
    def generate_reward_dataset(self):
        """
        보상 모델 훈련을 위한 전체 데이터셋 생성
        
        Returns:
            tuple: (훈련 데이터 파일 경로, 검증 데이터 파일 경로)
        """
        # 1. 피드백 데이터 로드
        feedback_count = self.load_feedback_data()
        if feedback_count == 0:
            logger.error("피드백 데이터를 찾을 수 없습니다")
            return None, None
        
        # 2. 디자인 데이터 로드
        design_count = self.load_designs_data()
        if design_count == 0:
            logger.error("디자인 데이터를 찾을 수 없습니다")
            return None, None
        
        # 3. 선호도 쌍 생성
        pair_count = self.generate_preference_pairs()
        if pair_count == 0:
            logger.error("선호도 쌍을 생성할 수 없습니다")
            return None, None
        
        # 4. 데이터셋 내보내기
        return self.export_preference_dataset()
    
    def generate_report(self, output_file=None):
        """
        피드백 데이터 분석 보고서 생성
        
        Args:
            output_file: 보고서 저장 파일 경로 (기본값: output_dir/feedback_report_*.json)
            
        Returns:
            str: 보고서 파일 경로
        """
        if not self.feedback_data:
            logger.error("보고서를 생성할 피드백 데이터가 없습니다")
            return None
        
        # 기본 파일 경로 생성
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"feedback_report_{timestamp}.json")
        
        # 디자인별 평균 점수
        design_scores = {}
        for feedback in self.feedback_data:
            design_id = feedback.get("design_id")
            if design_id and "ratings" in feedback:
                if design_id not in design_scores:
                    design_scores[design_id] = {
                        "count": 0,
                        "total_scores": {dim: 0 for dim in self.rating_weights},
                        "weighted_total": 0
                    }
                
                # 각 차원별 점수 합산
                for dim, weight in self.rating_weights.items():
                    if dim in feedback["ratings"]:
                        design_scores[design_id]["total_scores"][dim] += feedback["ratings"][dim]
                
                # 가중 점수 계산 및 추가
                weighted_score = self._calculate_weighted_score(feedback["ratings"])
                design_scores[design_id]["weighted_total"] += weighted_score
                design_scores[design_id]["count"] += 1
        
        # 평균 계산
        for design_id, scores in design_scores.items():
            count = scores["count"]
            if count > 0:
                scores["avg_scores"] = {
                    dim: scores["total_scores"][dim] / count 
                    for dim in self.rating_weights
                }
                scores["weighted_avg"] = scores["weighted_total"] / count
                
                # 불필요한 필드 제거
                del scores["total_scores"]
                del scores["weighted_total"]
        
        # 차원별 전체 평균
        dimension_stats = {}
        for dim in self.rating_weights:
            all_scores = [
                feedback["ratings"].get(dim) 
                for feedback in self.feedback_data 
                if "ratings" in feedback and dim in feedback["ratings"]
            ]
            
            if all_scores:
                dimension_stats[dim] = {
                    "count": len(all_scores),
                    "min": min(all_scores),
                    "max": max(all_scores),
                    "mean": sum(all_scores) / len(all_scores),
                    "weight": self.rating_weights[dim]
                }
        
        # 보고서 데이터 구성
        report_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "feedback_count": len(self.feedback_data),
                "design_count": len(design_scores),
                "pair_count": len(self.preference_pairs)
            },
            "rating_dimensions": dimension_stats,
            "design_scores": design_scores,
            "top_designs": sorted(
                [
                    {"design_id": design_id, "score": data["weighted_avg"], "feedback_count": data["count"]}
                    for design_id, data in design_scores.items()
                ],
                key=lambda x: x["score"],
                reverse=True
            )[:10]  # 상위 10개만
        }
        
        # 보고서 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"피드백 분석 보고서 생성 완료: {output_file}")
        return output_file

def main():
    """
    커맨드 라인 인터페이스
    """
    parser = argparse.ArgumentParser(description='인간 피드백 처리 및 선호도 데이터셋 생성')
    parser.add_argument('--feedback-dir', type=str, default=None,
                        help='피드백 JSON 파일 디렉토리')
    parser.add_argument('--designs-dir', type=str, default=None,
                        help='디자인 JSON 파일 디렉토리')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='처리된 데이터 저장 디렉토리')
    parser.add_argument('--weight-aesthetic', type=float, default=1.0,
                        help='심미적 품질 평가 가중치 (기본값: 1.0)')
    parser.add_argument('--weight-functionality', type=float, default=1.0,
                        help='기능성 평가 가중치 (기본값: 1.0)')
    parser.add_argument('--weight-innovation', type=float, default=1.0,
                        help='혁신성 평가 가중치 (기본값: 1.0)')
    parser.add_argument('--weight-feasibility', type=float, default=1.0,
                        help='실현 가능성 평가 가중치 (기본값: 1.0)')
    parser.add_argument('--weight-overall', type=float, default=2.0,
                        help='전체적인 평가 가중치 (기본값: 2.0)')
    parser.add_argument('--min-score-diff', type=float, default=0.5,
                        help='선호도 쌍으로 간주할 최소 점수 차이 (기본값: 0.5)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='훈련/검증 분할 비율 (기본값: 0.8)')
    parser.add_argument('--report-only', action='store_true',
                        help='데이터셋 생성 없이 보고서만 생성')
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = FeedbackProcessor(
        feedback_dir=args.feedback_dir,
        designs_dir=args.designs_dir,
        output_dir=args.output_dir
    )
    
    # 가중치 설정
    processor.set_rating_weights({
        "aesthetic": args.weight_aesthetic,
        "functionality": args.weight_functionality,
        "innovation": args.weight_innovation,
        "feasibility": args.weight_feasibility,
        "overall": args.weight_overall
    })
    
    # 보고서만 생성하는 경우
    if args.report_only:
        processor.load_feedback_data()
        processor.load_designs_data()
        report_file = processor.generate_report()
        if report_file:
            print(f"\n✅ 보고서가 생성되었습니다: {report_file}")
        return
    
    # 전체 데이터셋 생성
    train_file, valid_file = processor.generate_reward_dataset()
    
    if train_file and valid_file:
        print(f"\n✅ 보상 모델 데이터셋이 생성되었습니다:")
        print(f"   - 훈련 데이터: {train_file}")
        print(f"   - 검증 데이터: {valid_file}")
        
        # 보고서 생성
        report_file = processor.generate_report()
        if report_file:
            print(f"   - 분석 보고서: {report_file}")
    else:
        print("\n❌ 데이터셋 생성에 실패했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    main()
