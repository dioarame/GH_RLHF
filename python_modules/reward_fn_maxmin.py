#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 최대/최소 방향성 보상 함수 모듈 (SVR 및 일사량 범위 수정 버전)
입력: BCR, FAR, 겨울철 일사량(WinterTime), 표면적 체적비(SVR)
목표:
- BCR: 법적 제한 내에서 최대화 (경제적 효율성)
- FAR: 법적 제한 내에서 최대화 (건축 밀도 최대화)
- WinterTime: 최대화 (겨울철 일사량 확보, 상한 제거)
- SVR: 목표 범위(0.7~0.9) 내에서 최적값(0.8) 지향

실제 생성 범위를 반영한 현실적인 방향성 보상 함수
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class MaxMinDirectionalRewardFunction:
    """
    실제 생성 범위를 반영한 개선된 최대/최소 방향성 보상 함수
    경제적 효율성과 현실적 범위를 고려한 방향성 학습 유도
    """

    def __init__(
        self,
        # --- 법적 제한 (Legal Limits) ---
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,

        # --- 일사량 정규화 기준 (상한 제거) ---
        winter_sunlight_min: float = 50000.0,      # 최소 기준값
        winter_sunlight_excellent: float = 100000.0, # 우수 기준값 (상한 없음)
        
        # --- SVR 정규화 기준 (실제 범위 반영) ---
        svr_min: float = 0.5,           # 실제 최소값
        svr_max: float = 0.9,           # 실제 최대값
        svr_target_min: float = 0.7,    # 목표 범위 최소
        svr_target_max: float = 0.9,    # 목표 범위 최대
        svr_optimal: float = 0.8,       # 최적값

        # --- 가중치 (Weights) ---
        bcr_weight: float = 20.0,
        far_weight: float = 20.0,
        winter_sunlight_weight: float = 15.0,
        svr_weight: float = 15.0,
        improvement_weight: float = 10.0,
        
        # --- 패널티 가중치 (Penalty Weights) ---
        legality_violation_penalty: float = 30.0,
        
        # --- 평활화 파라미터 (Smoothing Parameters) ---
        reward_smoothing_factor: float = 0.3,
        
        # --- 기타 설정 ---
        zero_state_penalty: float = -10.0,
        invalid_far_penalty: float = -5.0
    ):
        # 법적 제한
        self.bcr_legal_limit = bcr_legal_limit_percent / 100.0
        self.far_legal_min_limit = far_legal_min_limit_percent / 100.0
        self.far_legal_max_limit = far_legal_max_limit_percent / 100.0

        # 일사량 정규화 기준 (상한 제거)
        self.winter_sunlight_min = winter_sunlight_min
        self.winter_sunlight_excellent = winter_sunlight_excellent
        
        # SVR 정규화 기준 (실제 범위)
        self.svr_min = svr_min
        self.svr_max = svr_max
        self.svr_target_min = svr_target_min
        self.svr_target_max = svr_target_max
        self.svr_optimal = svr_optimal
        
        # 가중치
        self.bcr_weight = bcr_weight
        self.far_weight = far_weight
        self.winter_sunlight_weight = winter_sunlight_weight
        self.svr_weight = svr_weight
        self.improvement_weight = improvement_weight
        
        self.legality_violation_penalty = legality_violation_penalty
        self.zero_state_penalty = zero_state_penalty
        self.invalid_far_penalty = invalid_far_penalty
        
        # 평활화 관련 설정
        self.reward_smoothing_factor = reward_smoothing_factor
        self.prev_raw_reward = None
        self.smoothed_reward = None
        
        # 이전 상태 추적
        self.prev_scores = None
        self.prev_state_values = None
        self.prev_total_reward = None
        
        # 초기화 시 정보 출력
        print("개선된 MaxMin 방향성 보상 함수 초기화됨 (SVR 및 일사량 범위 수정)")
        print(f"BCR 제한: {bcr_legal_limit_percent}% (최대화 목표)")
        print(f"FAR 제한: {far_legal_min_limit_percent}% ~ {far_legal_max_limit_percent}% (최대화 목표)")
        print(f"겨울 일사량: {winter_sunlight_min} 이상 (최대화 목표, 상한 없음)")
        print(f"SVR 목표 범위: {svr_target_min} ~ {svr_target_max} (최적: {svr_optimal})")

    def _calculate_bcr_score(self, bcr_value: float) -> Tuple[float, bool]:
        """
        BCR 값에 대한 점수(0-1)와 법규 위반 여부를 반환
        법적 제한 내에서 BCR이 높을수록 높은 점수 부여 (최대화 목표)
        """
        violated = False
        
        # 법적 제한 초과 시 위반
        if bcr_value > self.bcr_legal_limit:
            violated = True
            return 0.0, violated
        
        # 비정상 값 처리
        if bcr_value <= 0.01:
            return 0.1, violated
        
        # BCR 최대화 점수 계산 (개선된 곡선)
        # 법적 제한에 가까울수록 높은 점수, 하지만 너무 가파르지 않게
        normalized_bcr = bcr_value / self.bcr_legal_limit
        
        # 제곱근 함수로 완만한 증가 곡선 (경제적 효율성 고려)
        if normalized_bcr >= 0.8:  # 법적 제한의 80% 이상에서 높은 점수
            score = 0.8 + 0.2 * ((normalized_bcr - 0.8) / 0.2)  # 0.8~1.0
        elif normalized_bcr >= 0.5:  # 50~80% 구간
            score = 0.5 + 0.3 * ((normalized_bcr - 0.5) / 0.3)  # 0.5~0.8
        else:  # 50% 미만
            score = 0.2 + 0.3 * (normalized_bcr / 0.5)  # 0.2~0.5
        
        return score, violated

    def _calculate_far_score(self, far_value: float) -> Tuple[float, bool, bool]:
        """
        FAR 값에 대한 점수(0-1)와 최소/최대 법규 위반 여부를 반환
        법적 제한 내에서 FAR이 높을수록 높은 점수 부여 (최대화 목표)
        """
        min_violated = False
        max_violated = False

        # 비정상 값 처리
        if far_value < 0.01:
            return 0.1, min_violated, max_violated
            
        # 극단적으로 큰 값은 오류로 간주
        if far_value > 5600:
            return 0.0, min_violated, max_violated
        
        # 법적 제한 확인
        if far_value < self.far_legal_min_limit:
            min_violated = True
            return 0.0, min_violated, max_violated
        if far_value > self.far_legal_max_limit:
            max_violated = True
            return 0.0, min_violated, max_violated
        
        # FAR 최대화 점수 계산 (개선된 곡선)
        # 법적 범위 내에서 높을수록 좋음, 경제적 효율성 고려
        normalized_far = (far_value - self.far_legal_min_limit) / (self.far_legal_max_limit - self.far_legal_min_limit)
        
        # 상위 70% 구간에서 높은 점수 (경제적 효율성)
        if normalized_far >= 0.7:  # 상위 30% 구간
            score = 0.8 + 0.2 * ((normalized_far - 0.7) / 0.3)  # 0.8~1.0
        elif normalized_far >= 0.3:  # 중간 40% 구간
            score = 0.5 + 0.3 * ((normalized_far - 0.3) / 0.4)  # 0.5~0.8
        else:  # 하위 30% 구간
            score = 0.2 + 0.3 * (normalized_far / 0.3)  # 0.2~0.5
        
        return score, min_violated, max_violated

    def _calculate_winter_sunlight_score(self, winter_sunlight_value: float) -> float:
        """
        겨울철 일사량 값에 대한 점수(0-1.2) 반환
        일사량이 높을수록 높은 점수 부여 (최대화 목표, 상한 제거)
        """
        val = max(0, winter_sunlight_value)
        
        if val < self.winter_sunlight_min:
            # 최소값보다 작으면 비례적으로 낮은 점수
            ratio = val / self.winter_sunlight_min
            return 0.2 * ratio  # 0~0.2
        
        # 최소값 이상에서는 계속 증가
        if val < self.winter_sunlight_excellent:
            # 최소~우수 구간: 제곱근 함수로 완만한 증가
            ratio = (val - self.winter_sunlight_min) / (self.winter_sunlight_excellent - self.winter_sunlight_min)
            score = 0.2 + 0.8 * np.sqrt(ratio)  # 0.2~1.0
        else:
            # 우수 이상: 최대 점수 + 보너스 (상한 없음)
            bonus_ratio = min(0.2, (val - self.winter_sunlight_excellent) / self.winter_sunlight_excellent)
            score = 1.0 + bonus_ratio  # 1.0~1.2
        
        return score

    def _calculate_svr_score(self, svr_value: float) -> float:
        """
        SVR 값에 대한 점수(0-1) 반환
        목표 범위(0.7~0.9) 내에서 최적값(0.8)에 가까울수록 높은 점수
        """
        val = max(0, svr_value)
        
        # 전체 범위를 벗어나는 경우
        if val < self.svr_min or val > self.svr_max:
            return 0.1
        
        if self.svr_target_min <= val <= self.svr_target_max:
            # 목표 범위 내: 최적값과의 거리에 따라 점수
            distance_from_optimal = abs(val - self.svr_optimal)
            max_distance = max(
                self.svr_optimal - self.svr_target_min,
                self.svr_target_max - self.svr_optimal
            )
            if max_distance > 0:
                normalized_distance = distance_from_optimal / max_distance
                score = 1.0 - 0.3 * normalized_distance  # 0.7~1.0
            else:
                score = 1.0
        else:
            # 목표 범위 밖이지만 전체 범위 내
            if val < self.svr_target_min:
                # 목표 범위보다 낮음: SVR이 너무 낮으면 에너지 효율 저하
                distance_ratio = (self.svr_target_min - val) / (self.svr_target_min - self.svr_min)
                score = 0.7 - 0.5 * distance_ratio  # 0.2~0.7
            else:
                # 목표 범위보다 높음 (현재 설정에서는 발생하지 않음)
                distance_ratio = (val - self.svr_target_max) / (self.svr_max - self.svr_target_max)
                score = 0.7 - 0.5 * distance_ratio  # 0.2~0.7
        
        return max(0.1, score)
    
    def _calculate_improvement_bonus(self, current_scores: Dict[str, float]) -> float:
        """이전 상태 대비 개선 보너스 계산"""
        
        if self.prev_scores is None:
            return 0.0
        
        # 각 항목별 점수 변화량 합산
        delta_bcr_score = current_scores["bcr"] - self.prev_scores["bcr"]
        delta_far_score = current_scores["far"] - self.prev_scores["far"]
        delta_winter_score = current_scores["winter"] - self.prev_scores["winter"]
        delta_svr_score = current_scores["svr"] - self.prev_scores["svr"]
        
        # 각 변화량에 가중치 적용 (BCR/FAR에 더 높은 가중치)
        weighted_delta = (
            delta_bcr_score * 1.2 + 
            delta_far_score * 1.2 + 
            delta_winter_score * 0.8 + 
            delta_svr_score * 0.8
        ) / 4.0
        
        # 변화량의 크기에 따라 보너스/패널티 감소
        change_magnitude = abs(weighted_delta)
        scaled_bonus = np.sign(weighted_delta) * np.sqrt(change_magnitude)
        
        # 최종 개선 보너스 계산
        improvement_bonus = scaled_bonus * self.improvement_weight
        
        # 급격한 보너스/패널티 제한
        return np.clip(improvement_bonus, -self.improvement_weight / 2, self.improvement_weight)

    def calculate_reward(self, state: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        현재 상태 [BCR, FAR, WinterTime, SVR]를 기반으로 보상을 계산
        개선된 방향성 목표와 현실적 범위를 반영
        """
        if not isinstance(state, list) or len(state) != 4:
            raise ValueError("State must be a list of 4 floats: [BCR, FAR, WinterTime, SVR]")

        bcr, far, winter_sunlight, svr = state

        # 오류 상태 처리
        if bcr <= 0 and far <= 0:
            info = {'reward': self.zero_state_penalty, 'error': "Invalid state (BCR/FAR are zero or negative)"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'winter_sunlight_val': winter_sunlight, 'svr_val': svr,
                'bcr_score': 0, 'far_score': 0, 'winter_score': 0, 'svr_score': 0,
                'bcr_violated': True, 'far_min_violated': True, 'far_max_violated': False,
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.zero_state_penalty),
                'improvement_bonus': 0
            })
            return self.zero_state_penalty, info
        
        # 극단적으로 큰 FAR 값 처리
        if far > 5600:
            info = {'reward': self.invalid_far_penalty, 'error': f"Unrealistic FAR value: {far}"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'winter_sunlight_val': winter_sunlight, 'svr_val': svr,
                'bcr_score': 0.1, 'far_score': 0, 'winter_score': 0.5, 'svr_score': 0.5,
                'bcr_violated': False, 'far_min_violated': False, 'far_max_violated': True,
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.invalid_far_penalty),
                'improvement_bonus': 0
            })
            return self.invalid_far_penalty, info
        
        # 1. 각 요소별 점수 계산 (0-1 범위, 일사량은 1.2까지 가능)
        bcr_score, bcr_violated = self._calculate_bcr_score(bcr)
        far_score, far_min_violated, far_max_violated = self._calculate_far_score(far)
        winter_score = self._calculate_winter_sunlight_score(winter_sunlight)
        svr_score = self._calculate_svr_score(svr)

        current_scores = {
            "bcr": bcr_score,
            "far": far_score,
            "winter": winter_score,
            "svr": svr_score
        }

        # 2. 기본 보상 계산 (가중치 적용)
        weighted_bcr_reward = bcr_score * self.bcr_weight
        weighted_far_reward = far_score * self.far_weight
        weighted_winter_reward = min(winter_score, 1.0) * self.winter_sunlight_weight  # 클리핑
        weighted_svr_reward = svr_score * self.svr_weight
        
        base_reward_before_penalty = (weighted_bcr_reward + weighted_far_reward + 
                                    weighted_winter_reward + weighted_svr_reward)

        # 3. 법규 위반 패널티 적용
        legal_violation_detected = bcr_violated or far_min_violated or far_max_violated
        
        violation_details = []
        if bcr_violated:
            violation_details.append(f"BCR 초과: {bcr*100:.1f}% > {self.bcr_legal_limit*100:.1f}%")
        if far_min_violated:
            violation_details.append(f"FAR 최소 미달: {far*100:.1f}% < {self.far_legal_min_limit*100:.1f}%")
        if far_max_violated:
            violation_details.append(f"FAR 최대 초과: {far*100:.1f}% > {self.far_legal_max_limit*100:.1f}%")
        
        # 개선 인센티브 계산
        improvement_bonus = self._calculate_improvement_bonus(current_scores)
        
        if legal_violation_detected:
            # 위반 정도 계산
            violation_severity = 0.0
            
            if bcr_violated:
                bcr_excess = (bcr / self.bcr_legal_limit - 1.0) * 100
                violation_severity += min(2.0, bcr_excess / 10.0)
                
            if far_min_violated and far > 0:
                far_deficit = (1.0 - far / self.far_legal_min_limit) * 100
                violation_severity += min(2.0, far_deficit / 10.0)
                
            if far_max_violated:
                far_excess = (far / self.far_legal_max_limit - 1.0) * 100
                violation_severity += min(2.0, far_excess / 10.0)
            
            # 법규 위반 시 음수 기본값
            raw_reward = -30.0 * (1.0 + violation_severity)
            penalty_details = f"위반 심각도: {violation_severity:.2f}, 패널티: {raw_reward:.2f}"
        else:
            # 법규 준수 시 정상적인 보상 계산
            raw_reward = base_reward_before_penalty + improvement_bonus
            penalty_details = ""
        
        # 보상 스케일링
        scaled_reward = raw_reward / 15.0
        
        # 보상 평활화
        if self.prev_raw_reward is not None and self.reward_smoothing_factor > 0:
            smoothed_reward = (1 - self.reward_smoothing_factor) * scaled_reward + self.reward_smoothing_factor * self.prev_raw_reward
        else:
            smoothed_reward = scaled_reward
        
        # 클리핑
        final_reward = np.clip(smoothed_reward, -8.0, 8.0)
        
        # 상태 업데이트
        self.prev_scores = current_scores.copy()
        self.prev_state_values = state.copy()
        self.prev_raw_reward = scaled_reward
        self.smoothed_reward = smoothed_reward
        self.prev_total_reward = final_reward
        
        # 보상 계산 세부 정보
        info = {
            'bcr_val_percent': bcr * 100.0,
            'far_val_percent': far * 100.0,
            'winter_sunlight_val': winter_sunlight,
            'svr_val': svr,
            
            'bcr_score': bcr_score,
            'far_score': far_score,
            'winter_score': winter_score,
            'svr_score': svr_score,

            'weighted_bcr_reward': weighted_bcr_reward,
            'weighted_far_reward': weighted_far_reward,
            'weighted_winter_reward': weighted_winter_reward,
            'weighted_svr_reward': weighted_svr_reward,
            
            'base_reward_before_penalty': base_reward_before_penalty,
            
            'bcr_violated': bcr_violated,
            'far_min_violated': far_min_violated,
            'far_max_violated': far_max_violated,
            'legal_violation_detected': legal_violation_detected,
            'violation_details': violation_details,
            'penalty_details': penalty_details,
            
            'raw_reward_before_scaling': raw_reward,
            'scaled_reward': scaled_reward,
            'smoothed_reward': smoothed_reward,
            'final_reward': final_reward,
            'improvement_bonus': improvement_bonus
        }
        
        return final_reward, info

    def reset_prev_state(self):
        """에피소드 시작 시 이전 상태 관련 변수들을 초기화합니다."""
        self.prev_scores = None
        self.prev_state_values = None
        self.prev_raw_reward = None
        self.smoothed_reward = None
        self.prev_total_reward = None


# 테스트 코드
if __name__ == "__main__":
    # 개선된 MaxMin 방향성 보상 함수 초기화
    reward_fn = MaxMinDirectionalRewardFunction(
        bcr_legal_limit_percent=70.0,
        far_legal_min_limit_percent=200.0,
        far_legal_max_limit_percent=500.0,
        
        # 일사량 기준 (상한 제거)
        winter_sunlight_min=50000.0,
        winter_sunlight_excellent=100000.0,
        
        # SVR 실제 범위 반영
        svr_min=0.5,
        svr_max=0.9,
        svr_target_min=0.7,
        svr_target_max=0.9,
        svr_optimal=0.8,
        
        bcr_weight=20.0, 
        far_weight=20.0,
        winter_sunlight_weight=15.0, 
        svr_weight=15.0,
        improvement_weight=10.0
    )

    # 테스트 케이스 [BCR, FAR, WinterTime, SVR]
    test_cases = [
        {"name": "최대화 지향 조합", "state": [0.68, 4.9, 120000, 0.8]},    # BCR/FAR 높음, 일사량 우수, SVR 최적
        {"name": "경제적 최대화", "state": [0.69, 4.95, 110000, 0.75]},     # 법적 제한 근처
        {"name": "초고 일사량", "state": [0.65, 4.5, 150000, 0.8]},         # 일사량 보너스 테스트
        {"name": "SVR 목표 범위", "state": [0.60, 4.0, 95000, 0.85]},       # SVR 목표 범위 내
        {"name": "SVR 낮음", "state": [0.65, 4.5, 95000, 0.6]},            # SVR 목표 범위 밖
        {"name": "BCR 중간", "state": [0.50, 4.5, 95000, 0.8]},            # BCR 중간값
        {"name": "FAR 중간", "state": [0.65, 3.0, 95000, 0.8]},            # FAR 중간값
        {"name": "낮은 일사량", "state": [0.65, 4.5, 30000, 0.8]},          # 일사량 낮음
        {"name": "BCR 위반", "state": [0.72, 4.5, 95000, 0.8]},            # BCR 법적 위반
        {"name": "모든 지표 최대", "state": [0.69, 4.95, 140000, 0.8]},     # 모든 지표 최대화
    ]

    print("===== 개선된 MaxMin 방향성 보상 함수 테스트 =====")
    reward_fn.reset_prev_state()

    for case in test_cases:
        state_values = case["state"]
        print(f"\n--- {case['name']} ---")
        print(f"입력: BCR={state_values[0]*100:.1f}%, FAR={state_values[1]*100:.1f}%, Winter={state_values[2]}, SVR={state_values[3]}")

        reward, info = reward_fn.calculate_reward(state_values)
        
        print(f"최종 보상: {info['final_reward']:.3f}")
        print(f"  세부 점수: BCR={info['bcr_score']:.2f}, FAR={info['far_score']:.2f}, Winter={info['winter_score']:.2f}, SVR={info['svr_score']:.2f}")
        print(f"  가중치 적용: {info['base_reward_before_penalty']:.2f}")
        
        if info.get('legal_violation_detected', False):
            print(f"  ⚠️ 법규 위반: {', '.join(info['violation_details'])}")
        
        if info.get('improvement_bonus', 0) != 0:
            print(f"  개선 보너스: {info['improvement_bonus']:.2f}")
        
        # 방향성 확인
        bcr_direction = "↑최대화" if info['bcr_score'] > 0.7 else "→보통" if info['bcr_score'] > 0.4 else "↓낮음"
        far_direction = "↑최대화" if info['far_score'] > 0.7 else "→보통" if info['far_score'] > 0.4 else "↓낮음"
        winter_direction = "↑최대화" if info['winter_score'] > 0.8 else "→보통" if info['winter_score'] > 0.5 else "↓낮음"
        svr_direction = "✓목표범위" if 0.7 <= state_values[3] <= 0.9 else "✗범위밖"
        
        print(f"  방향성: BCR {bcr_direction}, FAR {far_direction}, 일사량 {winter_direction}, SVR {svr_direction}")