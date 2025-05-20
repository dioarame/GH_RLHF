#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 개선된 보상 함수 모듈 (안정화 버전)
입력: BCR, FAR, 여름철 일사량(SummerTime), 겨울철 일사량(WinterTime)
목표:
- BCR/FAR: 법규 내에서 경제적 가치(높은 밀도) 추구 및 적정 수준 유지
- SummerTime: 낮을수록 좋음 (냉방 부하 감소)
- WinterTime: 높을수록 좋음 (난방 부하 감소, 일사 획득)
- 안정적인 보상 함수로 학습 안정성 향상
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class EnhancedArchitectureRewardFunction:
    """안정화된 건축 설계 최적화를 위한 개선된 보상 함수 클래스"""

    def __init__(
        self,
        # --- 법적 제한 (Legal Limits) ---
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,

        # --- 목표값 (Target Values) ---
        bcr_target_percent: float = 65.0,  # 법적 제한에 약간 못 미치게 설정
        far_target_percent: float = 450.0,  # 법적 최대값에 약간 못 미치게 설정

        # --- 일사량 정규화 기준 (Normalization Caps) ---
        summer_sunlight_norm_cap: float = 200000.0,
        winter_sunlight_norm_cap: float = 200000.0,
        summer_sunlight_optimal: float = 80000.0,  # 여름 일사량 최적값
        winter_sunlight_optimal: float = 150000.0,  # 겨울 일사량 최적값

        # --- 가중치 (Weights) ---
        bcr_weight: float = 20.0,
        far_weight: float = 20.0,
        summer_sunlight_weight: float = 15.0,
        winter_sunlight_weight: float = 15.0,
        improvement_weight: float = 10.0,
        
        # --- 패널티 가중치 (Penalty Weights) ---
        legality_violation_penalty: float = 30.0,
        
        # --- 평활화 파라미터 (Smoothing Parameters) ---
        reward_smoothing_factor: float = 0.3,  # 보상 평활화 계수 (0: 평활화 없음, 1: 완전 평활화)
        
        # --- 기타 설정 ---
        zero_state_penalty: float = -10.0  # -100 -> -10으로 감소 (극단적 패널티 감소)
    ):
        # 법적 제한
        self.bcr_legal_limit = bcr_legal_limit_percent / 100.0
        self.far_legal_min_limit = far_legal_min_limit_percent / 100.0
        self.far_legal_max_limit = far_legal_max_limit_percent / 100.0

        # 목표값
        self.bcr_target = bcr_target_percent / 100.0
        self.far_target = far_target_percent / 100.0

        # 일사량 정규화 기준
        self.summer_sunlight_norm_cap = summer_sunlight_norm_cap
        self.winter_sunlight_norm_cap = winter_sunlight_norm_cap
        self.summer_sunlight_optimal = summer_sunlight_optimal
        self.winter_sunlight_optimal = winter_sunlight_optimal
        
        # 가중치
        self.bcr_weight = bcr_weight
        self.far_weight = far_weight
        self.summer_sunlight_weight = summer_sunlight_weight
        self.winter_sunlight_weight = winter_sunlight_weight
        self.improvement_weight = improvement_weight
        
        self.legality_violation_penalty = legality_violation_penalty
        self.zero_state_penalty = zero_state_penalty
        
        # 평활화 관련 설정
        self.reward_smoothing_factor = reward_smoothing_factor
        self.prev_raw_reward = None
        self.smoothed_reward = None
        
        # 이전 상태 추적
        self.prev_scores = None
        self.prev_state_values = None
        self.prev_total_reward = None

    def _calculate_bcr_score(self, bcr_value: float) -> Tuple[float, bool]:
        """
        BCR 값에 대한 점수(0-1)와 법규 위반 여부를 반환
        목표값에 가까울수록 높은 점수 부여
        """
        violated = False
        if bcr_value > self.bcr_legal_limit or bcr_value <= 0:
            violated = True
            return 0.0, violated
        
        # 목표값 기반 점수 계산 (가우시안 분포 사용)
        # 목표값에 정확히 맞으면 1.0, 멀어질수록 감소
        sigma = self.bcr_legal_limit / 3  # 표준편차를 적절히 설정
        score = np.exp(-0.5 * ((bcr_value - self.bcr_target) / sigma) ** 2)
        
        return score, violated

    def _calculate_far_score(self, far_value: float) -> Tuple[float, bool, bool]:
        """
        FAR 값에 대한 점수(0-1)와 최소/최대 법규 위반 여부를 반환
        목표값에 가까울수록 높은 점수 부여
        """
        min_violated = False
        max_violated = False

        if far_value < self.far_legal_min_limit:
            min_violated = True
            return 0.0, min_violated, max_violated
        if far_value > self.far_legal_max_limit:
            max_violated = True
            return 0.0, min_violated, max_violated
        
        # 목표값 기반 점수 계산 (가우시안 분포 사용)
        sigma = (self.far_legal_max_limit - self.far_legal_min_limit) / 4  # 표준편차를 적절히 설정
        score = np.exp(-0.5 * ((far_value - self.far_target) / sigma) ** 2)
        
        return score, min_violated, max_violated

    def _calculate_summer_sunlight_score(self, summer_sunlight_value: float) -> float:
        """
        여름철 일사량 값에 대한 점수(0-1) 반환
        최적값에 가까울수록 높은 점수, 낮을수록 좋음
        """
        # 0 이하의 값은 0으로 처리 (비정상 값 방지)
        val = max(0, summer_sunlight_value)
        
        # 최적값보다 낮으면 1.0, 높으면 감소하는 곡선
        if val <= self.summer_sunlight_optimal:
            score = 1.0
        else:
            # 최적값 초과시 점수 감소 (선형보다 완만한 감소 곡선)
            normalized_diff = (val - self.summer_sunlight_optimal) / (self.summer_sunlight_norm_cap - self.summer_sunlight_optimal)
            normalized_diff = min(1.0, normalized_diff)
            score = 1.0 - normalized_diff ** 0.8  # 제곱근으로 완만한 감소
        
        return score

    def _calculate_winter_sunlight_score(self, winter_sunlight_value: float) -> float:
        """
        겨울철 일사량 값에 대한 점수(0-1) 반환
        최적값에 가까울수록 높은 점수, 높을수록 좋음
        """
        val = max(0, winter_sunlight_value)
        
        # 최적값보다 높으면 1.0, 낮으면 감소하는 곡선
        if val >= self.winter_sunlight_optimal:
            score = 1.0
        else:
            # 최적값 미만시 점수 감소 (선형보다 완만한 감소 곡선)
            normalized_diff = (self.winter_sunlight_optimal - val) / self.winter_sunlight_optimal
            normalized_diff = min(1.0, normalized_diff)
            score = 1.0 - normalized_diff ** 0.8  # 제곱근으로 완만한 감소
        
        return score

    def _calculate_improvement_bonus(self, current_scores: Dict[str, float]) -> float:
        """이전 상태 대비 개선 보너스 계산 (안정화 버전)"""
        
        if self.prev_scores is None:
            return 0.0
        
        # 각 항목별 점수 변화량 합산 (0~1 스케일 점수 기준)
        delta_bcr_score = current_scores["bcr"] - self.prev_scores["bcr"]
        delta_far_score = current_scores["far"] - self.prev_scores["far"]
        delta_summer_score = current_scores["summer"] - self.prev_scores["summer"]
        delta_winter_score = current_scores["winter"] - self.prev_scores["winter"]
        
        # 각 변화량에 가중치 적용 (BCR/FAR에 더 높은 가중치)
        weighted_delta = (
            delta_bcr_score * 1.2 + 
            delta_far_score * 1.2 + 
            delta_summer_score * 0.8 + 
            delta_winter_score * 0.8
        ) / 4.0
        
        # 변화량의 크기에 따라 보너스/패널티 감소 (급격한 변화에 대한 보상 감소)
        change_magnitude = abs(weighted_delta)
        scaled_bonus = np.sign(weighted_delta) * np.sqrt(change_magnitude)
        
        # 최종 개선 보너스 계산
        improvement_bonus = scaled_bonus * self.improvement_weight
        
        # 급격한 보너스/패널티 제한
        return np.clip(improvement_bonus, -self.improvement_weight / 2, self.improvement_weight)

    def calculate_reward(self, state: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        현재 상태 [BCR, FAR, SummerTime, WinterTime]를 기반으로 보상을 계산.
        안정적인 보상 함수 구현 (평활화, 목표값 기반 점수, 적절한 패널티)
        """
        if not isinstance(state, list) or len(state) != 4:
            raise ValueError("State must be a list of 4 floats: [BCR, FAR, SummerTime, WinterTime]")

        bcr, far, summer_sunlight, winter_sunlight = state

        # 오류 상태 처리 (모든 값이 0 또는 비정상적인 경우)
        if bcr <= 0 and far <= 0:
            info = {'reward': self.zero_state_penalty, 'error': "Invalid state (BCR/FAR are zero or negative)"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'summer_sunlight_val': summer_sunlight, 'winter_sunlight_val': winter_sunlight,
                'bcr_score': 0, 'far_score': 0, 'summer_score': 0, 'winter_score': 0,
                'bcr_violated': True, 'far_min_violated': True, 'far_max_violated': False,
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.zero_state_penalty),
                'improvement_bonus': 0
            })
            return self.zero_state_penalty, info
        
        # 1. 각 요소별 점수 계산 (0-1 범위)
        bcr_score, bcr_violated = self._calculate_bcr_score(bcr)
        far_score, far_min_violated, far_max_violated = self._calculate_far_score(far)
        summer_score = self._calculate_summer_sunlight_score(summer_sunlight)
        winter_score = self._calculate_winter_sunlight_score(winter_sunlight)

        current_scores = {
            "bcr": bcr_score,
            "far": far_score,
            "summer": summer_score,
            "winter": winter_score
        }

        # 2. 기본 보상 계산 (가중치 적용)
        weighted_bcr_reward = bcr_score * self.bcr_weight
        weighted_far_reward = far_score * self.far_weight
        weighted_summer_reward = summer_score * self.summer_sunlight_weight
        weighted_winter_reward = winter_score * self.winter_sunlight_weight
        
        base_reward_before_penalty = (weighted_bcr_reward + weighted_far_reward + 
                                     weighted_summer_reward + weighted_winter_reward)

        # 3. 법규 위반 패널티 적용 (강화된 버전)
        legality_penalty = 0.0
        legal_violation_detected = bcr_violated or far_min_violated or far_max_violated
        
        violation_details = []
        if bcr_violated:
            violation_details.append(f"BCR 초과: {bcr*100:.1f}% > {self.bcr_legal_limit*100:.1f}%")
        if far_min_violated:
            violation_details.append(f"FAR 최소 미달: {far*100:.1f}% < {self.far_legal_min_limit*100:.1f}%")
        if far_max_violated:
            violation_details.append(f"FAR 최대 초과: {far*100:.1f}% > {self.far_legal_max_limit*100:.1f}%")
        
        # 개선 인센티브 계산 (계산 위치 수정)
        improvement_bonus = self._calculate_improvement_bonus(current_scores)
        
        if legal_violation_detected:
            # 위반 정도 계산
            violation_severity = 0.0
            
            if bcr_violated:
                bcr_excess = (bcr / self.bcr_legal_limit - 1.0) * 100  # % 초과
                violation_severity += min(2.0, bcr_excess / 10.0)
                
            if far_min_violated and far > 0:
                far_deficit = (1.0 - far / self.far_legal_min_limit) * 100  # % 부족
                violation_severity += min(2.0, far_deficit / 10.0)
                
            if far_max_violated:
                far_excess = (far / self.far_legal_max_limit - 1.0) * 100  # % 초과
                violation_severity += min(2.0, far_excess / 10.0)
            
            # 법규 위반 시 음수 기본값에서 시작
            raw_reward = -30.0 * (1.0 + violation_severity)
            penalty_details = f"위반 심각도: {violation_severity:.2f}, 패널티: {raw_reward:.2f}"
        else:
            # 법규 준수 시 정상적인 보상 계산
            current_design_score = base_reward_before_penalty  # 패널티 적용 안함
            raw_reward = current_design_score + improvement_bonus
            penalty_details = ""
        
        # 보상 스케일링 (강화된 버전 - 15.0으로 나눔)
        scaled_reward = raw_reward / 15.0
        
        # 보상 평활화 (기존과 동일한 로직 유지)
        if self.prev_raw_reward is not None and self.reward_smoothing_factor > 0:
            smoothed_reward = (1 - self.reward_smoothing_factor) * scaled_reward + self.reward_smoothing_factor * self.prev_raw_reward
        else:
            smoothed_reward = scaled_reward
        
        # 클리핑 범위 수정 (-8.0 ~ 8.0), 기존에는 -50.0 ~ 50.0 사용
        final_reward = np.clip(smoothed_reward, -8.0, 8.0)
        
        # 상태 업데이트
        self.prev_scores = current_scores.copy()
        self.prev_state_values = state.copy()
        self.prev_raw_reward = scaled_reward  # 스케일링된 값 저장
        self.smoothed_reward = smoothed_reward
        self.prev_total_reward = final_reward
        
        # 보상 계산 세부 정보에 새로운 정보 추가
        info = {
            'bcr_val_percent': bcr * 100.0,
            'far_val_percent': far * 100.0,
            'summer_sunlight_val': summer_sunlight,
            'winter_sunlight_val': winter_sunlight,
            
            'bcr_score': bcr_score,
            'far_score': far_score,
            'summer_score': summer_score,
            'winter_score': winter_score,

            'weighted_bcr_reward': weighted_bcr_reward,
            'weighted_far_reward': weighted_far_reward,
            'weighted_summer_reward': weighted_summer_reward,
            'weighted_winter_reward': weighted_winter_reward,
            
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
            'final_reward': final_reward
        }
        
        return final_reward, info

    def reset_prev_state(self):
        """에피소드 시작 시 이전 상태 관련 변수들을 초기화합니다."""
        self.prev_scores = None
        self.prev_state_values = None
        self.prev_raw_reward = None
        self.smoothed_reward = None
        self.prev_total_reward = None


# === 테스트 코드 예시 ===
if __name__ == "__main__":
    # 보상 함수 초기화 (목표값 기반 설정)
    reward_fn = EnhancedArchitectureRewardFunction(
        bcr_legal_limit_percent=70.0,
        far_legal_min_limit_percent=200.0,
        far_legal_max_limit_percent=500.0,
        bcr_target_percent=65.0,  # 목표 BCR (법적 한도의 약 93%)
        far_target_percent=450.0,  # 목표 FAR (법적 최대의 약 90%)
        summer_sunlight_norm_cap=200000.0,
        winter_sunlight_norm_cap=200000.0,
        summer_sunlight_optimal=80000.0,  # 여름 일사량 최적값
        winter_sunlight_optimal=150000.0,  # 겨울 일사량 최적값
        bcr_weight=20.0, 
        far_weight=20.0,
        summer_sunlight_weight=15.0, 
        winter_sunlight_weight=15.0,
        improvement_weight=10.0,
        legality_violation_penalty=50.0,
        reward_smoothing_factor=0.3,  # 30% 평활화
        zero_state_penalty=-10.0  # -100 -> -10으로 감소
    )

    # 테스트 상태 [BCR(소수), FAR(소수), SummerTime, WinterTime]
    test_cases = [
        {"name": "이상적인 상태 (첫 스텝)", "state": [0.65, 4.5, 80000, 150000]},  # 목표값에 맞춘 이상적 상태
        {"name": "개선된 상태", "state": [0.67, 4.6, 75000, 160000]},  # 약간 개선된 상태
        {"name": "BCR 약간 위반", "state": [0.72, 4.5, 85000, 150000]},  # BCR 약간 위반
        {"name": "BCR 크게 위반", "state": [0.85, 4.5, 85000, 150000]},  # BCR 크게 위반
        {"name": "FAR 최소 약간 미달", "state": [0.65, 1.9, 80000, 150000]},  # FAR 약간 미달
        {"name": "FAR 최소 크게 미달", "state": [0.65, 1.0, 80000, 150000]},  # FAR 크게 미달
        {"name": "FAR 최대 약간 초과", "state": [0.65, 5.1, 80000, 150000]},  # FAR 약간 초과
        {"name": "FAR 최대 크게 초과", "state": [0.65, 6.0, 80000, 150000]},  # FAR 크게 초과
        {"name": "여름 일사량 약간 나쁨", "state": [0.65, 4.5, 120000, 150000]},  # 여름 일사량 약간 나쁨
        {"name": "여름 일사량 매우 나쁨", "state": [0.65, 4.5, 180000, 150000]},  # 여름 일사량 매우 나쁨
        {"name": "겨울 일사량 약간 나쁨", "state": [0.65, 4.5, 80000, 100000]},  # 겨울 일사량 약간 나쁨
        {"name": "겨울 일사량 매우 나쁨", "state": [0.65, 4.5, 80000, 50000]},  # 겨울 일사량 매우 나쁨
        {"name": "약간 나빠진 상태", "state": [0.63, 4.4, 90000, 140000]},  # 약간 나빠진 상태
        {"name": "매우 나빠진 상태", "state": [0.55, 3.8, 110000, 110000]},  # 크게 나빠진 상태
        {"name": "오류 상태", "state": [0.0, 0.0, 0, 0]},  # 오류 상태
    ]

    print("===== 개선된 보상 함수 테스트 =====")
    reward_fn.reset_prev_state()  # 에피소드 시작 시 호출 가정

    for i, case in enumerate(test_cases):
        state_values = case["state"]
        print(f"\n--- 테스트: {case['name']} ---")
        print(f"입력 상태: BCR={state_values[0]*100:.1f}%, FAR={state_values[1]*100:.1f}%, Summer={state_values[2]}, Winter={state_values[3]}")

        if reward_fn.prev_state_values:  # 이전 상태 값 출력 (디버깅용)
             print(f"이전 상태: BCR={reward_fn.prev_state_values[0]*100:.1f}%, FAR={reward_fn.prev_state_values[1]*100:.1f}%, Summer={reward_fn.prev_state_values[2]}, Winter={reward_fn.prev_state_values[3]}")
        
        reward, info = reward_fn.calculate_reward(state_values)
        
        print(f"최종 보상: {info['final_reward']:.2f}")
        print(f"  세부 점수: BCR={info['bcr_score']:.2f}, FAR={info['far_score']:.2f}, Summer={info['summer_score']:.2f}, Winter={info['winter_score']:.2f}")
        print(f"  가중치 적용 보상 (패널티 전): {info['base_reward_before_penalty']:.2f}")
        
        if info['legality_penalty'] > 0:
            print(f"  법규 위반 패널티: -{info['legality_penalty']:.2f}")
            if info['bcr_violated']: print("    * BCR 위반")
            if info['far_min_violated']: print("    * FAR 최소 미달")
            if info['far_max_violated']: print("    * FAR 최대 초과")
        
        if 'error' in info:
             print(f"  오류: {info['error']}")
        
        if 'improvement_bonus' in info:
            print(f"  개선 인센티브: {info['improvement_bonus']:.2f}")
        
        if 'raw_reward' in info and 'smoothed_reward' in info:
            print(f"  원시 보상: {info['raw_reward']:.2f}, 평활화된 보상: {info['smoothed_reward']:.2f}")