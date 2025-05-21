#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 최대/최소 방향성 보상 함수 모듈 (개선된 버전)
입력: BCR, FAR, 겨울철 일사량(WinterTime), 표면적 체적비(SV_Ratio)
목표:
- BCR: 법적 제한 내에서 최대화 (토지 이용 효율 최대화)
- FAR: 법적 제한 내에서 최대화 (건축 밀도 최대화)
- WinterTime: 최대화 (겨울철 일사량 확보로 난방 부하 감소)
- SV_Ratio: 최소화 (표면적 체적비 최소화로 에너지 효율 증가)

이 보상 함수는 기존의 '최적 범위' 대신 각 항목별 최대화/최소화 방향에 높은 점수를 부여합니다.
이를 통해 더 넓은 디자인 공간 탐색이 가능해집니다.

개선된 버전에서는 OptimizedArchitectureRewardFunction 모듈과 일관된 처리 방식 적용:
- 보상 스케일링 (15.0으로 나눔)
- 보상 클리핑 (-8.0 ~ 8.0)
- 법규 위반시 심각도 기반 패널티
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class MaxMinDirectionalRewardFunction:
    """
    각 항목별 최대화/최소화 방향을 장려하는 보상 함수 클래스
    기존 최적 범위 보상 함수와의 비교를 위한 대조군으로 사용
    """

    def __init__(
        self,
        # --- 법적 제한 (Legal Limits) ---
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,

        # --- 일사량 정규화 기준 ---
        winter_sunlight_min: float = 5000.0,    # 최소 기준값
        winter_sunlight_max: float = 150000.0,  # 최대 기준값
        
        # --- 표면적 체적비 정규화 기준 ---
        sv_ratio_min: float = 0.5,      # 최소 기준값
        sv_ratio_max: float = 5.0,      # 최대 기준값

        # --- 가중치 (Weights) ---
        bcr_weight: float = 20.0,
        far_weight: float = 20.0,
        winter_sunlight_weight: float = 15.0,
        sv_ratio_weight: float = 15.0,
        improvement_weight: float = 10.0,
        
        # --- 패널티 가중치 (Penalty Weights) ---
        legality_violation_penalty: float = 30.0,
        
        # --- 평활화 파라미터 (Smoothing Parameters) ---
        reward_smoothing_factor: float = 0.3,  # 보상 평활화 계수 (0: 평활화 없음, 1: 완전 평활화)
        
        # --- 기타 설정 ---
        zero_state_penalty: float = -10.0,   # 비정상 상태에 대한 패널티
        invalid_far_penalty: float = -5.0    # 비정상적인 FAR 값에 대한 패널티
    ):
        # 법적 제한
        self.bcr_legal_limit = bcr_legal_limit_percent / 100.0
        self.far_legal_min_limit = far_legal_min_limit_percent / 100.0
        self.far_legal_max_limit = far_legal_max_limit_percent / 100.0

        # 정규화 기준
        self.winter_sunlight_min = winter_sunlight_min
        self.winter_sunlight_max = winter_sunlight_max
        
        self.sv_ratio_min = sv_ratio_min
        self.sv_ratio_max = sv_ratio_max
        
        # 가중치
        self.bcr_weight = bcr_weight
        self.far_weight = far_weight
        self.winter_sunlight_weight = winter_sunlight_weight
        self.sv_ratio_weight = sv_ratio_weight
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
        print("최대/최소 방향성 보상 함수 초기화됨 (개선된 버전)")
        print(f"BCR 제한: {bcr_legal_limit_percent}% (최대화 목표)")
        print(f"FAR 제한: {far_legal_min_limit_percent}% ~ {far_legal_max_limit_percent}% (최대화 목표)")
        print(f"겨울 일사량 정규화: {winter_sunlight_min} ~ {winter_sunlight_max} (최대화 목표)")
        print(f"표면적 체적비 정규화: {sv_ratio_min} ~ {sv_ratio_max} (최소화 목표)")
        print(f"보상 스케일링: 15.0으로 나눔, 클리핑: -8.0 ~ 8.0")

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
        if bcr_value <= 0.01:  # 거의 0에 가까운 BCR은 형태가 제대로 생성되지 않은 것으로 간주
            return 0.1, violated  # 낮은 점수 부여
        
        # BCR 최대화 점수 계산 (선형)
        # BCR이 법적 제한에 가까울수록 1에 가까운 점수
        score = bcr_value / self.bcr_legal_limit
        
        return score, violated

    def _calculate_far_score(self, far_value: float) -> Tuple[float, bool, bool]:
        """
        FAR 값에 대한 점수(0-1)와 최소/최대 법규 위반 여부를 반환
        법적 제한 내에서 FAR이 높을수록 높은 점수 부여 (최대화 목표)
        """
        min_violated = False
        max_violated = False

        # 비정상 값 처리
        if far_value < 0.01:  # FAR이 0에 가까우면 형태가 제대로 생성되지 않은 것
            return 0.1, min_violated, max_violated  # 낮은 점수 부여
            
        # 극단적으로 큰 값은 오류로 간주
        if far_value > 5600:  # 분석된 최대값 이상은 이상치로 간주
            return 0.0, min_violated, max_violated  # 매우 낮은 점수 부여
        
        # 법적 제한 확인
        if far_value < self.far_legal_min_limit:
            min_violated = True
            return 0.0, min_violated, max_violated
        if far_value > self.far_legal_max_limit:
            max_violated = True
            return 0.0, min_violated, max_violated
        
        # FAR 최대화 점수 계산 (선형)
        # 법적 범위 내에서 FAR이 최대값에 가까울수록 높은 점수
        normalized_far = (far_value - self.far_legal_min_limit) / (self.far_legal_max_limit - self.far_legal_min_limit)
        score = normalized_far
        
        return score, min_violated, max_violated

    def _calculate_winter_sunlight_score(self, winter_sunlight_value: float) -> float:
        """
        겨울철 일사량 값에 대한 점수(0-1) 반환
        일사량이 높을수록 높은 점수 부여 (최대화 목표)
        """
        # 0 이하의 값은 0으로 처리
        val = max(0, winter_sunlight_value)
        
        # 정규화된 범위를 벗어나는 값 처리
        if val < self.winter_sunlight_min:
            # 최소값보다 작으면 낮은 점수
            return 0.2
        
        if val > self.winter_sunlight_max:
            # 최대값보다 크면 최대 점수 (약간의 버퍼 제공)
            return 1.0
        
        # 겨울 일사량 최대화 점수 계산 (선형)
        # 최소값에서 최대값 사이에서 선형으로 증가
        normalized_val = (val - self.winter_sunlight_min) / (self.winter_sunlight_max - self.winter_sunlight_min)
        score = normalized_val
        
        return score

    def _calculate_sv_ratio_score(self, sv_ratio_value: float) -> float:
        """
        표면적 체적비 값에 대한 점수(0-1) 반환
        표면적 체적비가 낮을수록 높은 점수 부여 (최소화 목표 - 에너지 효율)
        """
        # 0 이하의 값은 0으로 처리 (비정상 값 방지)
        val = max(0, sv_ratio_value)
        
        # 정규화된 범위를 벗어나는 값 처리
        if val < self.sv_ratio_min:
            # 최소값보다 작으면 최대 점수 (약간의 버퍼 제공)
            return 1.0
        
        if val > self.sv_ratio_max:
            # 최대값보다 크면 낮은 점수
            return 0.2
        
        # 표면적 체적비 최소화 점수 계산 (선형 역방향)
        # 최소값에 가까울수록 높은 점수 (최소화 목표)
        normalized_val = (val - self.sv_ratio_min) / (self.sv_ratio_max - self.sv_ratio_min)
        score = 1.0 - normalized_val
        
        return score
    
    def _calculate_improvement_bonus(self, current_scores: Dict[str, float]) -> float:
        """이전 상태 대비 개선 보너스 계산 (안정화 버전)"""
        
        if self.prev_scores is None:
            return 0.0
        
        # 각 항목별 점수 변화량 합산 (0~1 스케일 점수 기준)
        delta_bcr_score = current_scores["bcr"] - self.prev_scores["bcr"]
        delta_far_score = current_scores["far"] - self.prev_scores["far"]
        delta_winter_score = current_scores["winter"] - self.prev_scores["winter"]
        delta_sv_ratio_score = current_scores["sv_ratio"] - self.prev_scores["sv_ratio"]
        
        # 각 변화량에 가중치 적용 (BCR/FAR에 더 높은 가중치)
        weighted_delta = (
            delta_bcr_score * 1.2 + 
            delta_far_score * 1.2 + 
            delta_winter_score * 0.8 + 
            delta_sv_ratio_score * 0.8
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
        현재 상태 [BCR, FAR, WinterTime, SV_Ratio]를 기반으로 보상을 계산.
        각 항목별 최대/최소 방향성에 따라 점수 부여
        """
        if not isinstance(state, list) or len(state) != 4:
            raise ValueError("State must be a list of 4 floats: [BCR, FAR, WinterTime, SV_Ratio]")

        bcr, far, winter_sunlight, sv_ratio = state

        # 오류 상태 처리 (모든 값이 0 또는 비정상적인 경우)
        if bcr <= 0 and far <= 0:
            info = {'reward': self.zero_state_penalty, 'error': "Invalid state (BCR/FAR are zero or negative)"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'winter_sunlight_val': winter_sunlight, 'sv_ratio_val': sv_ratio,
                'bcr_score': 0, 'far_score': 0, 'winter_score': 0, 'sv_ratio_score': 0,
                'bcr_violated': True, 'far_min_violated': True, 'far_max_violated': False,
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.zero_state_penalty),
                'improvement_bonus': 0
            })
            return self.zero_state_penalty, info
        
        # 극단적으로 큰 FAR 값 처리 (오류 또는 이상치)
        if far > 5600:
            info = {'reward': self.invalid_far_penalty, 'error': f"Unrealistic FAR value: {far}"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'winter_sunlight_val': winter_sunlight, 'sv_ratio_val': sv_ratio,
                'bcr_score': 0.1, 'far_score': 0, 'winter_score': 0.5, 'sv_ratio_score': 0.5,
                'bcr_violated': False, 'far_min_violated': False, 'far_max_violated': True,
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.invalid_far_penalty),
                'improvement_bonus': 0
            })
            return self.invalid_far_penalty, info
        
        # 1. 각 요소별 점수 계산 (0-1 범위)
        bcr_score, bcr_violated = self._calculate_bcr_score(bcr)
        far_score, far_min_violated, far_max_violated = self._calculate_far_score(far)
        winter_score = self._calculate_winter_sunlight_score(winter_sunlight)
        sv_ratio_score = self._calculate_sv_ratio_score(sv_ratio)

        current_scores = {
            "bcr": bcr_score,
            "far": far_score,
            "winter": winter_score,
            "sv_ratio": sv_ratio_score
        }

        # 2. 기본 보상 계산 (가중치 적용)
        weighted_bcr_reward = bcr_score * self.bcr_weight
        weighted_far_reward = far_score * self.far_weight
        weighted_winter_reward = winter_score * self.winter_sunlight_weight
        weighted_sv_ratio_reward = sv_ratio_score * self.sv_ratio_weight
        
        base_reward_before_penalty = (weighted_bcr_reward + weighted_far_reward + 
                                    weighted_winter_reward + weighted_sv_ratio_reward)

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
        
        # 클리핑 범위 수정 (-8.0 ~ 8.0)
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
            'winter_sunlight_val': winter_sunlight,
            'sv_ratio_val': sv_ratio,
            
            'bcr_score': bcr_score,
            'far_score': far_score,
            'winter_score': winter_score,
            'sv_ratio_score': sv_ratio_score,

            'weighted_bcr_reward': weighted_bcr_reward,
            'weighted_far_reward': weighted_far_reward,
            'weighted_winter_reward': weighted_winter_reward,
            'weighted_sv_ratio_reward': weighted_sv_ratio_reward,
            
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


# === 테스트 코드 예시 ===
if __name__ == "__main__":
    # 보상 함수 초기화 (직접 최대/최소 방향 설정)
    reward_fn = MaxMinDirectionalRewardFunction(
        bcr_legal_limit_percent=70.0,
        far_legal_min_limit_percent=200.0,
        far_legal_max_limit_percent=500.0,
        
        # 정규화 기준
        winter_sunlight_min=5000.0,
        winter_sunlight_max=150000.0,
        sv_ratio_min=0.5,
        sv_ratio_max=5.0,
        
        # 가중치
        bcr_weight=20.0, 
        far_weight=20.0,
        winter_sunlight_weight=15.0, 
        sv_ratio_weight=15.0,
        improvement_weight=10.0,
        
        # 기타 설정
        legality_violation_penalty=30.0,
        reward_smoothing_factor=0.3,
        zero_state_penalty=-10.0
    )

    # 테스트 상태 [BCR(소수), FAR(소수), WinterTime, SV_Ratio]
    test_cases = [
        {"name": "법적 제한에 가까운 BCR, 높은 FAR", "state": [0.68, 4.9, 120000, 0.6]},  # 법적 제한에 가까운 높은 BCR, 높은 FAR
        {"name": "중간 범위 BCR, 높은 FAR", "state": [0.5, 4.9, 120000, 0.6]},  # 중간 범위 BCR
        {"name": "낮은 BCR, 높은 FAR", "state": [0.2, 4.9, 120000, 0.6]},  # 낮은 BCR
        {"name": "높은 BCR, 중간 FAR", "state": [0.68, 3.5, 120000, 0.6]},  # 중간 범위 FAR
        {"name": "높은 BCR, 낮은 FAR", "state": [0.68, 2.1, 120000, 0.6]},  # 낮은 FAR (법적 최소에 가까움)
        {"name": "BCR 약간 위반", "state": [0.72, 4.5, 120000, 0.6]},  # BCR 약간 위반
        {"name": "BCR 크게 위반", "state": [0.85, 4.5, 120000, 0.6]},  # BCR 크게 위반
        {"name": "FAR 최소 위반", "state": [0.68, 1.9, 120000, 0.6]},  # FAR 최소 위반
        {"name": "FAR 최대 위반", "state": [0.68, 5.1, 120000, 0.6]},  # FAR 최대 위반
        {"name": "낮은 겨울 일사량", "state": [0.68, 4.5, 30000, 0.6]},  # 낮은 겨울 일사량
        {"name": "중간 겨울 일사량", "state": [0.68, 4.5, 80000, 0.6]},  # 중간 겨울 일사량
        {"name": "높은 겨울 일사량", "state": [0.68, 4.5, 140000, 0.6]},  # 높은 겨울 일사량
        {"name": "최소 표면적 체적비", "state": [0.68, 4.5, 120000, 0.5]},  # 최소 표면적 체적비
        {"name": "중간 표면적 체적비", "state": [0.68, 4.5, 120000, 2.0]},  # 중간 표면적 체적비
        {"name": "높은 표면적 체적비", "state": [0.68, 4.5, 120000, 4.0]},  # 높은 표면적 체적비
        {"name": "최적 조합", "state": [0.68, 4.9, 140000, 0.5]},  # 모든 지표에서 최적 방향 (BCR 높음, FAR 높음, 일사량 높음, SV 낮음)
        {"name": "최악 조합", "state": [0.2, 2.1, 10000, 4.0]},  # 모든 지표에서 비최적 방향
        {"name": "오류 상태", "state": [0.0, 0.0, 0, 0]},  # 오류 상태
    ]

    print("===== 최대/최소 방향성 보상 함수 테스트 (개선된 버전) =====")
    reward_fn.reset_prev_state()  # 에피소드 시작 시 호출 가정

    for i, case in enumerate(test_cases):
        state_values = case["state"]
        print(f"\n--- 테스트: {case['name']} ---")
        print(f"입력 상태: BCR={state_values[0]*100:.1f}%, FAR={state_values[1]*100:.1f}%, Winter={state_values[2]}, SV_Ratio={state_values[3]}")

        reward, info = reward_fn.calculate_reward(state_values)
        
        print(f"최종 보상: {info['final_reward']:.2f}")
        print(f"  세부 점수: BCR={info['bcr_score']:.2f}, FAR={info['far_score']:.2f}, Winter={info['winter_score']:.2f}, SV_Ratio={info['sv_ratio_score']:.2f}")
        print(f"  가중치 적용 보상 (패널티 전): {info['base_reward_before_penalty']:.2f}")
        
        if info.get('legal_violation_detected', False):
            print(f"  법규 위반 감지: {', '.join(info['violation_details'])}")
            print(f"  패널티 상세: {info['penalty_details']}")
        
        if 'improvement_bonus' in info and info['improvement_bonus'] != 0:
            print(f"  개선 인센티브: {info['improvement_bonus']:.2f}")
        
        if 'scaled_reward' in info:
            print(f"  원시 보상: {info['raw_reward_before_scaling']:.2f}, 스케일링: {info['scaled_reward']:.2f}, 평활화: {info['smoothed_reward']:.2f}")