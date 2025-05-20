#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 개선된 보상 함수 모듈 (안정화 및 최적화 버전)
입력: BCR, FAR, 여름철 일사량(SummerTime), 겨울철 일사량(WinterTime)
목표:
- BCR/FAR: 법규 내에서 경제적 가치(높은 밀도) 추구 및 적정 수준 유지
- SummerTime: 낮을수록 좋음 (냉방 부하 감소)
- WinterTime: 높을수록 좋음 (난방 부하 감소, 일사 획득)
- 안정적인 보상 함수로 학습 안정성 향상
- 분석된 최적 구간을 고려한 보상 함수 수정
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class OptimizedArchitectureRewardFunction:
    """안정화된 건축 설계 최적화를 위한 개선된 보상 함수 클래스"""

    def __init__(
        self,
        # --- 법적 제한 (Legal Limits) ---
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,

        # --- 목표값 (Target Values) - 분석 데이터 기반 조정 ---
        bcr_target_min_percent: float = 30.2,   # 분석된 최적 BCR 최소값 
        bcr_target_max_percent: float = 60.4,   # 분석된 최적 BCR 최대값
        far_target_min_percent: float = 200.0,  # 최소 FAR은 법적 제한으로 설정
        far_target_max_percent: float = 450.0,  # 최적 구간 내 적절한 값

        # --- 표면적 체적비 정규화 기준 (Surface-to-Volume Ratio) ---
        sv_ratio_min: float = 0.5,         # 표면적 체적비 최소값
        sv_ratio_max: float = 3.0,         # 표면적 체적비 최대값
        sv_ratio_optimal: float = 0.8,     # 표면적 체적비 최적값 (낮을수록 좋음)
        
        # --- 일사량 정규화 기준 (기존 겨울 일사량만 사용) ---
        winter_sunlight_min: float = 73000.0,   # 분석된 겨울 일사량 최소값에 근접 
        winter_sunlight_max: float = 81000.0,   # 분석된 겨울 일사량 최대값에 근접
        winter_sunlight_optimal: float = 80000.0,  # 겨울 일사량 최적값 (더 높은 쪽으로 설정)

        # --- 가중치 (Weights) ---
        bcr_weight: float = 20.0,
        far_weight: float = 20.0,
        sv_ratio_weight: float = 15.0,          # 표면적 체적비 가중치
        winter_sunlight_weight: float = 15.0,
        improvement_weight: float = 10.0,
        
        # --- 패널티 가중치 (Penalty Weights) ---
        legality_violation_penalty: float = 30.0,
        
        # --- 평활화 파라미터 (Smoothing Parameters) ---
        reward_smoothing_factor: float = 0.3,  # 보상 평활화 계수 (0: 평활화 없음, 1: 완전 평활화)
        
        # --- 기타 설정 ---
        zero_state_penalty: float = -10.0,  # 비정상 상태에 대한 패널티
        invalid_far_penalty: float = -5.0   # 비정상적인 FAR 값에 대한 패널티
    ):
        # 법적 제한
        self.bcr_legal_limit = bcr_legal_limit_percent / 100.0
        self.far_legal_min_limit = far_legal_min_limit_percent / 100.0
        self.far_legal_max_limit = far_legal_max_limit_percent / 100.0

        # 목표값 (최적구간)
        self.bcr_target_min = bcr_target_min_percent / 100.0
        self.bcr_target_max = bcr_target_max_percent / 100.0
        self.far_target_min = far_target_min_percent / 100.0
        self.far_target_max = far_target_max_percent / 100.0

        # 표면적 체적비 정규화 기준
        self.sv_ratio_min = sv_ratio_min
        self.sv_ratio_max = sv_ratio_max
        self.sv_ratio_optimal = sv_ratio_optimal
        
        # 겨울 일사량 정규화 기준
        self.winter_sunlight_min = winter_sunlight_min
        self.winter_sunlight_max = winter_sunlight_max
        self.winter_sunlight_optimal = winter_sunlight_optimal
        
        # 가중치
        self.bcr_weight = bcr_weight
        self.far_weight = far_weight
        self.sv_ratio_weight = sv_ratio_weight
        self.winter_sunlight_weight = winter_sunlight_weight
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

    def _calculate_bcr_score(self, bcr_value: float) -> Tuple[float, bool]:
        """
        BCR 값에 대한 점수(0-1)와 법규 위반 여부를 반환
        최적 구간에 가까울수록 높은 점수 부여
        """
        violated = False
        if bcr_value > self.bcr_legal_limit:
            violated = True
            return 0.0, violated
        
        # 비정상 값 처리
        if bcr_value <= 0.01:  # 거의 0에 가까운 BCR은 형태가 제대로 생성되지 않은 것으로 간주
            return 0.1, violated  # 완전히 0점은 아니지만 낮은 점수 부여
        
        # 목표 구간 기반 점수 계산
        # 목표 구간 내에 있으면 높은 점수, 벗어나면 거리에 따라 감소
        if self.bcr_target_min <= bcr_value <= self.bcr_target_max:
            # 최적 구간 내에서는 높은 점수
            # 구간 내에서도 중간 값에 가까울수록 더 높은 점수
            mid_point = (self.bcr_target_min + self.bcr_target_max) / 2
            distance_from_mid = abs(bcr_value - mid_point)
            max_distance = (self.bcr_target_max - self.bcr_target_min) / 2
            
            # 정규화된 거리 (0에 가까울수록 중간에 가까움)
            normalized_distance = distance_from_mid / max_distance
            
            # 중간에 가까울수록 1에 가까운 점수 (0.8~1.0 범위)
            score = 1.0 - 0.2 * normalized_distance
        else:
            # 최적 구간을 벗어나면 거리에 따라 감소하는 점수
            if bcr_value < self.bcr_target_min:
                # 너무 낮은 BCR
                distance = self.bcr_target_min - bcr_value
                max_distance = self.bcr_target_min  # 최대 거리는 0까지
                normalized_distance = min(1.0, distance / max_distance)
                score = 0.8 * (1.0 - normalized_distance)  # 0.8에서 시작해서 감소
            else:
                # 너무 높은 BCR (법적 제한 이내)
                distance = bcr_value - self.bcr_target_max
                max_distance = self.bcr_legal_limit - self.bcr_target_max  # 최대 거리는 법적 제한까지
                normalized_distance = min(1.0, distance / max_distance)
                score = 0.8 * (1.0 - normalized_distance)  # 0.8에서 시작해서 감소
        
        return score, violated

    def _calculate_far_score(self, far_value: float) -> Tuple[float, bool, bool]:
        """
        FAR 값에 대한 점수(0-1)와 최소/최대 법규 위반 여부를 반환
        최적 구간에 가까울수록 높은 점수 부여
        """
        min_violated = False
        max_violated = False

        # 비정상 값 처리
        if far_value < 0.01:  # FAR이 0에 가까우면 형태가 제대로 생성되지 않은 것
            return 0.1, min_violated, max_violated  # 낮은 점수 부여하고 위반은 아님
            
        # 극단적으로 큰 값은 오류로 간주 (분석 데이터 기준)
        if far_value > 5600:  # 분석된 최대값 이상은 이상치로 간주
            return 0.0, min_violated, max_violated  # 매우 낮은 점수 부여
        
        # 법적 제한 확인
        if far_value < self.far_legal_min_limit:
            min_violated = True
            return 0.0, min_violated, max_violated
        if far_value > self.far_legal_max_limit:
            max_violated = True
            return 0.0, min_violated, max_violated
        
        # 목표 구간 내에 있는지 확인
        if self.far_target_min <= far_value <= self.far_target_max:
            # 목표 구간 내에서는 높은 점수
            # 구간 내에서도 적정 값에 가까울수록 더 높은 점수
            mid_point = (self.far_target_min + self.far_target_max) / 2
            distance_from_mid = abs(far_value - mid_point)
            max_distance = (self.far_target_max - self.far_target_min) / 2
            
            # 정규화된 거리 (0에 가까울수록 중간에 가까움)
            normalized_distance = distance_from_mid / max_distance
            
            # 중간에 가까울수록 1에 가까운 점수 (0.8~1.0 범위)
            score = 1.0 - 0.2 * normalized_distance
        else:
            # 목표 구간을 벗어나면 거리에 따라 감소하는 점수
            if far_value < self.far_target_min:
                # 너무 낮은 FAR (법적 최소 이상)
                distance = self.far_target_min - far_value
                max_distance = self.far_target_min - self.far_legal_min_limit
                normalized_distance = min(1.0, distance / max_distance if max_distance > 0 else 1.0)
                score = 0.8 * (1.0 - normalized_distance)  # 0.8에서 시작해서 감소
            else:
                # 너무 높은 FAR (법적 최대 이하)
                distance = far_value - self.far_target_max
                max_distance = self.far_legal_max_limit - self.far_target_max
                normalized_distance = min(1.0, distance / max_distance if max_distance > 0 else 1.0)
                score = 0.8 * (1.0 - normalized_distance)  # 0.8에서 시작해서 감소
                
        return score, min_violated, max_violated

    def _calculate_summer_sunlight_score(self, summer_sunlight_value: float) -> float:
        """
        여름철 일사량 값에 대한 점수(0-1) 반환
        최적값에 가까울수록 높은 점수, 낮을수록 좋음
        분석된 최적 구간 (74,000~81,000)을 고려
        """
        # 0 이하의 값은 0으로 처리 (비정상 값 방지)
        val = max(0, summer_sunlight_value)
        
        # 너무 작거나 큰 값 처리 (정상 범위를 벗어남)
        if val < self.summer_sunlight_min / 2:  # 이상치 처리
            return 0.5  # 중간 점수 (너무 낮아도 좋지 않음)
        if val > self.summer_sunlight_max * 2:  # 이상치 처리
            return 0.2  # 낮은 점수 (너무 높으면 좋지 않음)
        
        # 최적값 기준 점수 계산 (여름철은 낮을수록 좋음)
        if val <= self.summer_sunlight_optimal:
            # 최적값 이하는 높은 점수
            # 단, 너무 낮아도 비현실적이므로 일정 수준 이하면 점수 감소
            if val < self.summer_sunlight_min:
                # 최소값보다 작으면 거리에 따라 감소
                normalized_diff = (self.summer_sunlight_min - val) / self.summer_sunlight_min
                return 0.8 - 0.3 * normalized_diff  # 0.8에서 시작해 감소
            else:
                # 최적 범위 내 (최소값~최적값)
                normalized_pos = (val - self.summer_sunlight_min) / (self.summer_sunlight_optimal - self.summer_sunlight_min)
                return 0.8 + 0.2 * normalized_pos  # 0.8에서 1.0까지
        else:
            # 최적값 초과시 점수 감소
            if val > self.summer_sunlight_max:
                # 최대값 초과 시 더 빠르게 감소
                normalized_diff = (val - self.summer_sunlight_max) / self.summer_sunlight_max
                return 0.6 - 0.4 * min(1.0, normalized_diff)  # 0.6에서 0.2까지 감소
            else:
                # 최적값~최대값 사이
                normalized_diff = (val - self.summer_sunlight_optimal) / (self.summer_sunlight_max - self.summer_sunlight_optimal)
                return 1.0 - 0.4 * normalized_diff  # 1.0에서 0.6까지 감소
        
    def _calculate_winter_sunlight_score(self, winter_sunlight_value: float) -> float:
        """
        겨울철 일사량 값에 대한 점수(0-1) 반환
        최적값에 가까울수록 높은 점수, 높을수록 좋음
        분석된 최적 구간 (73,000~80,000)을 고려
        """
        val = max(0, winter_sunlight_value)
        
        # 너무 작거나 큰 값 처리 (정상 범위를 벗어남)
        if val < self.winter_sunlight_min / 2:  # 이상치 처리
            return 0.2  # 낮은 점수 (너무 낮으면 좋지 않음)
        if val > self.winter_sunlight_max * 2:  # 이상치 처리
            return 0.5  # 중간 점수 (너무 높아도 비현실적)
        
        # 최적값 기준 점수 계산 (겨울철은 높을수록 좋음)
        if val >= self.winter_sunlight_optimal:
            # 최적값 이상은 높은 점수
            # 단, 너무 높아도 비현실적이므로 일정 수준 이상이면 점수 감소
            if val > self.winter_sunlight_max:
                # 최대값보다 크면 거리에 따라 감소
                normalized_diff = (val - self.winter_sunlight_max) / self.winter_sunlight_max
                return 0.8 - 0.3 * normalized_diff  # 0.8에서 시작해 감소
            else:
                # 최적 범위 내 (최적값~최대값)
                normalized_pos = (self.winter_sunlight_max - val) / (self.winter_sunlight_max - self.winter_sunlight_optimal)
                return 0.8 + 0.2 * (1 - normalized_pos)  # 0.8에서 1.0까지
        else:
            # 최적값 미만시 점수 감소
            if val < self.winter_sunlight_min:
                # 최소값 미만 시 더 빠르게 감소
                normalized_diff = (self.winter_sunlight_min - val) / self.winter_sunlight_min
                return 0.6 - 0.4 * min(1.0, normalized_diff)  # 0.6에서 0.2까지 감소
            else:
                # 최소값~최적값 사이
                normalized_diff = (self.winter_sunlight_optimal - val) / (self.winter_sunlight_optimal - self.winter_sunlight_min)
                return 1.0 - 0.4 * normalized_diff  # 1.0에서 0.6까지 감소

    def _calculate_sv_ratio_score(self, sv_ratio_value: float) -> float:
        """
        표면적 체적비 값에 대한 점수(0-1) 반환
        최적값에 가까울수록 높은 점수, 낮을수록 좋음
        """
        # 0 이하의 값은 0으로 처리 (비정상 값 방지)
        val = max(0, sv_ratio_value)
        
        # 너무 작거나 큰 값 처리 (정상 범위를 벗어남)
        if val < self.sv_ratio_min / 2:  # 이상치 처리
            return 0.7  # 너무 낮은 값도 현실적이지 않을 수 있으므로 중상 점수
        if val > self.sv_ratio_max * 2:  # 이상치 처리
            return 0.2  # 매우 낮은 점수 (표면적이 너무 높음)
        
        # 최적값 기준 점수 계산 (표면적 체적비는 낮을수록 좋음)
        if val <= self.sv_ratio_optimal:
            # 최적값 이하는 높은 점수
            # 단, 너무 낮아도 비현실적이므로 일정 수준 이하면 점수 감소
            if val < self.sv_ratio_min:
                # 최소값보다 작으면 거리에 따라 감소
                normalized_diff = (self.sv_ratio_min - val) / self.sv_ratio_min
                return 0.8 - 0.3 * normalized_diff  # 0.8에서 시작해 감소
            else:
                # 최적 범위 내 (최소값~최적값)
                normalized_pos = (val - self.sv_ratio_min) / (self.sv_ratio_optimal - self.sv_ratio_min)
                return 0.8 + 0.2 * (1 - normalized_pos)  # 0.8에서 1.0까지
        else:
            # 최적값 초과시 점수 감소
            if val > self.sv_ratio_max:
                # 최대값 초과 시 더 빠르게 감소
                normalized_diff = (val - self.sv_ratio_max) / self.sv_ratio_max
                return 0.6 - 0.4 * min(1.0, normalized_diff)  # 0.6에서 0.2까지 감소
            else:
                # 최적값~최대값 사이
                normalized_diff = (val - self.sv_ratio_optimal) / (self.sv_ratio_max - self.sv_ratio_optimal)
                return 0.8 - 0.2 * normalized_diff  # 0.8에서 0.6까지 감소
    
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
        안정적인 보상 함수 구현 (평활화, 목표값 기반 점수, 적절한 패널티)
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
    # 보상 함수 초기화 (분석 데이터 기반 최적화 설정)
    reward_fn = OptimizedArchitectureRewardFunction(
        bcr_legal_limit_percent=70.0,
        far_legal_min_limit_percent=200.0,
        far_legal_max_limit_percent=500.0,
        
        # 분석 데이터 기반 최적 범위
        bcr_target_min_percent=30.2,
        bcr_target_max_percent=60.4,
        far_target_min_percent=200.0,
        far_target_max_percent=450.0,
        
        # 분석 데이터 기반 일사량 범위
        summer_sunlight_min=74000.0,
        summer_sunlight_max=82000.0,
        summer_sunlight_optimal=75000.0,
        winter_sunlight_min=73000.0,
        winter_sunlight_max=81000.0,
        winter_sunlight_optimal=80000.0,
        
        bcr_weight=20.0, 
        far_weight=20.0,
        summer_sunlight_weight=15.0, 
        winter_sunlight_weight=15.0,
        improvement_weight=10.0,
        legality_violation_penalty=50.0,
        reward_smoothing_factor=0.3,
        zero_state_penalty=-10.0
    )

    # 테스트 상태 [BCR(소수), FAR(소수), SummerTime, WinterTime]
    test_cases = [
        {"name": "이상적인 상태 (최적 범위 내)", "state": [0.45, 4.0, 75000, 80000]},  # 분석된 최적 범위에 맞춘 이상적 상태
        {"name": "BCR 중간 범위", "state": [0.5, 4.0, 75000, 80000]},  # BCR 중간값
        {"name": "BCR 낮은 범위", "state": [0.31, 4.0, 75000, 80000]},  # BCR 낮은 범위 (하지만 최적 구간 내)
        {"name": "BCR 너무 낮음", "state": [0.15, 4.0, 75000, 80000]},  # BCR 너무 낮음 (최적 구간 밖)
        {"name": "BCR 높은 범위", "state": [0.60, 4.0, 75000, 80000]},  # BCR 높은 범위 (하지만 최적 구간 내)
        {"name": "BCR 약간 위반", "state": [0.72, 4.5, 75000, 80000]},  # BCR 약간 위반
        {"name": "FAR 적절함", "state": [0.45, 3.5, 75000, 80000]},  # FAR 적절함
        {"name": "FAR 최소 약간 미달", "state": [0.45, 1.9, 75000, 80000]},  # FAR 약간 미달
        {"name": "FAR 최소 크게 미달", "state": [0.45, 1.0, 75000, 80000]},  # FAR 크게 미달
        {"name": "FAR 최대 약간 초과", "state": [0.45, 5.1, 75000, 80000]},  # FAR 약간 초과
        {"name": "FAR 비정상적으로 높음", "state": [0.45, 6000.0, 75000, 80000]},  # FAR 비정상
        {"name": "여름 일사량 최적", "state": [0.45, 4.0, 75000, 80000]},  # 여름 일사량 최적
        {"name": "여름 일사량 높음", "state": [0.45, 4.0, 81000, 80000]},  # 여름 일사량 높음 (나쁨)
        {"name": "여름 일사량 매우 높음", "state": [0.45, 4.0, 90000, 80000]},  # 여름 일사량 매우 높음 (매우 나쁨)
        {"name": "겨울 일사량 최적", "state": [0.45, 4.0, 75000, 80000]},  # 겨울 일사량 최적
        {"name": "겨울 일사량 낮음", "state": [0.45, 4.0, 75000, 74000]},  # 겨울 일사량 낮음 (나쁨)
        {"name": "겨울 일사량 매우 낮음", "state": [0.45, 4.0, 75000, 65000]},  # 겨울 일사량 매우 낮음 (매우 나쁨)
        {"name": "모든 요소 약간 나쁨", "state": [0.25, 2.5, 84000, 72000]},  # 모든 요소가 약간 최적에서 벗어남
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