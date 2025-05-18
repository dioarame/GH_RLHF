#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축 설계 최적화를 위한 보상 함수 모듈 (계절별 성능 고려 버전)
입력: BCR, FAR, 여름철 일사량(SummerTime), 겨울철 일사량(WinterTime)
목표:
- BCR/FAR: 법규 내에서 경제적 가치(높은 밀도) 추구 및 적정 수준 유지
- SummerTime: 낮을수록 좋음 (냉방 부하 감소)
- WinterTime: 높을수록 좋음 (난방 부하 감소, 일사 획득)
- 전체 보상: 약 0 ~ 100점 범위
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ArchitectureRewardFunction_Seasonal:
    """건축 설계 최적화를 위한 보상 함수 클래스 (계절별 성능 고려)"""

    def __init__(
        self,
        # --- 법적 제한 (Legal Limits) ---
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,

        # --- 일사량 정규화 기준 (Normalization Caps) ---
        # 에이전트가 생성하는 값의 예상 범위 또는 이상적인 최대치를 고려하여 설정
        # 현재 여름 126,892 / 겨울 134,457 값을 참고
        summer_sunlight_norm_cap: float = 200000.0, # 이 값을 넘어가면 여름 점수 0
        winter_sunlight_norm_cap: float = 200000.0, # 이 값에 가까워지면 겨울 점수 1

        # --- 가중치 (Weights) - 총합 약 80점 목표 ---
        bcr_target_weight: float = 20.0,
        far_target_weight: float = 20.0,
        summer_sunlight_weight: float = 20.0,
        winter_sunlight_weight: float = 20.0,
        
        # --- 개선 인센티브 가중치 - 약 20점 목표 ---
        improvement_total_weight: float = 20.0, # 단순화 버전에서는 prev_scores가 아닌 prev_reward와 비교할 수도 있음
        
        # --- 패널티 가중치 (Penalty Weights) ---
        legality_violation_penalty_factor: float = 50.0,
        
        # --- 기타 설정 ---
        zero_state_penalty: float = -100.0
    ):
        # 법적 제한
        self.bcr_legal_limit = bcr_legal_limit_percent / 100.0
        self.far_legal_min_limit = far_legal_min_limit_percent / 100.0
        self.far_legal_max_limit = far_legal_max_limit_percent / 100.0

        # 일사량 정규화 기준
        self.summer_sunlight_norm_cap = summer_sunlight_norm_cap
        self.winter_sunlight_norm_cap = winter_sunlight_norm_cap
        
        # 가중치
        self.bcr_target_weight = bcr_target_weight
        self.far_target_weight = far_target_weight
        self.summer_sunlight_weight = summer_sunlight_weight
        self.winter_sunlight_weight = winter_sunlight_weight
        self.improvement_total_weight = improvement_total_weight # 이 부분은 나중에 단순화 가능성
        
        self.legality_violation_penalty_factor = legality_violation_penalty_factor
        self.zero_state_penalty = zero_state_penalty
        
        # 이전 상태 (단순화된 개선 인센티브를 위해 prev_total_reward만 저장할 수도 있음)
        self.prev_scores: Optional[Dict[str, float]] = None # 이전 점수 방식 유지 또는
        self.prev_total_reward: Optional[float] = None      # 이전 총 보상만 저장
        self.prev_state_values: Optional[List[float]] = None


    def _calculate_bcr_score(self, bcr_value: float) -> Tuple[float, bool]:
        """BCR 값에 대한 점수(0-1)와 법규 위반 여부를 반환. 상한(70%)에 가까울수록 높은 점수."""
        violated = False
        if bcr_value > self.bcr_legal_limit or bcr_value <= 0: # 0 이하도 비정상으로 간주
            violated = True
            return 0.0, violated # 법규 위반 시 0점
        
        # 법규 내에서는 상한에 가까울수록 선형적으로 점수 증가 (0 ~ 1)
        # (bcr_value / self.bcr_legal_limit) 로 단순화 가능, 또는 특정 목표치 설정 가능
        # 여기서는 법적 상한에 도달했을 때 1점
        score = bcr_value / self.bcr_legal_limit 
        return np.clip(score, 0.0, 1.0), violated

    def _calculate_far_score(self, far_value: float) -> Tuple[float, bool, bool]:
        """FAR 값에 대한 점수(0-1)와 최소/최대 법규 위반 여부를 반환.
           법규 내에서는 상한(500%)에 가까울수록 높은 점수, 하한(200%)은 만족해야 함."""
        min_violated = False
        max_violated = False

        if far_value < self.far_legal_min_limit:
            min_violated = True
            return 0.0, min_violated, max_violated # 최소치 미달 시 0점
        if far_value > self.far_legal_max_limit:
            max_violated = True
            return 0.0, min_violated, max_violated # 최대치 초과 시 0점
        
        # 법규 범위 내에서는 최대 상한에 가까울수록 점수 증가 (0 ~ 1)
        # (far_value - self.far_legal_min_limit) / (self.far_legal_max_limit - self.far_legal_min_limit)
        # 위 공식은 최소일 때 0점, 최대일 때 1점.
        # 여기서는 "높을수록 좋다"는 개념을 반영하여, 법적 최대치 대비 비율로 점수화.
        # 단, 이렇게 하면 far_legal_min_limit 근처 값들이 너무 낮은 점수를 받을 수 있음.
        # 사용자 요구사항(높을수록 경제적 가치)을 고려, far_max_limit에 가까울수록 1점에 수렴하도록.
        # (far_value / self.far_legal_max_limit) 를 사용하면 min_limit 조건이 약해질 수 있음.
        # 좀 더 정교하게는, min_limit을 기준으로 한 스케일링과 max_limit을 기준으로 한 스케일링을 조합.
        # 여기서는 (value - min) / (max - min) 공식을 사용하여 최소~최대 사이를 0~1로 매핑
        score = (far_value - self.far_legal_min_limit) / (self.far_legal_max_limit - self.far_legal_min_limit)
        return np.clip(score, 0.0, 1.0), min_violated, max_violated

    def _calculate_summer_sunlight_score_simple(self, summer_sunlight_value: float) -> float:
        """여름철 일사량 값에 대한 점수(0-1) 반환. 낮을수록 높은 점수 (단순 버전)."""
        # 0 이하의 값은 0으로 처리 (비정상 값 방지)
        val = max(0, summer_sunlight_value)
        normalized_summer = min(1.0, val / self.summer_sunlight_norm_cap)
        score = 1.0 - normalized_summer
        return score

    def _calculate_winter_sunlight_score_simple(self, winter_sunlight_value: float) -> float:
        """겨울철 일사량 값에 대한 점수(0-1) 반환. 높을수록 높은 점수 (단순 버전)."""
        val = max(0, winter_sunlight_value)
        normalized_winter = min(1.0, val / self.winter_sunlight_norm_cap)
        score = normalized_winter
        return score

    def calculate_reward(self, state: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        현재 상태 [BCR, FAR, SummerTime, WinterTime]를 기반으로 보상을 계산.
        BCR, FAR은 소수점 값 (예: 0.6 for 60%).
        SummerTime, WinterTime은 Ladybug 분석 결과 원본값.
        """
        if not isinstance(state, list) or len(state) != 4:
            raise ValueError("State must be a list of 4 floats: [BCR, FAR, SummerTime, WinterTime]")

        bcr, far, summer_sunlight, winter_sunlight = state

        # 오류 상태 처리 (모든 값이 0 또는 비정상적인 경우)
        if bcr <= 0 and far <= 0: # 일사량은 0일 수 있음
            info = {'reward': self.zero_state_penalty, 'error': "Invalid state (BCR/FAR are zero or negative)"}
            info.update({
                'bcr_val_percent': bcr * 100.0, 'far_val_percent': far * 100.0,
                'summer_sunlight_val': summer_sunlight, 'winter_sunlight_val': winter_sunlight,
                'bcr_score': 0, 'far_score': 0, 'summer_score': 0, 'winter_score': 0,
                'bcr_violated': True, 'far_min_violated': True, 'far_max_violated': False, # 가정
                'base_reward_before_penalty': 0, 'legality_penalty': abs(self.zero_state_penalty),
                'improvement_bonus': 0
            })
            return self.zero_state_penalty, info
        
        # 1. 각 요소별 점수 계산 (0-1 범위)
        bcr_score, bcr_violated = self._calculate_bcr_score(bcr)
        far_score, far_min_violated, far_max_violated = self._calculate_far_score(far)
        # summer_score = self._calculate_summer_sunlight_score(summer_sunlight) # 이전 방식
        # winter_score = self._calculate_winter_sunlight_score(winter_sunlight) # 이전 방식
        summer_score = self._calculate_summer_sunlight_score_simple(summer_sunlight) # 단순화 방식
        winter_score = self._calculate_winter_sunlight_score_simple(winter_sunlight) # 단순화 방식

        current_scores = {
            "bcr": bcr_score,
            "far": far_score,
            "summer": summer_score,
            "winter": winter_score
        }

        # 2. 기본 보상 계산 (가중치 적용, 최대 약 80점)
        weighted_bcr_reward = bcr_score * self.bcr_target_weight
        weighted_far_reward = far_score * self.far_target_weight
        weighted_summer_reward = summer_score * self.summer_sunlight_weight
        weighted_winter_reward = winter_score * self.winter_sunlight_weight
        
        base_reward_before_penalty = (weighted_bcr_reward + weighted_far_reward + 
                                      weighted_summer_reward + weighted_winter_reward)

        # 3. 법규 위반 패널티 적용
        legality_penalty = 0.0
        if bcr_violated:
            legality_penalty += self.legality_violation_penalty_factor
        if far_min_violated:
            legality_penalty += self.legality_violation_penalty_factor
        if far_max_violated: # 최대치 초과는 다른 종류의 위반으로 볼 수도 있음
            legality_penalty += self.legality_violation_penalty_factor * 1.5 # 예: 더 큰 페널티

        current_design_score = base_reward_before_penalty - legality_penalty
        
        # 4. 개선 인센티브 계산 (최대 약 20점)
        improvement_bonus = 0.0
        # 이전 상태의 점수와 현재 상태의 점수를 비교
        if self.prev_scores is not None:
            # 각 항목별 점수 변화량 합산 (0~1 스케일 점수 기준)
            # 개선되었으면 양수, 나빠졌으면 음수
            delta_bcr_score = current_scores["bcr"] - self.prev_scores["bcr"]
            delta_far_score = current_scores["far"] - self.prev_scores["far"]
            delta_summer_score = current_scores["summer"] - self.prev_scores["summer"]
            delta_winter_score = current_scores["winter"] - self.prev_scores["winter"]
            
            # 모든 점수 변화를 합산 (각 항목의 최대 변화는 1점)
            # 총 변화량은 -4점에서 +4점 사이가 될 수 있음.
            total_score_improvement = delta_bcr_score + delta_far_score + delta_summer_score + delta_winter_score
            
            # improvement_total_weight를 전체 점수 범위에 맞게 스케일링
            # 예: total_score_improvement가 +4일 때 improvement_total_weight 만큼의 보너스
            # total_score_improvement가 0 근처면 보너스 거의 없음
            # total_score_improvement가 음수면 페널티
            # 여기서는 단순하게 변화량에 가중치를 곱함. (최대 improvement_total_weight * 4점 가능성 -> 스케일링 필요)
            # 변화량의 합이 최대 4이므로, (total_score_improvement / 4.0) * self.improvement_total_weight 로 하면
            # 최대 improvement_total_weight 만큼의 보너스/페널티가 됨.
            improvement_bonus = (total_score_improvement / 4.0) * self.improvement_total_weight
            improvement_bonus = np.clip(improvement_bonus, -self.improvement_total_weight, self.improvement_total_weight)


        # 5. 최종 보상 계산
        final_reward = current_design_score + improvement_bonus
        final_reward = np.clip(final_reward, -100.0, 100.0) # 최종 보상 범위를 -100 ~ 100으로 제한 (선택적)

        # 이전 상태 업데이트 (다음 스텝을 위해)
        self.prev_scores = current_scores.copy()
        self.prev_state_values = state.copy()

        # 보상 계산 세부 정보
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
            'legality_penalty': legality_penalty, # 차감된 페널티 값
            
            'current_design_score_after_penalty': current_design_score,
            'improvement_bonus': improvement_bonus,
            'final_reward': final_reward
        }
        
        return final_reward, info

    def reset_prev_state(self):
        """에피소드 시작 시 이전 상태 관련 변수들을 초기화합니다."""
        self.prev_scores = None
        self.prev_state_values = None


# === 테스트 코드 예시 ===
if __name__ == "__main__":
    # 보상 함수 초기화 (기본값 사용 또는 필요시 파라미터 조정)
    reward_fn = ArchitectureRewardFunction_Seasonal(
        bcr_legal_limit_percent=70.0,
        far_legal_min_limit_percent=200.0,
        far_legal_max_limit_percent=500.0,
        summer_sunlight_optimal_max=150000.0, # 예시 값, 실제 Ladybug 결과 기반으로 튜닝
        winter_sunlight_optimal_min=300000.0, # 예시 값
        sunlight_extreme_summer=400000.0,
        sunlight_extreme_winter=50000.0,
        bcr_target_weight=20, far_target_weight=20,
        summer_sunlight_weight=20, winter_sunlight_weight=20,
        improvement_total_weight=20,
        legality_violation_penalty_factor=50 # 위반 항목당 50점 감점
    )

    # 테스트 상태 [BCR(소수), FAR(소수), SummerTime, WinterTime]
    test_cases = [
        {"name": "이상적인 상태 (첫 스텝)", "state": [0.65, 4.8, 100000, 350000]},
        {"name": "개선된 상태", "state": [0.70, 5.0, 80000, 400000]}, # 모든 면에서 개선
        {"name": "BCR 위반", "state": [0.75, 4.5, 120000, 320000]},
        {"name": "FAR 최소 미달", "state": [0.60, 1.8, 130000, 280000]},
        {"name": "FAR 최대 초과", "state": [0.60, 5.2, 130000, 280000]},
        {"name": "여름 일사량 나쁨", "state": [0.60, 4.0, 450000, 310000]},
        {"name": "겨울 일사량 나쁨", "state": [0.60, 4.0, 140000, 40000]},
        {"name": "모든 것이 나쁨 (법규는 지킴)", "state": [0.20, 2.1, 380000, 60000]}, # BCR, FAR 낮고, 여름 높고, 겨울 낮음
        {"name": "이전보다 약간 나빠짐", "state": [0.68, 4.9, 90000, 380000]}, # "개선된 상태"에서 약간 나빠짐
        {"name": "오류 상태", "state": [0.0, 0.0, 0, 0]},
    ]

    print("===== 계절별 보상 함수 테스트 =====")
    reward_fn.reset_prev_state() # 에피소드 시작 시 호출 가정

    for i, case in enumerate(test_cases):
        state_values = case["state"]
        print(f"\n--- 테스트: {case['name']} ---")
        print(f"입력 상태: BCR={state_values[0]*100:.1f}%, FAR={state_values[1]*100:.1f}%, Summer={state_values[2]}, Winter={state_values[3]}")

        if reward_fn.prev_state_values: # 이전 상태 값 출력 (디버깅용)
             print(f"이전 상태: BCR={reward_fn.prev_state_values[0]*100:.1f}%, FAR={reward_fn.prev_state_values[1]*100:.1f}%, Summer={reward_fn.prev_state_values[2]}, Winter={reward_fn.prev_state_values[3]}")
        
        reward, info = reward_fn.calculate_reward(state_values)
        
        print(f"최종 보상: {info['final_reward']:.2f}")
        print(f"  세부 점수: BCR={info['bcr_score']:.2f}, FAR={info['far_score']:.2f}, Summer={info['summer_score']:.2f}, Winter={info['winter_score']:.2f}")
        print(f"  가중치 적용 보상 (패널티 전): {info['base_reward_before_penalty']:.2f}")
        if info['legality_penalty'] > 0:
            print(f"  법규 위반 페널티: -{info['legality_penalty']:.2f}")
            if info['bcr_violated']: print("    * BCR 위반")
            if info['far_min_violated']: print("    * FAR 최소 미달")
            if info['far_max_violated']: print("    * FAR 최대 초과")
        if 'error' in info:
             print(f"  오류: {info['error']}")
        
        print(f"  개선 인센티브: {info['improvement_bonus']:.2f}")

    # 특정 시나리오: 이전 상태 대비 개선/악화 상세 확인
    print("\n--- 개선/악화 시나리오 테스트 ---")
    reward_fn.reset_prev_state()
    
    # 1. 기준 상태
    print("\n1. 기준 상태:")
    state1 = [0.60, 4.0, 150000, 300000] # 모든 요소가 중간 정도
    reward1, info1 = reward_fn.calculate_reward(state1)
    print(f"   상태: {state1}, 보상: {reward1:.2f}")
    print(f"   점수: BCR={info1['bcr_score']:.2f}, FAR={info1['far_score']:.2f}, Summer={info1['summer_score']:.2f}, Winter={info1['winter_score']:.2f}")


    # 2. 모든 면에서 개선된 상태
    print("\n2. 모든 면에서 개선:")
    state2 = [0.65, 4.5, 120000, 350000]
    reward2, info2 = reward_fn.calculate_reward(state2)
    print(f"   상태: {state2}, 보상: {reward2:.2f} (개선 보너스: {info2['improvement_bonus']:.2f})")
    print(f"   점수: BCR={info2['bcr_score']:.2f}, FAR={info2['far_score']:.2f}, Summer={info2['summer_score']:.2f}, Winter={info2['winter_score']:.2f}")

    # 3. 일부는 개선, 일부는 악화
    print("\n3. 일부 개선, 일부 악화:")
    state3 = [0.68, 4.2, 180000, 330000] # BCR 개선, FAR 악화, Summer 악화, Winter 개선
    reward3, info3 = reward_fn.calculate_reward(state3)
    print(f"   상태: {state3}, 보상: {reward3:.2f} (개선 보너스/페널티: {info3['improvement_bonus']:.2f})")
    print(f"   점수: BCR={info3['bcr_score']:.2f}, FAR={info3['far_score']:.2f}, Summer={info3['summer_score']:.2f}, Winter={info3['winter_score']:.2f}")