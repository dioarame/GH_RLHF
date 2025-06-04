#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
보상 함수 어댑터 모듈 (SVR 및 일사량 범위 수정 버전)

이 모듈은 개선된 보상 함수를 RL 환경에 쉽게 통합하기 위한 어댑터를 제공합니다.
기존 코드의 변경을 최소화하면서 새로운 보상 함수를 사용할 수 있습니다.

주요 개선사항:
- SVR 범위를 실제 생성 범위(0.5~0.9)로 수정
- 일사량 상한 제거하여 높을수록 좋은 점수
- BCR/FAR 경제적 효율성 고려한 점수 계산
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Type

# 보상 함수 임포트 시도 - 직접 임포트
try:
    from reward_fn_original import EnhancedArchitectureRewardFunction
    ENHANCED_REWARD_AVAILABLE = True
except ImportError:
    EnhancedArchitectureRewardFunction = None
    ENHANCED_REWARD_AVAILABLE = False
    print("경고: EnhancedArchitectureRewardFunction을 임포트할 수 없습니다.")

try:
    from reward_fn_optimized import OptimizedArchitectureRewardFunction
    OPTIMIZED_REWARD_AVAILABLE = True
except ImportError:
    OptimizedArchitectureRewardFunction = None
    OPTIMIZED_REWARD_AVAILABLE = False
    print("경고: OptimizedArchitectureRewardFunction을 임포트할 수 없습니다.")

try:
    from reward_fn_maxmin import MaxMinDirectionalRewardFunction
    MAXMIN_REWARD_AVAILABLE = True
except ImportError:
    MaxMinDirectionalRewardFunction = None
    MAXMIN_REWARD_AVAILABLE = False
    print("경고: MaxMinDirectionalRewardFunction을 임포트할 수 없습니다.")

try:
    # 원본 보상 함수가 있으면 임포트 시도
    from reward_function import ArchitectureRewardFunction_Seasonal
    ORIGINAL_REWARD_AVAILABLE = True
except ImportError:
    ArchitectureRewardFunction_Seasonal = None
    ORIGINAL_REWARD_AVAILABLE = False
    print("경고: ArchitectureRewardFunction_Seasonal을 임포트할 수 없습니다.")

# 최소한 하나의 보상 함수를 사용할 수 있는지 확인
REWARD_FUNCTIONS_AVAILABLE = (ENHANCED_REWARD_AVAILABLE or OPTIMIZED_REWARD_AVAILABLE or 
                              ORIGINAL_REWARD_AVAILABLE or MAXMIN_REWARD_AVAILABLE)


class RewardFunctionAdapter:
    """
    보상 함수 어댑터 클래스 (개선된 버전)
    
    여러 보상 함수를 통합하여 RL 환경에서 사용할 수 있게 합니다.
    SVR 및 일사량 범위 수정을 반영한 개선된 보상 함수들을 지원합니다.
    """
    
    def __init__(
        self,
        reward_type: str = "optimized",
        
        # 건축 제한 파라미터
        bcr_legal_limit_percent: float = 70.0,
        far_legal_min_limit_percent: float = 200.0,
        far_legal_max_limit_percent: float = 500.0,  # 법적 최대값 유지
        
        # SVR 관련 파라미터
        svr_min: float = 0.6,
        svr_max: float = 0.93,
        svr_target_min: float = 0.7,
        svr_target_max: float = 0.9,
        svr_optimal: float = 0.8,
        
        # 일사량 관련 파라미터
        winter_sunlight_min: float = 70000.0,
        winter_sunlight_good: float = 100000.0,
        winter_sunlight_excellent: float = 140000.0,
        
        # BCR/FAR 효율성 파라미터 (실제 데이터 기반 조정)
        bcr_efficiency_min_percent: float = 45.0,     # 실제 데이터 참고
        bcr_efficiency_max_percent: float = 50.0,     # 실제 데이터 참고
        far_efficiency_min_percent: float = 350.0,    # 법적 범위 내 중상위
        far_efficiency_max_percent: float = 450.0,    # 법적 범위 내 상위
        
        # 추가 설정
        use_seasonal: bool = True,
        debug: bool = False
    ):
        """
        보상 함수 어댑터 초기화 (개선된 버전)
        
        Args:
            reward_type: 사용할 보상 함수 유형
            bcr_legal_limit_percent: BCR 법적 제한 (백분율)
            far_legal_min_limit_percent: 최소 FAR 법적 제한 (백분율)
            far_legal_max_limit_percent: 최대 FAR 법적 제한 (백분율)
            svr_min: SVR 최소값 (실제 생성 범위)
            svr_max: SVR 최대값 (실제 생성 범위)
            svr_target_min: SVR 목표 범위 최소값
            svr_target_max: SVR 목표 범위 최대값
            svr_optimal: SVR 최적값
            winter_sunlight_min: 겨울 일사량 최소 기준
            winter_sunlight_good: 겨울 일사량 양호 기준
            winter_sunlight_excellent: 겨울 일사량 우수 기준
            bcr_efficiency_min_percent: BCR 효율성 최소값 (백분율)
            bcr_efficiency_max_percent: BCR 효율성 최대값 (백분율)
            far_efficiency_min_percent: FAR 효율성 최소값 (백분율)
            far_efficiency_max_percent: FAR 효율성 최대값 (백분율)
            use_seasonal: 계절별 일사량 사용 여부
            debug: 디버그 모드 활성화 여부
        """
        self.reward_type = reward_type
        self.bcr_legal_limit = bcr_legal_limit_percent
        self.far_legal_min_limit = far_legal_min_limit_percent
        self.far_legal_max_limit = far_legal_max_limit_percent
        self.use_seasonal = use_seasonal
        self.debug = debug
        
        # 개선된 파라미터들 저장
        self.svr_params = {
            'svr_min': svr_min,
            'svr_max': svr_max,
            'svr_target_min': svr_target_min,
            'svr_target_max': svr_target_max,
            'svr_optimal': svr_optimal
        }
        
        self.sunlight_params = {
            'winter_sunlight_min': winter_sunlight_min,
            'winter_sunlight_good': winter_sunlight_good,
            'winter_sunlight_excellent': winter_sunlight_excellent
        }
        
        # 효율성 파라미터 저장
        self.efficiency_params = {
        'bcr_efficiency_min_percent': bcr_efficiency_min_percent,
        'bcr_efficiency_max_percent': bcr_efficiency_max_percent,
        'far_efficiency_min_percent': far_efficiency_min_percent,
        'far_efficiency_max_percent': far_efficiency_max_percent
        }
        # 보상 함수 인스턴스
        self.reward_function = self._create_reward_function()
        
        # 보상 함수 이름 및 설정 출력
        print(f"개선된 보상 함수 유형: {reward_type}")
        print(f"BCR 제한: {bcr_legal_limit_percent}%")
        print(f"FAR 제한: {far_legal_min_limit_percent}% ~ {far_legal_max_limit_percent}%")
        print(f"SVR 목표 범위: {svr_target_min} ~ {svr_target_max} (최적: {svr_optimal})")
        print(f"일사량 기준: {winter_sunlight_min} 이상 (상한 없음)")
        print(f"계절별 일사량 사용: {'예' if use_seasonal else '아니오'}")
    
    def _create_reward_function(self):
        """설정에 따라 적절한 보상 함수 인스턴스 생성"""
        
        if not REWARD_FUNCTIONS_AVAILABLE:
            print("경고: 어떤 보상 함수 모듈도 찾을 수 없어 기본 보상 함수를 사용합니다.")
            return self._create_default_reward_function()
        
        # reward_type에 따라 적절한 보상 함수 생성
        if self.reward_type == "original" and ORIGINAL_REWARD_AVAILABLE:
            return self._create_original_reward_function()
            
        elif self.reward_type == "enhanced" and ENHANCED_REWARD_AVAILABLE:
            return self._create_enhanced_reward_function()
            
        elif self.reward_type == "optimized" and OPTIMIZED_REWARD_AVAILABLE:
            return self._create_optimized_reward_function()
            
        elif self.reward_type == "maxmin" and MAXMIN_REWARD_AVAILABLE:
            return self._create_maxmin_reward_function()
        
        # 요청한 보상 함수를 사용할 수 없는 경우 대체 가능한 보상 함수를 찾음
        print(f"요청한 보상 함수 '{self.reward_type}'을(를) 사용할 수 없습니다. 대체 보상 함수를 찾는 중...")
        
        if OPTIMIZED_REWARD_AVAILABLE:
            print("개선된 최적화 보상 함수를 대신 사용합니다.")
            return self._create_optimized_reward_function()
        elif MAXMIN_REWARD_AVAILABLE:
            print("개선된 MaxMin 방향성 보상 함수를 대신 사용합니다.")
            return self._create_maxmin_reward_function()
        elif ENHANCED_REWARD_AVAILABLE:
            print("향상된 보상 함수를 대신 사용합니다.")
            return self._create_enhanced_reward_function()
        elif ORIGINAL_REWARD_AVAILABLE:
            print("원본 보상 함수를 대신 사용합니다.")
            return self._create_original_reward_function()
        else:
            print("사용 가능한 보상 함수가 없습니다. 기본 보상 함수를 사용합니다.")
            return self._create_default_reward_function()
    
    def _create_default_reward_function(self):
        """기본 보상 함수 생성"""
        
        class DefaultRewardFunction:
            def __init__(self):
                self.name = "Default"
                
            def calculate_reward(self, state):
                # 간단한 기본 보상 함수
                bcr, far, winter, svr = state
                
                # BCR, FAR 법적 제한 확인
                bcr_legal = bcr <= 0.7  # 70%
                far_legal_min = far >= 2.0  # 200%
                far_legal_max = far <= 5.0  # 500%
                
                # 법적 제한 위반 시 패널티
                if not bcr_legal or not far_legal_min or not far_legal_max:
                    reward = -10.0
                else:
                    # 간단한 보상 계산
                    reward = 10.0
                
                return reward, {"reward": reward}
                
            def reset_prev_state(self):
                pass
        
        return DefaultRewardFunction()
    
    def _create_original_reward_function(self):
        """원본 보상 함수 생성 (기존 방식 유지)"""
        if not ORIGINAL_REWARD_AVAILABLE or ArchitectureRewardFunction_Seasonal is None:
            print("원본 보상 함수를 사용할 수 없습니다. 기본 보상 함수를 대신 사용합니다.")
            return self._create_default_reward_function()
        
        try:
            # 원본 보상 함수 초기화 (기존 방식)
            return ArchitectureRewardFunction_Seasonal(
                bcr_legal_limit_percent=self.bcr_legal_limit,
                far_legal_min_limit_percent=self.far_legal_min_limit,
                far_legal_max_limit_percent=self.far_legal_max_limit,
                summer_sunlight_norm_cap=200000.0,
                winter_sunlight_norm_cap=200000.0,
                
                # 계절별 사용 여부에 따라 가중치 조정
                summer_sunlight_weight=20.0 if self.use_seasonal else 10.0,
                winter_sunlight_weight=20.0 if self.use_seasonal else 10.0,
            )
        except Exception as e:
            print(f"원본 보상 함수 초기화 중 오류: {e}. 기본 보상 함수를 대신 사용합니다.")
            return self._create_default_reward_function()
    
    def _create_enhanced_reward_function(self):
        """향상된 보상 함수 생성 (기존 방식 유지)"""
        if not ENHANCED_REWARD_AVAILABLE or EnhancedArchitectureRewardFunction is None:
            print("향상된 보상 함수를 사용할 수 없습니다. 기본 보상 함수를 대신 사용합니다.")
            return self._create_default_reward_function()
            
        try:
            # 향상된 보상 함수 초기화 (기존 방식)
            return EnhancedArchitectureRewardFunction(
                bcr_legal_limit_percent=self.bcr_legal_limit,
                far_legal_min_limit_percent=self.far_legal_min_limit,
                far_legal_max_limit_percent=self.far_legal_max_limit,
                bcr_target_percent=65.0,
                far_target_percent=450.0,
                summer_sunlight_norm_cap=200000.0,
                winter_sunlight_norm_cap=200000.0,
                summer_sunlight_optimal=80000.0,
                winter_sunlight_optimal=150000.0,
                
                # 계절별 가중치 조정
                summer_sunlight_weight=15.0 if self.use_seasonal else 7.5,
                winter_sunlight_weight=15.0 if self.use_seasonal else 7.5,
            )
        except Exception as e:
            print(f"향상된 보상 함수 초기화 중 오류: {e}. 기본 보상 함수를 대신 사용합니다.")
            return self._create_default_reward_function()
    
    def _create_optimized_reward_function(self):
        """개선된 최적화 보상 함수 생성"""
        if not OPTIMIZED_REWARD_AVAILABLE or OptimizedArchitectureRewardFunction is None:
            print("최적화된 보상 함수를 사용할 수 없습니다. 향상된 보상 함수를 대신 사용합니다.")
            if ENHANCED_REWARD_AVAILABLE:
                return self._create_enhanced_reward_function()
            else:
                return self._create_default_reward_function()
            
        try:
            # 개선된 최적화 보상 함수 초기화
            return OptimizedArchitectureRewardFunction(
                bcr_legal_limit_percent=self.bcr_legal_limit,
                far_legal_min_limit_percent=self.far_legal_min_limit,
                far_legal_max_limit_percent=self.far_legal_max_limit,
                
                # 경제적 효율성 목표
                bcr_efficiency_min_percent=self.efficiency_params['bcr_efficiency_min_percent'],
                bcr_efficiency_max_percent=self.efficiency_params['bcr_efficiency_max_percent'],
                far_efficiency_min_percent=self.efficiency_params['far_efficiency_min_percent'],
                far_efficiency_max_percent=self.efficiency_params['far_efficiency_max_percent'],
                
                # SVR 실제 범위 반영
                svr_min=self.svr_params['svr_min'],
                svr_max=self.svr_params['svr_max'],
                svr_target_min=self.svr_params['svr_target_min'],
                svr_target_max=self.svr_params['svr_target_max'],
                svr_optimal=self.svr_params['svr_optimal'],
                
                # 일사량 기준 (상한 제거)
                winter_sunlight_min=self.sunlight_params['winter_sunlight_min'],
                winter_sunlight_good=self.sunlight_params['winter_sunlight_good'],
                winter_sunlight_excellent=self.sunlight_params['winter_sunlight_excellent'],
                
                # 가중치 설정
                bcr_weight=20.0,
                far_weight=20.0,
                svr_weight=15.0 if self.use_seasonal else 7.5,
                winter_sunlight_weight=15.0 if self.use_seasonal else 7.5,
            )
        except Exception as e:
            print(f"최적화된 보상 함수 초기화 중 오류: {e}. 향상된 보상 함수를 대신 사용합니다.")
            if ENHANCED_REWARD_AVAILABLE:
                return self._create_enhanced_reward_function()
            else:
                return self._create_default_reward_function()
    
    def _create_maxmin_reward_function(self):
        """개선된 MaxMin 방향성 보상 함수 생성"""
        if not MAXMIN_REWARD_AVAILABLE or MaxMinDirectionalRewardFunction is None:
            print("MaxMin 방향성 보상 함수를 사용할 수 없습니다. 최적화된 보상 함수를 대신 사용합니다.")
            if OPTIMIZED_REWARD_AVAILABLE:
                return self._create_optimized_reward_function()
            elif ENHANCED_REWARD_AVAILABLE:
                return self._create_enhanced_reward_function()
            else:
                return self._create_default_reward_function()
            
        try:
            # 개선된 MaxMin 방향성 보상 함수 초기화
            return MaxMinDirectionalRewardFunction(
                bcr_legal_limit_percent=self.bcr_legal_limit,
                far_legal_min_limit_percent=self.far_legal_min_limit,
                far_legal_max_limit_percent=self.far_legal_max_limit,
                
                # 일사량 정규화 기준 (상한 제거)
                winter_sunlight_min=self.sunlight_params['winter_sunlight_min'],
                winter_sunlight_excellent=self.sunlight_params['winter_sunlight_excellent'],
                
                # SVR 정규화 기준 (실제 범위)
                svr_min=self.svr_params['svr_min'],
                svr_max=self.svr_params['svr_max'],
                svr_target_min=self.svr_params['svr_target_min'],
                svr_target_max=self.svr_params['svr_target_max'],
                svr_optimal=self.svr_params['svr_optimal'],
                
                # 가중치 설정
                bcr_weight=20.0,
                far_weight=20.0,
                winter_sunlight_weight=15.0 if self.use_seasonal else 7.5,
                svr_weight=15.0 if self.use_seasonal else 7.5,
                improvement_weight=10.0,
                
                # 기타 설정
                legality_violation_penalty=30.0,
                reward_smoothing_factor=0.3,
                zero_state_penalty=-10.0
            )
        except Exception as e:
            print(f"MaxMin 방향성 보상 함수 초기화 중 오류: {e}. 최적화된 보상 함수를 대신 사용합니다.")
            if OPTIMIZED_REWARD_AVAILABLE:
                return self._create_optimized_reward_function()
            elif ENHANCED_REWARD_AVAILABLE:
                return self._create_enhanced_reward_function()
            else:
                return self._create_default_reward_function()
    
    def calculate_reward(self, state: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        상태 벡터에 기반한 보상 계산
        
        Args:
            state: 상태 벡터 [BCR, FAR, WinterTime, SVR] 또는 [BCR, FAR, Sunlight]
            
        Returns:
            Tuple[float, Dict[str, Any]]: (보상 값, 추가 정보)
        """
        # 상태 벡터 형식 검증
        if not isinstance(state, list):
            raise ValueError("상태는 리스트 형식이어야 합니다.")
        
        # 3차원 상태를 4차원으로 확장 (필요한 경우)
        if len(state) == 3:
            bcr, far, sunlight = state
            # SVR 기본값 설정 (중간값 사용)
            default_svr = (self.svr_params['svr_target_min'] + self.svr_params['svr_target_max']) / 2
            state_4d = [bcr, far, sunlight, default_svr]
        elif len(state) == 4:
            state_4d = state
        else:
            raise ValueError(f"상태는 3차원 또는 4차원이어야 합니다. 현재: {len(state)}차원")
        
        try:
            # 보상 계산
            reward, info = self.reward_function.calculate_reward(state_4d)
        except Exception as e:
            print(f"보상 계산 중 오류: {e}. 기본 보상을 반환합니다.")
            # 오류 발생 시 기본값 반환
            reward = -5.0 if state_4d[0] > 0.7 or state_4d[1] < 2.0 or state_4d[1] > 5.0 else 5.0
            info = {"reward": reward, "error": str(e)}
        
        # 디버그 모드일 경우 추가 정보 출력
        if self.debug:
            print(f"상태: {state_4d}")
            print(f"보상: {reward}")
            print(f"상세 정보: {info}")
        
        return reward, info
    
    def reset_prev_state(self):
        """이전 상태 초기화 (에피소드 시작 시 호출)"""
        if hasattr(self.reward_function, 'reset_prev_state'):
            self.reward_function.reset_prev_state()


# 쉬운 사용을 위한 팩토리 함수 (개선된 버전)
def create_reward_function(
    reward_type: str = "optimized",
    bcr_limit: float = 70.0,
    far_min: float = 200.0,
    far_max: float = 500.0,
    
    # SVR 관련 파라미터
    svr_min: float = 0.6,
    svr_max: float = 0.93,
    svr_target_min: float = 0.7,
    svr_target_max: float = 0.9,
    svr_optimal: float = 0.8,
    
    # 일사량 관련 파라미터
    winter_sunlight_min: float = 70000.0,
    winter_sunlight_good: float = 100000.0,
    winter_sunlight_excellent: float = 140000.0,
    
    use_seasonal: bool = True,
    debug: bool = False
) -> RewardFunctionAdapter:
    """보상 함수 생성 팩토리 함수"""
    
    return RewardFunctionAdapter(
        reward_type=reward_type,
        bcr_legal_limit_percent=bcr_limit,
        far_legal_min_limit_percent=far_min,
        far_legal_max_limit_percent=far_max,
        
        # SVR 파라미터
        svr_min=svr_min,
        svr_max=svr_max,
        svr_target_min=svr_target_min,
        svr_target_max=svr_target_max,
        svr_optimal=svr_optimal,
        
        # 일사량 파라미터
        winter_sunlight_min=winter_sunlight_min,
        winter_sunlight_good=winter_sunlight_good,
        winter_sunlight_excellent=winter_sunlight_excellent,
        
        # 효율성 파라미터 (실제 데이터 기반 기본값)
        bcr_efficiency_min_percent=45.0,
        bcr_efficiency_max_percent=50.0,
        far_efficiency_min_percent=350.0,
        far_efficiency_max_percent=450.0,
        
        use_seasonal=use_seasonal,
        debug=debug
    )


# 테스트 예제 (개선된 범위 반영)
if __name__ == "__main__":
    # 개선된 보상 함수 생성
    reward_fn = create_reward_function(
        reward_type="optimized",  # 개선된 최적화 보상 함수 사용
        bcr_limit=70.0,
        far_min=200.0,
        far_max=500.0,
        
        # SVR 실제 범위 반영
        svr_min=0.5,
        svr_max=0.9,
        svr_target_min=0.7,
        svr_target_max=0.9,
        svr_optimal=0.8,
        
        # 일사량 기준 (상한 제거)
        winter_sunlight_min=50000.0,
        winter_sunlight_good=80000.0,
        winter_sunlight_excellent=100000.0,
        
        use_seasonal=True,
        debug=True
    )
    
    # 테스트 상태 (개선된 범위 반영)
    test_states = [
        [0.65, 4.5, 95000, 0.8],     # 최적 조합
        [0.69, 4.9, 120000, 0.75],   # 경제적 효율성 높음
        [0.72, 4.5, 85000, 0.8],     # BCR 위반
        [0.55, 1.9, 80000, 0.85],    # FAR 최소 위반
        [0.60, 4.0, 30000, 0.6],     # 낮은 일사량, SVR 목표 범위 밖
        [0.66, 4.8, 150000, 0.85],   # 초고 일사량 (보너스 테스트)
    ]
    
    print("\n=== 개선된 보상 함수 테스트 ===")
    reward_fn.reset_prev_state()
    
    for i, state in enumerate(test_states):
        print(f"\n테스트 {i+1}: 상태 = BCR={state[0]*100:.1f}%, FAR={state[1]*100:.1f}%, Winter={state[2]}, SVR={state[3]}")
        reward, info = reward_fn.calculate_reward(state)
        print(f"보상: {reward:.4f}")
        
        # 개선된 보상 함수 결과 세부 정보 출력
        if 'bcr_score' in info:
            print(f"  BCR 점수: {info['bcr_score']:.3f}, FAR 점수: {info['far_score']:.3f}")
            print(f"  겨울 일사량 점수: {info['winter_score']:.3f}, SVR 점수: {info['svr_score']:.3f}")
            
            if info.get('legal_violation_detected', False):
                print(f"  법규 위반: {', '.join(info['violation_details'])}")
            else:
                print(f"  법규 준수: BCR {info['bcr_val_percent']:.1f}%, FAR {info['far_val_percent']:.1f}%")