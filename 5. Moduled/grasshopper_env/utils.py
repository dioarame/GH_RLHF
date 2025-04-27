# grasshopper_env/utils.py
import requests
import base64
import json
import decimal
import traceback
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class ComputeClient:
    """Rhino.Compute API 호출 및 결과 파싱 클래스"""
    
    def __init__(self, compute_url: str, gh_definition_path: str):
        """
        Rhino.Compute 클라이언트 초기화
        
        Args:
            compute_url: Rhino.Compute 서버 URL
            gh_definition_path: Grasshopper 정의 파일 경로
        """
        self.compute_url = compute_url
        self.gh_definition_path = gh_definition_path
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.retry_count = 3
        self.retry_delay = 0.5
    
    def _create_payload(self, inputs: Dict[str, Any]) -> dict:
        """Compute 요청 페이로드 생성"""
        try:
            with open(self.gh_definition_path, 'rb') as f:
                gh_definition_bytes = f.read()
            gh_definition_b64 = base64.b64encode(gh_definition_bytes).decode('utf-8')
        except Exception as e:
            print(f"❌ 페이로드 생성 중 GH 파일 읽기 오류: {e}")
            return {"algo": None, "values": []}

        values_list = []
        for name, value in inputs.items():
            inner_tree_data = [{"data": value}]
            values_list.append({"ParamName": name, "InnerTree": {"{ 0; }": inner_tree_data}})

        return {
            "algo": gh_definition_b64,
            "pointer": None,
            "values": values_list
        }
    
    def call_compute(self, inputs: Dict[str, Any] = None, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Rhino.Compute 서버 호출"""
        if inputs is None:
            inputs = {}
            
        payload = self._create_payload(inputs)
        if payload.get("algo") is None:
            return None

        try:
            response = self.session.post(self.compute_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"❌ Rhino.Compute request timed out after {timeout} seconds.")
            return None
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ Rhino.Compute HTTP Error: {http_err.response.status_code}")
            try:
                print(f"   Response: {http_err.response.text[:500]}...")
                error_json = http_err.response.json()
                if "errors" in error_json: print(f"   Compute Errors: {error_json['errors']}")
                if "warnings" in error_json: print(f"   Compute Warnings: {error_json['warnings']}")
            except: pass
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Rhino.Compute request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode JSON response: {e}")
            return None
    
    def get_param_data(self, response: Dict[str, Any], param_name: str) -> Optional[str]:
        """Compute 응답에서 특정 파라미터 데이터 추출"""
        if response is None:
            return None
            
        for item in response.get("values", []):
            if item.get("ParamName") == param_name:
                inner_tree = item.get("InnerTree", {})
                first_key = next(iter(inner_tree), None)
                if first_key:
                    # InnerTree의 첫 번째 항목에서 data 추출 시도
                    data_item_list = inner_tree.get(first_key, [])
                    if data_item_list and len(data_item_list) > 0:
                        data_item = data_item_list[0]
                        if isinstance(data_item, dict):
                            data_raw = data_item.get("data")
                            # 문자열 앞뒤 따옴표 제거 (필요시)
                            if isinstance(data_raw, str) and len(data_raw) > 1:
                                if (data_raw.startswith('"') and data_raw.endswith('"')) or (data_raw.startswith("'") and data_raw.endswith("'")):
                                    data_raw = data_raw[1:-1]
                            return data_raw
        return None
    
    def parse_state_reward(self, response: Dict[str, Any], 
                          state_param_name: str, 
                          reward_param_name: str,
                          obs_shape: Tuple[int]) -> Tuple[np.ndarray, float]:
        """Compute 응답에서 상태와 보상 파싱"""
        state_raw = self.get_param_data(response, state_param_name)
        reward_raw = self.get_param_data(response, reward_param_name)
        
        # 기본값 설정
        state = np.zeros(obs_shape, dtype=np.float32)
        reward = -1e3
        
        # 상태 값 파싱
        if state_raw is not None:
            try:
                parsed_state = [float(x.strip()) for x in state_raw.split(',') if x.strip()]
                if len(parsed_state) == obs_shape[0]:
                    state = np.array(parsed_state, dtype=np.float32)
                elif parsed_state:
                    print(f"[Parse Warning] State 차원 불일치 ({len(parsed_state)} vs {obs_shape[0]}). 0벡터 유지.")
            except ValueError:
                print(f"[Parse Warning] State '{state_raw}' 파싱 실패. 0벡터 유지.")

        # 보상 값 파싱
        if reward_raw is not None:
            try:
                reward = float(reward_raw)
            except ValueError:
                print(f"[Parse Warning] Reward '{reward_raw}' 파싱 실패. -1e3 유지.")
                
        return state, reward
    
    def parse_slider_info(self, slider_info_raw: str) -> List[Tuple[float, float, float]]:
        """슬라이더 설정 정보 파싱"""
        slider_infos = []
        print(f"\n[슬라이더 파싱] 원본 데이터: '{slider_info_raw}'")
        
        if slider_info_raw is None:
            return slider_infos
            
        try:
            # 문자열 전처리
            processed_data = slider_info_raw.replace('\\r\\n', '\n').replace('\\n', '\n')
            lines = processed_data.strip().splitlines()
            print(f"{len(lines)}개 라인 발견.")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                print(f"라인 {i+1}: '{line}'")
                
                parts = [part.strip() for part in line.split(',')]
                print(f"  분할 결과: {parts}")

                if len(parts) >= 3:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    rounding = float(parts[2])
                    slider_infos.append((min_val, max_val, rounding))
                    print(f"  파싱 성공: min={min_val}, max={max_val}, rounding={rounding}")
                elif len(parts) >= 2:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    slider_infos.append((min_val, max_val, 0.01))  # 기본 rounding 값
                    print(f"  파싱 성공 (기본 Rounding 적용): min={min_val}, max={max_val}, rounding=0.01")
                else:
                    print(f"  라인 {i+1} 파싱 실패: 충분한 값이 없음")
        except Exception as e:
            print(f"  라인 파싱 중 오류 발생: {e}")
            traceback.print_exc()

        return slider_infos
    
    def check_server(self) -> bool:
        """Rhino.Compute 서버 상태 확인"""
        try:
            base_url = self.compute_url.split('/grasshopper')[0]
            r = requests.get(f"{base_url}/version", timeout=5)
            r.raise_for_status()
            print(f"✅ Rhino.Compute 서버가 작동 중입니다. 버전: {r.json()}")
            return True
        except Exception as e:
            print(f"❌ Rhino.Compute 서버 연결 실패: {e}")
            return False
    
    def close(self):
        """세션 정리"""
        if self.session:
            try:
                self.session.close()
                print("HTTP 세션 닫힘.")
            except Exception as e:
                print(f"HTTP 세션 닫기 중 오류: {e}")


def get_decimal_places(rounding_value):
    """Rounding 값에 기반하여 소수점 자릿수를 반환합니다."""
    try:
        if rounding_value is None or rounding_value <= 0:
            return 10  # 기본값
        d = decimal.Decimal(str(rounding_value))
        exponent = d.as_tuple().exponent
        if isinstance(exponent, int) and exponent < 0:
            return abs(exponent)
        else:
            return 0  # 정수 단위 Rounding
    except Exception as e:
        print(f"Error getting decimal places for {rounding_value}: {e}")
        return 10  # 예외 발생 시 기본값