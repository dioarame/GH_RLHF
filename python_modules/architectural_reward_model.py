import torch
import torch.nn as nn
import numpy as np
import os
import sys

# === sklearn 임포트를 여기에 추가 ===
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available for improved model support")

# === 실제 모델 클래스 정의 ===
class ProbabilisticArchitecturalModel_V2(nn.Module):
    """
    확률적 건축 보상 모델 V2 - gemini_reward_model_train.py와 동일한 구조
    """
    def __init__(self, state_dim=4, hidden_dims=[128, 64, 32]):
        super(ProbabilisticArchitecturalModel_V2, self).__init__()
        
        # gemini_reward_model_train.py와 동일한 구조
        self.performance_processor = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32)
        )
        
        self.sv_ratio_transformer = nn.Sequential(
            nn.Linear(1, 16), 
            nn.ReLU(), 
            nn.Linear(16, 8)
        )
        
        self.conditional_gate = nn.Sequential(
            nn.Linear(8, 16), 
            nn.ReLU(), 
            nn.Linear(16, 32), 
            nn.Sigmoid()
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(32 + 8, hidden_dims[1]),  # 40 → 64
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),  # 64 → 32
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[2], 1)  # 32 → 1
        )
        
        # 가중치 초기화 적용
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        순전파 - gemini_reward_model_train.py와 동일한 로직
        """
        perf_metrics = state[:, :3]  # BCR, FAR, Sunlight
        sv_ratio = state[:, 3:4]     # SV_Ratio
        
        # 성능 지표 처리
        perf_features = self.performance_processor(perf_metrics)
        
        # SV ratio 처리
        sv_value = self.sv_ratio_transformer(sv_ratio)
        
        # 조건부 게이팅 (Sigmoid 활성화)
        gate_weights = self.conditional_gate(sv_value)
        
        # 게이팅된 성능 특징
        gated_perf_features = perf_features * gate_weights
        
        # 최종 특징 결합
        combined_features = torch.cat([gated_perf_features, sv_value], dim=-1)
        
        # 보상 예측
        return self.reward_head(combined_features).squeeze(-1)

# === SimplifiedRewardModel 클래스를 여기에 추가 ===
class SimplifiedRewardModel(nn.Module):
    """개선된 단순화된 보상 모델"""
    def __init__(self, state_dim=4, hidden_dim=96):
        super(SimplifiedRewardModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                if module.out_features == 1:
                    nn.init.constant_(module.bias, -0.02)
                else:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state):
        return self.network(state)

# === 기존 테스트 코드 (순환 임포트 제거) ===

# --- 경로 설정 ---
current_script_dir_abs = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_script_dir_abs, '..', '..'))
MODULES_DIR = os.path.join(BASE_DIR, "python_modules")

if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)
    print(f"Added '{MODULES_DIR}' to sys.path for importing.")

# 🔧 순환 임포트 제거: 이미 이 파일에서 정의했으므로 임포트 불필요
print("ProbabilisticArchitecturalModel_V2 defined in this module")

def test_specific_reward_model(model_filename, test_states):
    """
    지정된 보상 모델 파일과 테스트 상태를 사용하여 보상을 예측하고 출력합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = os.path.join(current_script_dir_abs, model_filename)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Successfully loaded checkpoint from {model_filename}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 모델 생성 시 필요한 state_dim과 hidden_dims를 checkpoint에서 가져옴
    state_dim = checkpoint.get('state_dim', 4)
    config_from_checkpoint = checkpoint.get('config', {})
    hidden_dims = config_from_checkpoint.get('hidden_dims', [128, 64, 32])

    try:
        model = ProbabilisticArchitecturalModel_V2(state_dim=state_dim, hidden_dims=hidden_dims)
        print(f"Instantiated ProbabilisticArchitecturalModel_V2 with state_dim={state_dim} and hidden_dims={hidden_dims}")
    except Exception as e:
        print(f"Error instantiating ProbabilisticArchitecturalModel_V2: {e}")
        return

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded state_dict into ProbabilisticArchitecturalModel_V2.")
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict: {e}")
        print("This indicates a mismatch between the model architecture defined in the script and the one in the .pt file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading state_dict: {e}")
        return

    model.to(device)
    model.eval()

    scaler_mean = checkpoint.get('scaler_mean')
    scaler_scale = checkpoint.get('scaler_scale')

    if scaler_mean is None or scaler_scale is None:
        print("Warning: Scaler information (mean/scale) not found in the model file. Inputs will not be scaled.")
        use_scaler = False
    else:
        scaler_mean = np.array(scaler_mean, dtype=np.float32)
        scaler_scale = np.array(scaler_scale, dtype=np.float32)
        use_scaler = True
        print(f"Scaler Mean: {scaler_mean}")
        print(f"Scaler Scale: {scaler_scale}")

    print("\n--- Model Prediction Test ---")
    for i, state_original in enumerate(test_states):
        print(f"\nTest Case {i+1}:")
        print(f"  Original Input State: {state_original}")

        state_np = np.array(state_original, dtype=np.float32).reshape(1, -1)

        if use_scaler:
            if state_np.shape[1] != len(scaler_mean):
                print(f"  Error: Input state dimension {state_np.shape[1]} does not match scaler dimension {len(scaler_mean)}.")
                continue
            state_scaled_np = (state_np - scaler_mean) / scaler_scale
            print(f"  Scaled Input State:   {state_scaled_np.flatten().tolist()}")
        else:
            state_scaled_np = state_np

        state_tensor = torch.tensor(state_scaled_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            raw_reward = model(state_tensor).item()

        final_scaled_reward = 8.0 * np.tanh(raw_reward / 2.0)

        print(f"  Raw Reward (from model output): {raw_reward:.4f}")
        print(f"  Final Scaled Reward (tanh & 8x): {final_scaled_reward:.4f}")

if __name__ == "__main__":
    MODEL_FILENAME = "architectural_reward_model_v2_aug_symmetry_20250528_100822.pt"

    sample_test_states = [
        [0.65, 4.5, 95000, 0.8],
        [0.50, 3.0, 70000, 0.7],
        [0.72, 4.5, 80000, 0.8], # BCR > 0.7
        [0.60, 1.8, 80000, 0.8], # FAR < 2.0
        [0.40, 2.5, 50000, 0.6],
        [0.0, 0.0, 0.0, 0.0],     # Zero state
        [0.60, 3.5, 150000, 0.85], # 높은 Sunlight, 높은 SV_Ratio
        [0.55, 4.0, 60000, 0.75],  # 일반적인 값
    ]

    test_specific_reward_model(MODEL_FILENAME, sample_test_states)