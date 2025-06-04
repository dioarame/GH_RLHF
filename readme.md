# GH_RLHF (Grasshopper Reinforcement Learning from Human Feedback)

이 프로젝트는 Grasshopper와 Python 간의 통신을 기반으로 강화학습에서 인간 피드백(RLHF, Reinforcement Learning from Human Feedback)을 구현하는 시스템입니다. 건축 및 디자인 분야에서 파라메트릭 모델링과 인공지능을 결합하여 더 나은 설계 결과를 도출합니다.
또한, 이 프로젝트는 2025년 성균관대학교 공과대학 글로벌스마트시티 융합전공 및 DesignInformaticsGroup(DIG) 소속 양승원 박사과정의 학위논문의 주요 연구 개발의 결과물입니다.

## 프로젝트 개요

GH_RLHF는 파라메트릭 디자인 도구인 Grasshopper와 강화학습을 결합하여 인간의 피드백을 통해 더 나은 설계 결과를 도출하는 프레임워크입니다. 이 시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **Grasshopper 컴포넌트**: C# 스크립트로 작성된 Grasshopper 컴포넌트
2. **서버**: 인간 피드백 데이터를 수집하기 위한 웹 기반 GUI
3. **강화학습 모델**: Python으로 구현된 보상 모델 및 학습 알고리즘

## 시스템 아키텍처 및 파이프라인

GH_RLHF 파이프라인은 다음과 같은 순환적 과정으로 구성됩니다:

```
+------------------------+       +------------------------+       +------------------------+
|     Grasshopper        | ----> |    Python RL 모듈       | ----> |        웹 GUI          |
|   (파라미터 생성)        |       |   (디자인 옵션 생성)     |       |   (디자인 시각화)       |
+------------------------+       +------------------------+       +------------------------+
          ^                                                                  |
          |                                                                  |
          |                                                                  v
+------------------------+       +------------------------+       +------------------------+
|    최적화된 디자인       | <---- |     정책 최적화         | <---- |      인간 피드백        |
|  (새로운 파라미터 적용)   |       |  (강화학습 알고리즘)     |       |  (평가 및 선호도 입력)   |
+------------------------+       +------------------------+       +------------------------+
                                          ^                                 |
                                          |                                 |
                                          |                                 v
                                 +------------------------+       +------------------------+
                                 |     보상 모델 학습       | <---- |   피드백 데이터 저장     |
                                 |  (인간 선호도 모델링)     |       |   (JSON 형식 저장)     |
                                 +------------------------+       +------------------------+
```

### 1. Grasshopper 컴포넌트

`Grasshopper Component` 폴더의 `RLHFComponent.cs`는 Grasshopper와 Python 사이의 통신을 담당하는 핵심 컴포넌트입니다. 이 컴포넌트는 다음과 같은 역할을 수행합니다:

- HTTP 요청을 통한 Python 서버와의 데이터 교환
- 디자인 파라미터 생성 및 전송
- 학습된 모델의 결과를 Grasshopper 환경에 시각화
- 설계 제약 조건 관리 및 적용

**컴포넌트 주요 기능**:
- `SendDesignParameters()`: 현재 디자인 파라미터를 Python 서버로 전송
- `ReceiveOptimizedParameters()`: 최적화된 파라미터를 서버로부터 수신
- `UpdateVisualization()`: 수신된 파라미터를 기반으로 Grasshopper 내 시각화 갱신

**컴포넌트 빌드 과정**:
1. Visual Studio 또는 다른 C# IDE에서 RLHFComponent.cs 컴파일
2. 생성된 DLL 파일을 Grasshopper용 GHA 파일로 변환
   ```
   "C:\Program Files\Rhino 7\Plug-ins\Grasshopper\GrasshopperDeveloperUtils\GhPython.exe" /package DLL을GHA로변환 DLL파일경로
   ```
3. 생성된 GHA 파일을 Grasshopper 컴포넌트 폴더(보통 `%AppData%\Grasshopper\Libraries`)에 배치

### 2. 웹 기반 GUI 서버

`server` 폴더에는 Node.js 기반의 웹 서버와 인간 피드백을 수집하기 위한 인터페이스가 포함되어 있습니다:

**서버 구성요소**:
- **Express.js 백엔드**: `app.js`에서 API 엔드포인트 및 라우팅 처리
- **웹소켓 통신**: 실시간 데이터 전송을 위한 Socket.IO 구현
- **정적 리소스**: HTML, CSS, JavaScript로 구현된 피드백 인터페이스

**주요 API 엔드포인트**:
- `/api/designs`: 디자인 옵션 데이터 제공
- `/api/feedback`: 사용자 피드백 수집
- `/api/sessions`: 세션 관리

사용자는 GUI를 통해 생성된 디자인 옵션들을 비교하고 평가할 수 있으며, 이 데이터는 JSON 형식으로 저장되어 보상 모델 학습에 활용됩니다.

### 3. Python 강화학습 모듈

`rl` 폴더의 Python 코드는 다음과 같은 주요 기능을 담당합니다:

**주요 구성 요소**:
- **보상 모델(`rl/models/reward_model.py`)**: 인간 피드백 데이터를 기반으로 설계의 품질 예측
- **정책 모델(`rl/models/policy_model.py`)**: 보상을 최대화하는 디자인 파라미터 생성
- **학습 알고리즘(`rl/train.py`)**: PPO(Proximal Policy Optimization) 기반 정책 최적화
- **유틸리티 함수(`rl/utils/`)**: 데이터 처리 및 통신을 위한 보조 기능

**보상 모델 아키텍처**:
- 다층 퍼셉트론(MLP) 네트워크 사용
- 디자인 파라미터를 입력으로 받아 품질 점수 예측
- 인간 피드백 데이터로 지도 학습

**정책 모델 구현**:
- 가우시안 정책 네트워크 사용
- 현재 상태를 입력으로 받아 디자인 파라미터 분포 출력
- 강화학습을 통해 기대 보상 최대화

## 데이터 흐름 상세

RLHF 시스템의 데이터 흐름은 다음과 같은 단계로 이루어집니다:

1. **초기화 단계**:
   - Grasshopper에서 디자인 파라미터 공간 정의 및 제약 조건 설정
   - Python 서버 및 웹 GUI 서버 실행

2. **디자인 생성 단계**:
   - 정책 모델이 현재 상태에서 디자인 파라미터 샘플링
   - 샘플링된 파라미터를 Grasshopper로 전송하여 3D 모델 생성
   - 생성된 디자인의 시각적 표현(이미지, 메트릭스) 추출

3. **피드백 수집 단계**:
   - 웹 GUI를 통해 사용자에게 여러 디자인 옵션 제시
   - 사용자가 각 디자인에 대한 선호도 점수 및 코멘트 제공
   - 피드백 데이터가 서버에 저장되고 Python 모듈로 전송

4. **모델 학습 단계**:
   - 수집된 피드백을 기반으로 보상 모델 재학습
   - 업데이트된 보상 모델을 사용하여 정책 모델 최적화
   - 새로운 정책을 통한 향상된 디자인 파라미터 생성

5. **반복 및 개선 단계**:
   - 최적화된 파라미터로 새로운 디자인 생성
   - 사용자 피드백 재수집 및 모델 재학습
   - 설계 품질이 수렴될 때까지 과정 반복

## 데이터 형식 예시

### Grasshopper -> Python 데이터 형식
```json
{
  "design_parameters": [0.5, 0.3, 0.7, 0.2, 0.6],
  "constraints": {
    "max_height": 10.0,
    "min_width": 5.0,
    "structural_limit": 15.0
  },
  "session_id": "design_session_001",
  "iteration": 5
}
```

### Python -> GUI 데이터 형식
```json
{
  "design_options": [
    {
      "id": "design_001",
      "preview_url": "http://localhost:8000/previews/design_001.png",
      "parameters": [0.5, 0.3, 0.7, 0.2, 0.6],
      "metrics": {
        "height": 8.7,
        "width": 6.2,
        "material_usage": 120.5
      }
    },
    {
      "id": "design_002",
      "preview_url": "http://localhost:8000/previews/design_002.png",
      "parameters": [0.4, 0.6, 0.5, 0.3, 0.5],
      "metrics": {
        "height": 7.5,
        "width": 7.1,
        "material_usage": 115.2
      }
    }
  ],
  "session_id": "design_session_001",
  "iteration": 5
}
```

### GUI -> Python 피드백 데이터 형식
```json
{
  "feedback": [
    {
      "design_id": "design_001",
      "score": 7.5,
      "preference_rank": 1,
      "comments": "좋은 비율이지만 높이가 약간 낮음",
      "aspects": {
        "aesthetics": 8,
        "functionality": 7,
        "innovation": 6
      }
    },
    {
      "design_id": "design_002",
      "score": 5.2,
      "preference_rank": 2,
      "comments": "균형이 맞지 않음",
      "aspects": {
        "aesthetics": 5,
        "functionality": 6,
        "innovation": 4
      }
    }
  ],
  "user_id": "user_123",
  "session_id": "design_session_001",
  "timestamp": "2025-06-04T23:15:32Z"
}
```

### Python -> Grasshopper 최적화 결과 데이터 형식
```json
{
  "optimized_parameters": [0.55, 0.35, 0.65, 0.25, 0.58],
  "expected_reward": 8.2,
  "confidence": 0.85,
  "session_id": "design_session_001",
  "iteration": 6
}
```

## 강화학습 및 보상 모델 세부 사항

### 보상 모델 학습 과정

1. **데이터 전처리**:
   - 수집된 피드백 데이터를 정규화하고 증강
   - 페어와이즈(pairwise) 비교 데이터 생성
   - 훈련/검증 데이터셋 분할

2. **모델 아키텍처**:
   - 입력층: 디자인 파라미터 및 메트릭스 (차원: 파라미터 수 + 메트릭스 수)
   - 은닉층: 3개의 완전 연결 계층 (256, 128, 64 유닛)
   - 출력층: 단일 스칼라 값 (예측된 보상 점수)
   - 활성화 함수: ReLU (은닉층), 선형 (출력층)

3. **학습 목표**:
   - 인간 선호도와 일치하는 보상 함수 학습
   - Mean Squared Error(MSE) 손실 최소화
   - Adam 옵티마이저 사용, 학습률: 0.001

### 정책 최적화 과정

1. **PPO 알고리즘 구현**:
   - 행동 공간: 연속적인 디자인 파라미터
   - 상태 공간: 현재 디자인 상태 및 제약 조건
   - 클리핑 파라미터(ε): 0.2
   - 할인 계수(γ): 0.99

2. **정책 네트워크 아키텍처**:
   - 액터-크리틱 구조 사용
   - 액터(정책) 네트워크: 3개의 완전 연결 계층 (128, 64, 32 유닛)
   - 크리틱(가치) 네트워크: 3개의 완전 연결 계층 (128, 64, 32 유닛)
   - 출력: 파라미터 분포의 평균 및 표준편차

3. **최적화 과정**:
   - 학습 배치 크기: 64
   - 에포크 수: 10 (배치당)
   - 학습률: 3e-4 (선형 감소 스케줄)
   - 엔트로피 계수: 0.01 (탐색 장려)

## 설치 및 실행 방법

### 사전 요구 사항

- **소프트웨어**:
  - Rhinoceros 7 이상
  - Grasshopper
  - Python 3.8 이상
  - Node.js 14 이상
  - npm 또는 yarn

- **Python 라이브러리**:
  - PyTorch 1.8 이상
  - NumPy, SciPy, Pandas
  - Flask (API 서버용)
  - Matplotlib (시각화용)

### 의존성 설치

```bash
# Python 의존성 설치
pip install -r requirements.txt

# 서버 의존성 설치
cd server
npm install
```

### Grasshopper 컴포넌트 설치

1. `Grasshopper Component` 폴더의 C# 스크립트를 컴파일:
   ```bash
   # Visual Studio를 사용하는 경우
   msbuild RLHFComponent.csproj /p:Configuration=Release
   
   # 또는 .NET CLI 사용
   dotnet build RLHFComponent.csproj -c Release
   ```

2. DLL을 GHA 파일로 변환:
   ```bash
   "C:\Program Files\Rhino 7\Plug-ins\Grasshopper\GrasshopperDeveloperUtils\GhPython.exe" /package RLHFComponent bin\Release\RLHFComponent.dll
   ```

3. GHA 파일을 Grasshopper 컴포넌트 폴더에 복사:
   ```bash
   copy bin\Release\RLHFComponent.gha "%AppData%\Grasshopper\Libraries\"
   ```

### 서버 실행

```bash
# 웹 GUI 서버 실행
cd server
npm start

# 별도 터미널에서 Python API 서버 실행
cd rl
python api_server.py --port 5000
```

### 강화학습 모듈 실행

```bash
# 보상 모델 학습
python rl/train_reward_model.py --data_path data/feedback --model_save_path models/reward

# 정책 최적화
python rl/train_policy.py --reward_model models/reward/latest.pt --save_path models/policy
```

## 파이프라인 실행 예시

1. **Grasshopper 설정**:
   - Rhino 및 Grasshopper 실행
   - RLHF 컴포넌트를 캔버스에 추가
   - 디자인 파라미터 입력 및 제약 조건 설정
   - 컴포넌트 연결 및 초기화

2. **서버 시작**:
   - 웹 GUI 서버 실행 (`npm start`)
   - Python API 서버 실행 (`python api_server.py`)

3. **초기 디자인 생성**:
   - Grasshopper에서 "Generate Designs" 버튼 클릭
   - 초기 파라미터로 디자인 샘플 생성
   - 생성된 디자인이 웹 GUI에 표시됨

4. **피드백 수집**:
   - 사용자가 웹 브라우저에서 GUI 접속 (기본: http://localhost:8000)
   - 제시된 디자인 옵션에 대해 평가 및 피드백 제공
   - "Submit Feedback" 버튼 클릭하여 데이터 전송

5. **모델 학습 및 최적화**:
   - 수집된 피드백으로 보상 모델 학습
   - 학습된 보상 모델로 정책 최적화
   - 최적화된 파라미터 생성

6. **결과 적용**:
   - 최적화된 파라미터가 Grasshopper로 전송됨
   - 새로운 디자인이 Grasshopper에서 자동 생성
   - 결과 시각화 및 평가

7. **반복 과정**:
   - 필요에 따라 과정 반복 (단계 3-6)
   - 더 많은 피드백 수집으로 모델 정확도 향상
   - 디자인 품질 점진적 개선

## 주요 파일 및 폴더 구조

```
GH_RLHF/
│
├── Grasshopper Component/ - C# 스크립트 및 Grasshopper 컴포넌트
│   ├── RLHFComponent.cs - 메인 컴포넌트 소스 코드
│   ├── RLHFComponent.csproj - 프로젝트 파일
│   └── bin/ - 빌드된 DLL 및 GHA 파일
│
├── server/ - 인간 피드백 수집을 위한 웹 서버
│   ├── public/ - 정적 파일 (HTML, CSS, 이미지)
│   │   ├── index.html - 메인 인터페이스
│   │   ├── css/ - 스타일시트
│   │   └── js/ - 프론트엔드 스크립트
│   ├── src/ - 서버 소스 코드
│   │   ├── app.js - Express 서버 설정
│   │   ├── routes/ - API 라우터
│   │   └── controllers/ - 요청 처리 로직
│   └── package.json - 의존성 정보
│
├── rl/ - 강화학습 모듈
│   ├── models/ - 보상 모델 및 정책 모델
│   │   ├── reward_model.py - 보상 모델 클래스
│   │   └── policy_model.py - 정책 모델 클래스
│   ├── utils/ - 유틸리티 함수
│   │   ├── data_utils.py - 데이터 처리 함수
│   │   └── visualization.py - 결과 시각화
│   ├── train.py - 모델 학습 스크립트
│   ├── api_server.py - Python API 서버
│   └── config.py - 설정 파일
│
├── data/ - 학습 데이터 및 결과
│   ├── feedback/ - 수집된 인간 피드백 데이터
│   │   └── *.json - 세션별 피드백 데이터
│   └── models/ - 학습된 모델 저장소
│       ├── reward/ - 보상 모델 체크포인트
│       └── policy/ - 정책 모델 체크포인트
│
└── configs/ - 구성 파일
    ├── rl_config.json - 강화학습 설정
    ├── model_config.json - 모델 아키텍처 설정
    └── server_config.json - 서버 설정
```

## 문제 해결 및 팁

- **Grasshopper 연결 오류**: 방화벽 설정 확인 및 포트 허용
- **모델 수렴 문제**: 학습률 조정 및 배치 크기 최적화
- **메모리 사용량**: 대규모 모델 학습 시 GPU 메모리 관리
- **실시간 성능**: 복잡한 디자인의 경우 평가 시간 고려

## 확장 및 개선 방향

- **다양한 보상 모델**: 여러 측면(미학, 기능성, 구조적 안정성 등)을 고려한 다중 보상 모델
- **메타 학습**: 새로운 디자인 문제에 빠르게 적응하기 위한 메타 학습 구현
- **하이브리드 접근법**: 규칙 기반 시스템과 학습 기반 시스템의 결합
- **다중 사용자 피드백**: 여러 전문가의 의견을 통합하는 앙상블 방법

## 참고 사항

- 이 프로젝트는 Rhinoceros 7 및 Grasshopper와 호환됩니다.
- Python 3.8 이상이 필요합니다.
- 웹 서버 실행을 위해 Node.js 14 이상이 필요합니다.
- 복잡한 3D 모델 생성 및 평가에는 고성능 하드웨어가 권장됩니다.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.
