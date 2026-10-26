# On-Device AI Flow

ODAF는 고정 스크립트 모음이 아니라, Google 무료 API Key 기반 AI Agent가 각 stage의 목적을 해석하고 필요한 실행·분석·재시도·최소 수정을 수행하는 Streamlit 워크벤치입니다.

## Pipeline

1. `Stage 1 - Model Design`
	- 데이터 준비
	- 모델 코드 분석/리팩터링
	- 변경 diff 및 원본/수정본 아티팩트 보관
2. `Stage 2 - TFLite Convert`
	- SavedModel 생성
	- FP32/INT8 TFLite 변환
	- Calibration Data 생성
	- LNE 변환용 raw input / reference output 생성
	- 변환 실패 분석 및 최소 수정안 제안
3. `Stage 3 - LNE Convert + SPIKE Validation`
	- Windows Local ODAF → Linux VM Worker 연동
	- SSH/scp 기반 원격 LNE DDK compile
	- SPIKE validation
	- log parse / validation / AI root-cause analysis
4. `Stage 4 - Production Simulation`
	- Test Case
	- Case Study
	- Simulation
	- Visualization
5. `Stage 5 - Executive Sign`
	- Stage 1~4 결과 종합
	- 최종 승인 리포트 생성

## Directory Structure

```text
pipeline_stages/
  stage1_model_design/
  stage2_tflite_convert/
  stage3_lne_convert/
  stage4_production_simulation/
  stage5_executive_sign/
```

## Outputs

각 stage는 `outputs/<stage_slug>/` 하위에 아티팩트를 저장합니다.

- `outputs/stage1_model_design/`
- `outputs/stage2_tflite_convert/`
- `outputs/stage3_lne_convert/`
- `outputs/stage4_production_simulation/`
- `outputs/stage5_executive_sign/`

## Windows Local + Linux VM Worker

- **Windows Local ODAF**: Stage 1/2 산출물 생성, Streamlit UI, Stage 3 orchestration
- **Linux VM Worker**: LNE DDK compile, SPIKE 실행, 원격 로그 생성
- **Transport**: `ssh`, `scp`
- **Config Policy**: 원격 host/path/command는 사용자가 입력하거나 실제 환경값에서만 로드하며, 코드에서 임의 생성하지 않습니다.

## Run

```powershell
python -m venv .odaf_env
.\.odaf_env\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run main.py
```

## Notes

- `GEMINI_API_KEY`가 있으면 Google API 기반 분석을 사용합니다.
- API가 없거나 실패하면 로컬 규칙 기반 fallback으로 동작합니다.
- Stage 3 원격 VM 설정은 UI 또는 `.env` 기반 기본값으로 저장해서 사용할 수 있습니다.

## Architecture

ODAF 흐름:

- Stage 1: AI Model Development
- Stage 2: TFLite Conversion
- Stage 3: LNE Conversion + SPIKE Validation
- Stage 4: Production Simulation
- Stage 5: Result Summary / Executive Sign

## Runtime Architecture

```text
Windows ODAF
    |
    | SSH / SCP
    v
Linux LNE Worker VM
    |
    +-- LNE DDK
    +-- DQC Build
    +-- SPIKE Validation
```

## LNE Worker Setup

상세 환경 구성 방법은 다음 문서를 참조하십시오.

- `vm_setup/README.md`
- `vm_setup/setup_lne_vm.sh`
- `vm_setup/verify_lne_vm.sh`

## Security

Repository에는 다음을 포함하지 않습니다.

- `.env`
- API Key
- Password
- SSH Private Key
- LNE DDK Binary / Toolchain
- Runtime Output

실제 Secret 값은 로컬 환경 또는 사용자별 `.env` 파일에만 보관하고 Git에는 기록하지 않습니다.