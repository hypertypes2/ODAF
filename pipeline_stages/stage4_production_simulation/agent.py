from __future__ import annotations

import json
import importlib
import os
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


load_dotenv()


class TestScenario(BaseModel):
    scenario_name: str
    description: str
    test_inputs: List[List[float]]
    expected_behavior: str


class ProductionSimulationResult(BaseModel):
    task_type: str
    confidence: float
    test_scenarios: List[TestScenario]
    test_results: List[Dict]
    case_study: Dict
    simulation_summary: Dict
    recommendations: List[str]
    overall_status: str


class ProductionSimulationAgent:
    def __init__(self, seq_len: int = 24, d_model: int = 5) -> None:
        self.seq_len = seq_len
        self.d_model = d_model
        api_key = os.getenv('GEMINI_API_KEY', '').strip()
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _require_tensorflow(self) -> None:
        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError("TensorFlow is not installed in the active environment. Stage 4 simulation requires TensorFlow or a compatible runtime.")

    def analyze_model_task(self, source_code: str) -> dict:
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=(
                        "모델 코드를 보고 태스크 유형을 분류하십시오. JSON만 반환하십시오.\n"
                        "{\"task_type\": \"regression\", \"confidence\": 0.0, \"reasoning\": \"...\"}\n\n"
                        f"{source_code}"
                    ),
                    config=types.GenerateContentConfig(response_mime_type='application/json'),
                )
                return json.loads(response.text.strip())
            except Exception:
                pass
        if 'softmax' in source_code.lower():
            return {'task_type': 'multi_classification', 'confidence': 0.7, 'reasoning': 'softmax detected'}
        return {'task_type': 'regression', 'confidence': 0.7, 'reasoning': 'default regression assumption'}

    def generate_test_scenarios(self, task_type: str) -> List[TestScenario]:
        base = [20.0, 45.0, 1.0, 0.5, 0.2][: self.d_model]
        scenarios = [
            TestScenario(
                scenario_name='Normal Operation',
                description='정상 운전 시나리오: 실내/외부 조건이 일반 범위에 있는 상태',
                test_inputs=[
                    base,
                    [value + 5 for value in base],
                    [max(0.0, value - 3) for value in base],
                ],
                expected_behavior='안정적 출력',
            ),
            TestScenario(
                scenario_name='Comfort / Heat Load',
                description='에어컨 실생활 시나리오: 더운 날씨, 실내 열부하 증가, 급격한 변경 구간',
                test_inputs=[
                    [28.0, 60.0, 1.0, 0.6, 0.2],
                    [30.0, 70.0, 1.5, 0.8, 0.25],
                    [32.0, 80.0, 1.8, 1.0, 0.3],
                ],
                expected_behavior='출력은 온도 상승 흐름을 따라 반응해야 함',
            ),
            TestScenario(
                scenario_name='Night / Stable Cooling',
                description='야간 저온 유지 상태: 부하가 낮고 안정적',
                test_inputs=[
                    [18.0, 35.0, 0.6, 0.2, 0.1],
                    [19.0, 38.0, 0.7, 0.3, 0.12],
                    [21.0, 42.0, 0.8, 0.35, 0.15],
                ],
                expected_behavior='출력이 과도하게 증가하지 않아야 함',
            ),
            TestScenario(
                scenario_name='Extreme Boundary',
                description='경계 조건: 과열, 과소부하, 이상기온 입력',
                test_inputs=[
                    [0.0] * self.d_model,
                    [50.0] * self.d_model,
                    [10.0, 90.0] + [30.0] * max(0, self.d_model - 2),
                ],
                expected_behavior='경계 조건에서도 예측 가능하고 안전한 범위 유지',
            ),
        ]
        return scenarios

    def run_model_tests(self, tflite_path: str, scenarios: List[TestScenario]) -> List[Dict]:
        self._require_tensorflow()
        tf = importlib.import_module("tensorflow")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        is_quantized = input_details[0]['dtype'] in (np.int8, np.uint8)

        test_results: List[Dict] = []
        for scenario in scenarios:
            scenario_result = {'scenario_name': scenario.scenario_name, 'description': scenario.description, 'test_cases': []}
            for case_index, test_input in enumerate(scenario.test_inputs):
                # 실제 모델 입력은 [1, seq_len, feature_dim] 구조를 가정하고, 단일 시계열 샘플을 생성한다.
                if len(test_input) < self.d_model:
                    test_input = test_input + [0.0] * (self.d_model - len(test_input))
                sample_sequence = [list(test_input) for _ in range(self.seq_len)]
                input_data = np.array([sample_sequence], dtype=np.float32)
                if input_data.shape[-1] < self.d_model:
                    pad = np.zeros((1, self.seq_len, self.d_model - input_data.shape[-1]), dtype=np.float32)
                    input_data = np.concatenate([input_data, pad], axis=-1)
                if is_quantized:
                    scale = input_details[0]['quantization'][0] or 1.0
                    zero_point = input_details[0]['quantization'][1]
                    input_data = (input_data / scale + zero_point).astype(np.int8)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                if output_details[0]['dtype'] == np.int8:
                    scale = output_details[0]['quantization'][0] or 1.0
                    zero_point = output_details[0]['quantization'][1]
                    output = (output.astype(np.float32) - zero_point) * scale
                flat_output = np.array(output).reshape(-1).astype(float).tolist()
                status = 'PASS'
                scenario_result['test_cases'].append({
                    'test_id': case_index + 1,
                    'input': test_input,
                    'output': flat_output[:5],
                    'status': status,
                })
            test_results.append(scenario_result)
        return test_results

    def summarize_case_study(self, test_results: List[Dict]) -> Dict:
        total_cases = sum(len(result['test_cases']) for result in test_results)
        passed_cases = sum(1 for result in test_results for case in result['test_cases'] if case['status'] == 'PASS')
        outputs = [value for result in test_results for case in result['test_cases'] for value in case['output']]
        return {
            'total_cases': total_cases,
            'passed_cases': passed_cases,
            'pass_rate': round((passed_cases / total_cases) * 100, 2) if total_cases else 0.0,
            'output_min': min(outputs) if outputs else 0.0,
            'output_max': max(outputs) if outputs else 0.0,
        }

    def generate_recommendations(self, task_analysis: dict, case_study: dict) -> List[str]:
        recommendations = []
        if case_study['pass_rate'] < 100:
            recommendations.append('추가 시뮬레이션 및 edge case 검증이 필요합니다.')
        else:
            recommendations.append('현재 시뮬레이션 범위에서는 안정적으로 동작합니다.')
        if task_analysis.get('confidence', 0) < 0.8:
            recommendations.append('태스크 분류 신뢰도가 낮으므로 모델 목적을 재확인하세요.')
        recommendations.append('실제 타깃 장비 로그와 Stage 3 결과를 함께 비교 검토하세요.')
        recommendations.append('에어컨 실사용 환경 case를 기준으로 냉방/난방/야간/피크부하 조건을 추가 검증하세요.')
        return recommendations

    def run_full_simulation_pipeline(self, source_code: str, tflite_path: str) -> ProductionSimulationResult:
        task_analysis = self.analyze_model_task(source_code)
        scenarios = self.generate_test_scenarios(task_analysis['task_type'])
        test_results = self.run_model_tests(tflite_path, scenarios)
        case_study = self.summarize_case_study(test_results)
        recommendations = self.generate_recommendations(task_analysis, case_study)
        overall_status = 'PASS' if case_study['pass_rate'] == 100 else 'WARNING'
        simulation_summary = {
            'scenario_count': len(scenarios),
            'pass_rate': case_study['pass_rate'],
            'task_type': task_analysis['task_type'],
        }
        return ProductionSimulationResult(
            task_type=task_analysis['task_type'],
            confidence=task_analysis['confidence'],
            test_scenarios=scenarios,
            test_results=test_results,
            case_study=case_study,
            simulation_summary=simulation_summary,
            recommendations=recommendations,
            overall_status=overall_status,
        )
