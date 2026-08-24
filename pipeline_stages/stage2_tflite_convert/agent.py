import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from pydantic import BaseModel

class BenchmarkResult(BaseModel):
    target_board: str
    precision: str
    estimated_latency_ms: float
    estimated_mse: float
    memory_footprint_kb: float
    status: str

class DatasetDrivenTFLiteConvertAgent:
    def __init__(self, seq_len: int = 24, d_model: int = 5):
        self.seq_len = seq_len
        self.d_model = d_model
        self.batch_size = 1

    def load_from_binary_folder(self, folder_path: str, num_samples: int = 1000):
        """
        [바이너리 디렉토리 복원 공정]
        지정된 폴더 내의 개별 바이너리 파일들을 바이너리 역직렬화하여 
        정확히 [Seq_Len, D_Model] 형상의 32비트 부동소수점 시계열 텐서 배열로 정밀 복원합니다.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"⚠️ 시스템 오류: 입력하신 경로 [{folder_path}]가 존재하지 않습니다.")

        # 폴더 내 모든 파일 리스트 추출 (서브 디렉토리 제외한 순수 파일들)
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if os.path.isfile(os.path.join(folder_path, f))]
        
        if not all_files:
            raise ValueError(f"⚠️ 오류: 해당 폴더 [{folder_path}] 내부에 바이너리 데이터가 존재하지 않습니다.")

        # 대표성 확보를 위한 최대 1,000개 무작위 파일 샘플링 규칙 바인딩
        sample_size = min(num_samples, len(all_files))
        selected_files = np.random.choice(all_files, size=sample_size, replace=False)

        recovered_windows = []
        expected_bytes = self.seq_len * self.d_model * 4  # float32 = 4 bytes

        for file_path in selected_files:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
                
            # 바이트 스트림 크기가 정적 차원 규격과 완벽히 일치하는지 정밀 검증
            if len(raw_bytes) != expected_bytes:
                continue  # 규격 불일치 파손 파일 안전 스킵 방어선
                
            # 부호화 바이트 배열을 float32 넘파이 배열로 복원 후 원본 형상 재매핑
            window_data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(self.seq_len, self.d_model)
            recovered_windows.append(window_data)

        if not recovered_windows:
            raise ValueError("⚠️ 오류: 폴더 내 파일들이 요구되는 온디바이스 입력 차원 규격과 맞지 않습니다.")

        X_all = np.stack(recovered_windows)
        
        # TFLite 컴파일러 인터페이스 전용 정적 Generator 클로저 빌드
        def representative_dataset_gen():
            for i in range(len(X_all)):
                sample = np.expand_dims(X_all[i], axis=0)
                yield [sample]

        return representative_dataset_gen, X_all

    def run_benchmarking(self, target_board: str, precision: str, base_model_mse: float = 0.0125) -> BenchmarkResult:
        """공학적 트레이드오프 수식 연산 가동 (기존 가상 벤치마크 사양 100% 보존)"""
        is_stm = (target_board == "STM32H7 (Cortex-M7 MCU)")
        
        if precision == "FP32 (No Quantization)":
            latency_factor, error_multiplier, memory_kb = (85.0 if is_stm else 12.5), 1.0, 142.0
            status = "WARNING (메모리 부족 위험)" if is_stm else "PASS"
        elif precision == "FP16 (Half Precision)":
            latency_factor, error_multiplier, memory_kb = (46.7 if is_stm else 6.8), 1.02, 71.0
            status = "PASS"
        elif precision == "INT8 (Full Integer Quantization)":
            latency_factor, error_multiplier, memory_kb = (15.3 if is_stm else 3.1), 1.25, 35.5
            status = "PASS"
        else:
            latency_factor, error_multiplier, memory_kb = (12.5, 1.0, 142.0)
            status = "PASS"

        return BenchmarkResult(
            target_board=target_board,
            precision=precision,
            estimated_latency_ms=round(latency_factor, 2),
            estimated_mse=round(base_model_mse * error_multiplier, 5),
            memory_footprint_kb=memory_kb,
            status=status
        )

    def convert_model(self, instantiated_model, precision: str, rep_gen=None, output_dir: str = "outputs") -> str:
        """Keras/TF 정적 추론 그래프 고정 및 TFLite 물리 바이너리 빌드"""
        os.makedirs(output_dir, exist_ok=True)

        @tf.function(input_signature=[tf.TensorSpec([self.batch_size, self.seq_len, self.d_model], tf.float32)])
        def run_model(x):
            return instantiated_model(x)

        concrete_func = run_model.get_concrete_function()
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])

        file_suffix = "fp32"
        if precision == "FP16 (Half Precision)":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            file_suffix = "fp16"
        elif precision == "INT8 (Full Integer Quantization)":
            if rep_gen is None:
                raise ValueError("⚠️ INT8 양자화를 위해서는 대표성 데이터셋(Calibration Dataset) 지포트가 필수입니다.")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_operations = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            file_suffix = "int8"

        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, f"deployed_model_{file_suffix}.tflite")
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        return tflite_path

    def verify_tflite_io(self, tflite_path: str) -> dict:
        """변환 완료된 TFLite 파일의 물리 I/O 텐서 차원 유효성 검증"""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return {
            "input_shape": input_details['shape'].tolist(),
            "output_shape": output_details['shape'].tolist(),
            "file_size_kb": round(os.path.getsize(tflite_path) / 1024.0, 2)
        }
