from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


class DataPreparationAgent:
    def __init__(self) -> None:
        self.stats: Dict[str, object] = {}
        self.preprocessing_log: list[str] = []

    def load_and_analyze(self, csv_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
        df = pd.read_csv(csv_path)
        stats: Dict[str, object] = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "dtypes": {key: str(value) for key, value in df.dtypes.to_dict().items()},
            "missing_values": {key: int(value) for key, value in df.isnull().sum().to_dict().items()},
            "missing_percentage": {key: float(value) for key, value in (df.isnull().sum() / max(len(df), 1) * 100).to_dict().items()},
            "duplicates": int(df.duplicated().sum()),
        }
        self.stats = stats
        return df, stats

    def preprocess_data(
        self,
        df: pd.DataFrame,
        *,
        remove_na: bool = True,
        normalize: bool = True,
        remove_duplicates: bool = True,
    ) -> pd.DataFrame:
        processed = df.copy()
        self.preprocessing_log = []
        original_rows = len(processed)

        if remove_duplicates:
            duplicate_count = int(processed.duplicated().sum())
            if duplicate_count:
                processed = processed.drop_duplicates()
                self.preprocessing_log.append(f"중복 행 {duplicate_count}개 제거")

        if remove_na:
            missing_count = int(processed.isnull().sum().sum())
            if missing_count:
                rows_before = len(processed)
                processed = processed.dropna()
                self.preprocessing_log.append(f"결측치 포함 행 {rows_before - len(processed)}개 제거")

        if normalize:
            numeric_columns = processed.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                min_value = processed[column].min()
                max_value = processed[column].max()
                if max_value - min_value > 0:
                    processed[column] = (processed[column] - min_value) / (max_value - min_value)
            if len(numeric_columns) > 0:
                self.preprocessing_log.append(f"수치형 컬럼 {len(numeric_columns)}개 Min-Max 정규화")

        self.preprocessing_log.append(f"전처리 완료: {original_rows}행 → {len(processed)}행")
        return processed

    def generate_visualizations(self, df: pd.DataFrame) -> Dict[str, bytes]:
        visualizations: Dict[str, bytes] = {}

        fig, ax = plt.subplots(figsize=(10, 4))
        missing_series = df.isnull().sum()
        if int(missing_series.sum()) > 0:
            missing_series[missing_series > 0].plot(kind="bar", ax=ax, color="coral")
            ax.set_ylabel("결측치 개수")
        else:
            ax.text(0.5, 0.5, "결측치 없음", ha="center", va="center", fontsize=14)
        ax.set_title("결측치 분포")
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        visualizations["missing_values"] = buffer.read()
        plt.close(fig)

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            fig, axes = plt.subplots(min(3, len(numeric_columns)), 1, figsize=(10, 3 * min(3, len(numeric_columns))))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            for index, column in enumerate(numeric_columns[:3]):
                ax = axes[index]
                df[column].hist(bins=30, ax=ax, color="skyblue", edgecolor="black")
                ax.set_title(f"{column} 분포")
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            visualizations["distribution"] = buffer.read()
            plt.close(fig)

        return visualizations

    def save_processed_data(self, df: pd.DataFrame, output_path: str | Path) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)

    def get_preprocessing_summary(self) -> str:
        return "\n".join(self.preprocessing_log)
