# src/utils/custom_pipeline_builder.py
import os
from collections import OrderedDict
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


class CustomPipelineBuilder:
    def __init__(
        self,
        persist_path: str = "artifacts/final_pipeline.pkl",
        load_if_exists: bool = True,
    ):
        self.persist_path = Path(persist_path)
        self.steps = OrderedDict()

        if load_if_exists and self.persist_path.exists():
            print(f"[CustomPipelineBuilder] Loading from {self.persist_path}")
            self._load()
        else:
            print("[CustomPipelineBuilder] Initializing new pipeline.")

    def add_step(self, name: str, transformer):
        if name in self.steps:
            print(f"[CustomPipelineBuilder] Overwriting step: {name}")
        self.steps[name] = transformer
        return self  # chainable

    def build_pipeline(self) -> Pipeline:
        return Pipeline([(k, v) for k, v in self.steps.items()])

    def save(self):
        pipeline = self.build_pipeline()
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.persist_path)
        print(f"[CustomPipelineBuilder] Saved pipeline to {self.persist_path}")

    def _load(self):
        pipeline = joblib.load(self.persist_path)
        self.steps = OrderedDict(pipeline.steps)
        print(
            f"[CustomPipelineBuilder] Loaded pipeline with steps: {list(self.steps.keys())}"
        )

    def export_pipeline(self):
        return self.build_pipeline()

    # builder = CustomPipelineBuilder()
    # builder.add_step("encoder", encoder)
    # builder.save()


# final_pipeline = CustomPipelineBuilder().export_pipeline()
# mlflow.sklearn.log_model(final_pipeline, artifact_path="full_pipeline")
