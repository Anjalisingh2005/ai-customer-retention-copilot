"""Submit churn training as a SageMaker job.

We package the project's train script as a SageMaker ScriptProcessor / Estimator
entry point. Artifacts (model + preprocessor + FAISS) are uploaded to S3 so the
ECS-deployed API can pull them at boot.

Run locally:
    AWS_PROFILE=... python deployment/aws/sagemaker_training.py
"""
from __future__ import annotations

import os
import sys

try:
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn
except ImportError:
    print("Install the optional deps: pip install boto3 sagemaker", file=sys.stderr)
    sys.exit(1)


REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "churn-copilot-artifacts")
INSTANCE_TYPE = os.environ.get("SAGEMAKER_INSTANCE", "ml.m5.xlarge")


def main() -> None:
    if not ROLE:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN env var to a SageMaker execution role ARN.")

    session = sagemaker.Session(default_bucket=S3_BUCKET)
    estimator = SKLearn(
        entry_point="scripts/train_model.py",
        source_dir=".",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={"upload_to_s3": "1"},
        output_path=f"s3://{S3_BUCKET}/output",
        base_job_name="churn-copilot-train",
    )
    estimator.fit()
    print("Training job submitted. Model artifacts will land at:", estimator.model_data)


if __name__ == "__main__":
    main()
