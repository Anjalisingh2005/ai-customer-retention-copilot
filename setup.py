from setuptools import find_packages, setup

setup(
    name="churn-copilot",
    version="0.1.0",
    description="AI-powered customer retention copilot: churn + SHAP + LangGraph + RAG + FastAPI.",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.10",
)
