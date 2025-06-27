"""
IMDB情感分析專案安裝設定
"""

from setuptools import setup, find_packages

setup(
    name="imdb-sentiment-analysis",
    version="1.0.0",
    author="統計碩士深度學習專案",
    description="IMDB電影評論情感分析深度學習系統",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
    ],
)