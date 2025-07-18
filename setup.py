from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="goquant-sentiment-trader",
    version="0.1.0",
    author="GoQuant Team",
    author_email="team@goquant.com",
    description="Comprehensive sentiment analysis and trade signal generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goquant/sentiment-trader",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "prod": [
            "redis>=4.6.0",
            "psycopg2-binary>=2.9.0",
            "gunicorn>=21.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "goquant-sentiment=goquant.cli:main",
        ],
    },
)
