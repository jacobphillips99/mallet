from setuptools import setup, find_packages

setup(
    name="vlm_autoeval_robot_benchmark",
    version="0.1.0",
    description="API-driven VLM server for AutoEval robot benchmarking",
    author="Jacob Phillips",
    author_email="jacob.phillips8905@gmail.com",
    packages=find_packages(),
    install_requires=[
        "litellm",
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "aiohttp",
        "pyyaml",
    ],
    entry_points={
        'console_scripts': [
            'vlm-autoeval-server=vlm_autoeval_robot_benchmark.__main__:main',
        ],
    },
    python_requires=">=3.8",
) 