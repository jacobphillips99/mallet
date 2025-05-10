from setuptools import find_packages, setup

setup(
    name="mallet",
    version="0.1.0",
    description="Cloud-based tools and evaluations for VLMs to control real-world robots",
    author="Jacob Phillips",
    author_email="jacob.phillips8905@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
