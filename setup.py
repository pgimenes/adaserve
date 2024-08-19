from setuptools import setup, find_packages

requirements = [
    "torch",
    "transformers",
]

setup(
    name="ada",
    version="1.0.0",
    description="Adaptive Inference Serving Engine for Generative Language Models",
    author="Pedro Gimenes",
    author_email="pedro.gimenes19@imperial.ac.uk",
    license_files=("LICENSE",),
    python_requires=">=3.10.2",
    package_dir={
        "": "src",
    },
    packages=find_packages("src"),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ada=ada.main:main",
        ],
    },
)
