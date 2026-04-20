import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loralib",
    version="0.1.3",
    author="Pu Wang",
    author_email="pu.wang@esat.kuleuven.be",
    description=(
        "Extended LoRA library based on Microsoft LoRA, with additional "
        "parameter-efficient fine-tuning backends including SSVD, SSVD-O, "
        "DoRA, SVFT, PiSSA, and related LoRA variants."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangpuup/LoRA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
