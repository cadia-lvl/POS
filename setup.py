"""The POS package."""
from setuptools import setup, find_packages

setup(
    name="pos",
    version="0.0.1",
    description="A POS tagger for Icelandic",
    author="Haukur Páll Jónsson",
    url="https://github.com/cadia-lvl/pos",
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=["click==7.1.2"],
    extras_require={"dev": ["black", "pydocstyle", "pylint", "mypy", "pytest"]},
    entry_points="""
        [console_scripts]
        pos=pos.main:cli
    """,
)

