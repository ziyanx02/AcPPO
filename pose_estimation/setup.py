from setuptools import setup, find_packages

setup(
    name="robot_display",
    version="0.1.0",
    description="A Python package for displaying robot.",
    author="Ziyan Xiong",
    author_email="ziyanx02@gmail.com",
    url="https://github.com/ziyanx02",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "genesis-world>=0.2.0",
    ],
    include_package_data=True,
)