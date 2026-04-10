from setuptools import setup, find_packages

setup(
    name="AutoClean",                   # 包名（随便取，但最好和项目一致）
    version="0.1.0",                  # 版本号
    description="Time series cleaning and anomaly detection toolkit",
    author="Your Name",
    packages=find_packages(),         # 自动找到所有含 __init__.py 的包
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],                                # 依赖，可以按需补充
    python_requires=">=3.8",          # 兼容的 Python 版本
)
