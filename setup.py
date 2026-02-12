
from setuptools import setup, find_packages

setup(
    name='latentgee',
    version='0.1.0',
    description='Unsupervised Batch Correction with LatentGEE-U',
    author='Seungrin Yang',
    author_email='slyang@kribb.re.kr',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'pyyaml',
        'optuna',
        'hdbscan'
    ],
    python_requires='>=3.8',
)