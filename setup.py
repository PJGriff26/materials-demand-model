"""
Setup script for Materials Demand Model package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'Materials Demand Model - Monte Carlo simulation for energy infrastructure'

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
else:
    requirements = ['numpy>=1.24.0', 'pandas>=2.0.0', 'scipy>=1.11.0', 'matplotlib>=3.7.0']

setup(
    name='materials-demand-model',
    version='1.0.0',
    author='Materials Demand Research Team',
    author_email='your-email@institution.edu',
    description='Research-grade Monte Carlo simulation for energy infrastructure materials demand',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/materials_demand_model',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'sensitivity': [
            'SALib>=1.4.0',
        ],
        'sampling': [
            'pyDOE2>=1.3.0',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['*.py'],
        'data': ['*.csv'],
    },
    entry_points={
        'console_scripts': [
            'materials-demand=examples.run_simulation:main',
        ],
    },
)
