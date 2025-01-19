import glob

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    setup_requires=["pbr>=1.8", "setuptools>=17.1"],
    pbr=True,
    scripts=glob.glob("scripts/*.py"),
    name="spec2psm",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'spec2psm=scripts.spec2psm_cli:main',
        ],
    },
    url="https://github.com/thinkle12/spec2psm",
    license="Apache-2",
    author="Trent Hinkle",
    author_email="trenth12@gmail.com",
    description="Python Package for training, evaluating, and inference of a peptide spectrum to peptide sequence transformer model with pytorch.",
    keywords=['deep learning', 'proteomics', 'mass spectrometry', 'pytorch'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
