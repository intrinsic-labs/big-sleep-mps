import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['big_sleep']
from version import __version__

setup(
  name = 'big-sleep-mps',
  packages = find_packages(),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'dream = big_sleep.cli:main',
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Big Sleep (CLIP + BigGAN text-to-image) ported to Apple Silicon / MPS',
  author = 'Ryan Murdock, Phil Wang; MPS port by Asher Pope',
  url = 'https://github.com/intrinsic-labs/big-sleep-mps',
  python_requires = '>=3.9',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text to image',
    'generative adversarial networks',
    'apple silicon',
    'mps'
  ],
  install_requires=[
    'torch>=2.0.0',
    'einops>=0.3',
    'fire',
    'ftfy',
    'pytorch-pretrained-biggan>=0.1.0',
    'regex',
    'torchvision>=0.15.0',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
