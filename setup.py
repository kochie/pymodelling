from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name='pymodelling',
  version='0.1.1',
  description='A set of tools for modelling systems',
  url='https://github.com/kochie/pymodelling',
  author='Robert Koch',
  author_email='robert@kochie.io',
  ong_description=long_description,
  long_description_content_type="text/markdown",
  license='MIT',
  packages=find_packages(),
  install_requires=[
    'numpy',
    'matplotlib'
  ],
  zip_safe=False,
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)