"""Packaging file for CVU Installation
"""
import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(name="cvu-python",
      version="0.0.2",
      description="Computer Vision deployment tools for dummies and experts.",
      long_description=README,
      long_description_content_type="text/markdown",
      include_package_data=True,
      url="https://github.com/BlueMirrors/cvu",
      author="BlueMirrors",
      author_email="contact.bluemirrors@gmail.com",
      license="Apache Software License v2.0",
      classifiers=[
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
      ],
      packages=find_packages(exclude=("examples", )),
      install_requires=["opencv-python", "vidsz", "numpy", "gdown"])
