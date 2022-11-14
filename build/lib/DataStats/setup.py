import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="DataStats",
    version="0.0.1",
    description="Analysis of English summarization datasets",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://git-dbs.ifi.uni-heidelberg.de/practicals/2022-jiahui-li",
    author="Jiahui Li",
    author_email="jiahui.li@stud.uni-heidelberg.de",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
)