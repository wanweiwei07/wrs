from setuptools import setup, find_packages


def requirements_from_file(file_name):
    return open(file_name).read().splitlines()


setup(
    name="wrs",
    version="0.0.0",
    packages=find_packages(),
    description="The WRS Robot Planning & Control System",
    author="Weiwei Wan",
    author_email="wan@sys.es.osaka-u.ac.jp",
    url="https://github.com/wanweiwei07/wrs",
    platforms=['Windows', 'Linux Ubuntu'],
    install_requires=requirements_from_file('requirements.txt'),
    include_package_data=True,
)