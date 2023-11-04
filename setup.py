from setuptools import find_packages, setup


setup(
    author="MSR-I Systems Group",
    python_requires='>=3.10',
    description="A LLM inference cluster simulator",
    include_package_data=True,
    keywords='simulator',
    name='simulator',
    packages=find_packages(include=['simulator', 'simulator.*']),
    version='0.0.1',
)
