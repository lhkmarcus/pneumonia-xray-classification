from typing import List
from setuptools import setup, find_packages

def get_requirements(filepath:str)->List[str]:
    """
    Returns list of requirements.
    """
    with open(filepath) as f_obj:
        requirements = f_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .") 

    return requirements

setup(
    name="pneumonia_xray_classification",
    version="0.0.1",
    author="lhkmarcus",
    author_email="lim.marcus.hk@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)