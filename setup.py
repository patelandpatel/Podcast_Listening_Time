from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] # removing the new line character 

        if HYPEN_E_DOT in requirements: # if '-e .' is present in the requirements, we remove it 
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Podcast_lisening_time_prediction',
version='0.0.1',
author='Parth',
author_email='ParthPatel3343@gmail.com',
packages=find_packages(),  ### Automatically find packages in the directory
install_requires=get_requirements('requirements.txt')

)