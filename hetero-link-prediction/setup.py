from setuptools import setup, find_packages

setup(
    name='hetero_link_prediction',
    version='0.1.0',
    description='A baseline heterogeneous link prediction project',
    author='AI Assistant',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch_geometric',
        'scikit-learn',
        'pandas',
        'pyyaml',
        'tqdm'
    ],
)
