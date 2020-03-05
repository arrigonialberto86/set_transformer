from setuptools import setup, find_packages

setup(name='set_transformer',
      version='0.0.1',
      description='Set transformer TF implementation',
      author='Alberto Arrigoni',
      author_email='arrigonialberto86@gmail.com',
      url='https://github.com/arrigonialberto86',
      requires=['numpy', 'pandas', 'scipy', 'seaborn', 'pillow'],
      packages=find_packages()
)
