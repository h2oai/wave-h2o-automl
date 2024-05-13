from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomInstall(install):
    def run(self):
        # Execute o comando make setup antes da instalação
        subprocess.run(['make', 'setup'])
        # Ativar o ambiente virtual após a instalação
        subprocess.run(['source', 'venv/bin/activate'], shell=True)
        # Chame o método run da classe pai para continuar a instalação normalmente
        install.run(self)

setup(
    name='wave-h2o-automl',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    cmdclass={
        'install': CustomInstall,
    },
)