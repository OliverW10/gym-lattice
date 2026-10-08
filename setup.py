from setuptools import find_packages, setup

requirements = [
      'gymnasium',
      'numpy'
]

setup(name='gym_lattice',
      version='0.0.2',
      packages=find_packages(),
      install_requires=requirements,
      description="An HP 2D Lattice Gym Environment for Protein Folding",
      author="Lester James V. Miranda",
      author_email='ljvmiranda@gmail.com',
      url='https://github.com/ljvmiranda921/gym-lattice'
)