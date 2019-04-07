from setuptools import setup

setup(name="mocksurvey",
      version="0.0.1.dev6",
      description="Some useful tools for conducting realistic mock surveys out of galaxies populated by `halotools` models.",
      url="http://github.com/AlanPearl/mocksurvey",
      author="Alan Pearl",
      author_email="alanpearl@pitt.edu",
      license="MIT",
      packages=["mocksurvey"],
      install_requires = [
		"halotools",
		"Corrfunc",
		"emcee>=3.0rc2",
		"corner"],
      zip_safe=True,
      )
