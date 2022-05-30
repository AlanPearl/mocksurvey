# mocksurvey

Some useful tools for conducting mock surveys of galaxies from the UniverseMachine empirical model.

# Author
- Alan Pearl

# Prerequisites
<!-- - g++ (check with `g++ --version`, install with `conda install gxx_linux-64`) -->
<!-- - gcc (check with `gcc --version`, install with `conda install gcc_linux-64`) -->
- `python=3.9.*` or `python=3.8.*`
- At the time of writing, a prerequisite (`halotools>=0.7`) is incompatible with `setuptools>=58`. You can fix this by running `pip install "setuptools<58"`. It is also currently incompatible with `python>=3.10`.
- Optional, but recommended to install Corrfunc (`pip install Corrfunc`)
  - Requires gsl installation (check with `gsl-config --version`, install with `conda install -c conda-forge gsl`)
### Example to automatically install prerequisites using a conda environment:
```
conda create -n py39-mocksurvey python=3.9 "setuptools<58" gsl
conda activate py39-mocksurvey
pip install Corrfunc
```

# Installation and Getting Started
```
pip install --upgrade git+https://github.com/AlanPearl/mocksurvey.git
```
<!--
```
cd /path/to/download/source/code
git clone https://github.com/AlanPearl/mocksurvey
pip install ./mocksurvey
```
-->

#### Set path to a directory where you would like to store downloaded data
```
python -m mocksurvey set-data-path /path/to/download/mocksurvey/data
```
#### Download and install UniverseMachine source code
```
cd /path/to/download/source/code
git clone https://bitbucket.org/pbehroozi/universemachine
cd universemachine
make
python -m mocksurvey config UM set-lightcone-executable ./lightcone
```
#### Download UniverseMachine snapshots. This redshift range (0 - inf) downloads all ~300 GB. You may choose a narrower range or enter a single value to download a single snapshot.
```
python -m mocksurvey download-um 0 inf
```
#### Download UltraVISTA photometry
```
python -m mocksurvey download-uvista
```
#### [Optional] Download raw synthetic spectra to assign to mock galaxies (~30 GB)
```
python -m mocksurvey download-uvista-mock-spectra
```
#### Get started by building a mock survey
```
python -m mocksurvey lightcone [--help]
```
#### View all commands
```
python -m mocksurvey --help
```

<!-- Deprecated -->
<!-- 
#### If you would like to run halotools models, using the SMDPL simulation
```
cd /path/to/download/source/code
bash mocksurvey/get_smdpl
```
-->
