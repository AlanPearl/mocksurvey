# mocksurvey

Some useful tools for conducting mock surveys of galaxies from the UniverseMachine empirical model.

# Author
___
- Alan Pearl

# Prerequisites
___
- numpy >= 1.7 (check with `pip show numpy`, install with `pip install numpy`)
- g++ (check with `g++ --version`, install with `conda install gxx_linux-64`)
- gcc (check with `gcc --version`, install with `conda install gcc_linux-64`)
- `pip install "halotools>=0.7"`
### for halotools:
- Note: At the time of writing, halotools is incompatible with `setuptools>=58`. You can fix this by running `pip install "setuptools<58"`.
### for Corrfunc (optional install: `pip install Corrfunc`):
- gsl (check with `gsl-config --version`, install with `conda install -c conda-forge gsl`)

# Installation Instructions
___
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
