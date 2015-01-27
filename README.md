This documentation is incomplete, hang in there


## About

This code uses the algorithm found in Radionov, Athanassoula & Sotnikova (2009)
for generating the initial conditions for a galaxy simulation
with the code GADGET-2, including live gaseous/stellar disk, halo and bulge
components.


## Required libraries
 
* NumPy (python-numpy)
* SciPy (python-scipy)
* [pyGadgetReader](https://bitbucket.org/rthompson/pygadgetreader)


## Usage

### galaxy.py

    usage: galaxy.py [-h] [-o init.dat] [-cores CORES]
    
    optional arguments:
      -h, --help    show this help message and exit
      -o init.dat   The name of the output file.
      -cores CORES  The number of cores to use.


## Author

    Rafael Ruggiero
    Undergraduate student at Universidade de São Paulo (USP), Brazil
    Contact: bluewhale [at] cecm.usp.br


## Disclaimer

Feel free to use this code in your work, but please link this page
in your paper.
