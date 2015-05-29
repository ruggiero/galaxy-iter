## About

This code uses the algorithm found in Radionov, Athanassoula & Sotnikova
(2009) for generating the initial conditions for a galaxy simulation
with the code GADGET-2, including live gaseous/stellar disk, halo and
bulge components.

To use it, you need to compile 4 different modified versions of GADGET-2
and put them in the galaxy/ folder, with the names gadget_0, gadget_1,
gadget_2 and gadget_3. They respectively only allow gas particles,
dark matter particles, disk particles and bulge particles to move.
I manually updated GADGET-2 for that.  Additionally, gadget_0 must be
compiled with the -DISOTHERMAL flag._

This code is highly experimental, and I don't use it anymore for three
reasons: it takes a long time to generate the initial conditions; it
has an unidentified bug which makes halo and bulge particles accumulate
around the z=0 axis once the galaxy is simulated; and I have another code
that just works™, which is available in the galaxy-jeans repository.


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
