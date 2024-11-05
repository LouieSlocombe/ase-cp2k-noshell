# ase-cp2k-noshell
This package is a modified ASE CP2K calculator that does not use cp2k-shell.

Run with the following command in your `.bashrc`

`export ASE_CP2K_COMMAND="srun --hint=nomultithread --distribution=block:block cp2k.psmp -i cp2k.inp >> cp2k.out"`

Install by `pip install git+https://github.com/LouieSlocombe/ase-cp2k-noshell.git`.
