# ase-cp2k-noshell
Code for the ASE CP2K calculator but without using cp2k-shell

Run with
export ASE_CP2K_COMMAND="srun --hint=nomultithread --distribution=block:block cp2k.psmp -i cp2k.inp >> cp2k.out"
