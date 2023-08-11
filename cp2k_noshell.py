"""This module defines an ASE interface to CP2K.

https://www.cp2k.org/
Author: Louie Slocombe
"""

import os
import os.path
from warnings import warn
import numpy as np

from ase.units import Rydberg, Ha, Bohr
from ase.calculators.calculator import FileIOCalculator, SCFError


class CP2K(FileIOCalculator):
    """ASE-Calculator for CP2K.

        CP2K is a program to perform atomistic and molecular simulations of solid
        state, liquid, molecular, and biological systems. It provides a general
        framework for different methods such as e.g., density functional theory
        (DFT) using a mixed Gaussian and plane waves approach (GPW) and classical
        pair and many-body potentials.

        CP2K is freely available under the GPL license.
        It is written in Fortran 2003 and can be run efficiently in parallel.

        Check https://www.cp2k.org about how to obtain and install CP2K.
        Make sure that you also have the CP2K-shell available, since it is required
        by the CP2K-calulator.

        The command used by the calculator to launch CP2K is
        ``cp2k``. To run a parallelized simulation use something like this::

            CP2K.command="srun cp2k.psmp -i cp2k.inp >> cp2k.out"

        Arguments:

        basis_set: str
            Name of the basis set to be use.
            The default is ``DZVP-MOLOPT-SR-GTH``.
        basis_set_file: str
            Filename of the basis set file.
            Default is ``BASIS_MOLOPT``.
            Set the environment variable $CP2K_DATA_DIR
            to enabled automatic file discovered.
        charge: float
            The total charge of the system.  Default is ``0``.
        command: str
            The command used to launch CP2K.
            If ``command`` is not passed as an argument to the
            constructor, the class-variable ``CP2K.command``,
            and then the environment variable
            ``$ASE_CP2K_COMMAND`` are checked.
            Eventually, ``cp2k_shell`` is used as default.
        cutoff: float
            The cutoff of the finest grid level.  Default is ``400 * Rydberg``.
        debug: bool
            Flag to enable debug mode. This will keep all
            CP2K outputs. Default is ``False``.
        force_eval_method: str
            The method CP2K uses to evaluate energies and forces.
            The default is ``Quickstep``, which is CP2K's
            module for electronic structure methods like DFT.
        inp: str
            CP2K input template. If present, the calculator will
            augment the template, e.g. with coordinates, and use
            it to launch CP2K. Hence, this generic mechanism
            gives access to all features of CP2K.
            Note, that most keywords accept ``None`` to disable the generation
            of the corresponding input section.

            This input template is important for advanced CP2K
            inputs, but is also needed for e.g. controlling the Brillouin
            zone integration. The example below illustrates some common
            options::

                inp = '''&FORCE_EVAL
                   &DFT
                     &KPOINTS
                       SCHEME MONKHORST-PACK 12 12 8
                     &END KPOINTS
                     &SCF
                       ADDED_MOS 10
                       &SMEAR
                         METHOD FERMI_DIRAC
                         ELECTRONIC_TEMPERATURE [K] 500.0
                       &END SMEAR
                     &END SCF
                   &END DFT
                 &END FORCE_EVAL
                '''

        max_scf: int
            Maximum number of SCF iteration to be performed for
            one optimization. Default is ``50``.
        poisson_solver: str
            The poisson solver to be used. Currently, the only supported
            values are ``auto`` and ``None``. Default is ``auto``.
        potential_file: str
            Filename of the pseudo-potential file.
            Default is ``POTENTIAL``.
            Set the environment variable $CP2K_DATA_DIR
            to enabled automatic file discovered.
        pseudo_potential: str
            Name of the pseudo-potential to be use.
            Default is ``auto``. This tries to infer the
            potential from the employed XC-functional,
            otherwise it falls back to ``GTH-PBE``.
        stress_tensor: bool
            Indicates whether the analytic stress-tensor should be calculated.
            Default is ``True``.
        uks: bool
            Requests an unrestricted Kohn-Sham calculations.
            This is need for spin-polarized systems, ie. with an
            odd number of electrons. Default is ``False``.
        xc: str
            Name of exchange and correlation functional.
            Accepts all functions supported by CP2K itself or libxc.
            Default is ``LDA``.
        print_level: str
            PRINT_LEVEL of global output.
            Possible options are:
            DEBUG Everything is written out, useful for debugging purposes only
            HIGH Lots of output
            MEDIUM Quite some output
            LOW Little output
            SILENT Almost no output
            Default is 'LOW'
        """

    implemented_properties = ['energy', 'free_energy', 'forces']
    command = None

    default_parameters = dict(
        auto_write=False,
        basis_set='DZVP-MOLOPT-SR-GTH',
        basis_set_file='BASIS_MOLOPT',
        charge=0,
        cutoff=400 * Rydberg,
        force_eval_method="Quickstep",
        inp='',
        max_scf=50,
        potential_file='POTENTIAL',
        pseudo_potential='auto',
        stress_tensor=True,
        uks=False,
        poisson_solver='auto',
        xc='LDA',
        print_level='LOW')

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='cp2k', atoms=None, command=None, debug=False, **kwargs):
        """Construct CP2K-calculator object."""
        self.debug = debug
        self.label = None
        self.parameters = None
        self.results = None
        self.atoms = None

        # Several places are check to determine self.command
        if command is not None:
            self.command = command
        elif CP2K.command is not None:
            self.command = CP2K.command
        elif 'ASE_CP2K_COMMAND' in os.environ:
            self.command = os.environ['ASE_CP2K_COMMAND']
        else:
            self.command = 'cp2k'

        FileIOCalculator.__init__(self, restart=restart,
                                  ignore_bad_restart_file=ignore_bad_restart_file,
                                  label=label, atoms=atoms, **kwargs)

    def _generate_input(self):
        """Generates a CP2K input file and returns a string with the contents."""
        p = self.parameters
        # Adds a force evaluation block to the input file.
        root = parse_input(p.inp)
        # Add the label to the global section
        root.add_keyword('GLOBAL', 'PROJECT ' + self.label)
        # Add the forces to the print section
        root.add_keyword('FORCE_EVAL/PRINT/FORCES',
                         '_SECTION_PARAMETERS_ ON')
        # Add the total energy to the print section??
        root.add_keyword('FORCE_EVAL/PRINT/PROGRAM_RUN_INFO',
                            '_SECTION_PARAMETERS_ ON')
        # Add the number of atoms to the print section
        root.add_keyword('FORCE_EVAL/PRINT/TOTAL_NUMBERS',
                            '_SECTION_PARAMETERS_ ON')

        # all the rest...
        if p.print_level:
            root.add_keyword('GLOBAL', 'PRINT_LEVEL ' + p.print_level)
        if p.force_eval_method:
            root.add_keyword('FORCE_EVAL', 'METHOD ' + p.force_eval_method)
        if p.stress_tensor:
            root.add_keyword('FORCE_EVAL', 'STRESS_TENSOR ANALYTICAL')
            root.add_keyword('FORCE_EVAL/PRINT/STRESS_TENSOR',
                             '_SECTION_PARAMETERS_ ON')
        if p.basis_set_file:
            root.add_keyword('FORCE_EVAL/DFT',
                             'BASIS_SET_FILE_NAME ' + p.basis_set_file)
        if p.potential_file:
            root.add_keyword('FORCE_EVAL/DFT',
                             'POTENTIAL_FILE_NAME ' + p.potential_file)
        if p.cutoff:
            root.add_keyword('FORCE_EVAL/DFT/MGRID',
                             'CUTOFF [eV] %.18e' % p.cutoff)
        if p.max_scf:
            root.add_keyword('FORCE_EVAL/DFT/SCF', 'MAX_SCF %d' % p.max_scf)
            root.add_keyword('FORCE_EVAL/DFT/LS_SCF', 'MAX_SCF %d' % p.max_scf)

        if p.xc:
            legacy_libxc = ""
            for functional in p.xc.split():
                functional = functional.replace("LDA", "PADE")  # resolve alias
                xc_sec = root.get_subsection('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL')
                # libxc input section changed over time
                if functional.startswith("XC_"):
                    s = InputSection(name=functional[3:])
                    xc_sec.subsections.append(s)
                else:
                    s = InputSection(name=functional.upper())
                    xc_sec.subsections.append(s)
            if legacy_libxc:
                root.add_keyword('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/LIBXC',
                                 'FUNCTIONAL ' + legacy_libxc)

        if p.uks:
            root.add_keyword('FORCE_EVAL/DFT', 'UNRESTRICTED_KOHN_SHAM ON')

        if p.charge and p.charge != 0:
            root.add_keyword('FORCE_EVAL/DFT', 'CHARGE %d' % p.charge)

        # add Poisson solver if needed
        if p.poisson_solver == 'auto' and not any(self.atoms.get_pbc()):
            root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PERIODIC NONE')
            root.add_keyword('FORCE_EVAL/DFT/POISSON', 'PSOLVER  MT')

        # write coords
        syms = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        for elm, pos in zip(syms, positions):
            line = '{} {:.13e} {:.13e} {:.13e}'.format(elm, pos[0], pos[1], pos[2])
            root.add_keyword('FORCE_EVAL/SUBSYS/COORD', line, unique=False)

        # write cell
        pbc = ''.join([a for a, b in zip('XYZ', self.atoms.get_pbc()) if b])
        if len(pbc) == 0:
            pbc = 'NONE'
        root.add_keyword('FORCE_EVAL/SUBSYS/CELL', 'PERIODIC ' + pbc)
        c = self.atoms.get_cell()
        for i, a in enumerate('ABC'):
            line = '{} {:.13e} {:.13e} {:.13e}'.format(a, c[i, 0], c[i, 1], c[i, 2])
            root.add_keyword('FORCE_EVAL/SUBSYS/CELL', line)

        # determine pseudo-potential
        potential = p.pseudo_potential
        if p.pseudo_potential == 'auto':
            if p.xc and p.xc.upper() in ('LDA', 'PADE', 'BP', 'BLYP', 'PBE',):
                potential = 'GTH-' + p.xc.upper()
            else:
                msg = 'No matching pseudo potential found, using GTH-PBE'
                warn(msg, RuntimeWarning)
                potential = 'GTH-PBE'  # fall back

        # write atomic kinds
        if p.basis_set or p.pseudo_potential:
            subsys = root.get_subsection('FORCE_EVAL/SUBSYS').subsections
            kinds = dict([(s.params, s) for s in subsys if s.name == "KIND"])
            for elem in set(self.atoms.get_chemical_symbols()):
                if elem not in kinds.keys():
                    s = InputSection(name='KIND', params=elem)
                    subsys.append(s)
                    kinds[elem] = s
                if p.basis_set:
                    kinds[elem].keywords.append('BASIS_SET ' + p.basis_set)
                if potential:
                    kinds[elem].keywords.append('POTENTIAL ' + potential)

        output_lines = ['!!! Generated by ASE !!!'] + root.write()
        return '\n'.join(output_lines)

    def write_input(self, atoms, properties=None, system_changes=None):
        """Writes the CP2K input file."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        file_string = self._generate_input()  ### SHOULD THIS PICK UP ATOMS AND NOT SELF.ATOMS???
        file_in = self.label + '.inp'
        file_out = self.label + '.out'
        with open(file_in, 'w') as fileobj:
            fileobj.write(file_string)
        # check if the output file exists and remove it if it does
        if self.debug is False and os.path.isfile(file_out):
            os.remove(file_out)

    def read(self, label):
        """Reads the CP2K output file."""
        raise NotImplementedError

    def read_results(self):
        """Reads the CP2K output file and returns the results."""
        filename = self.label + '.out'
        data = np.genfromtxt(filename, comments="None", delimiter="\n", dtype=str)
        self.results["energy"] = read_energy(data)
        self.results['free_energy'] = self.results['energy']
        self.results["forces"] = read_forces(data)
        # self.results["stress"] = read_stress(data)

        # # check if SCF converged
        # if "SCF run NOT converged" in data:
        #     raise SCFError()

class InputSection:
    """Represents a section of a CP2K input file"""

    def __init__(self, name, params=None):
        self.name = name.upper()
        self.params = params
        self.keywords = []
        self.subsections = []

    def write(self):
        """Outputs input section as string"""
        output = []
        for k in self.keywords:
            output.append(k)
        for s in self.subsections:
            if s.params:
                output.append('&%s %s' % (s.name, s.params))
            else:
                output.append('&%s' % s.name)
            for line in s.write():
                output.append('   %s' % line)
            output.append('&END %s' % s.name)
        return output

    def add_keyword(self, path, line, unique=True):
        """Adds a keyword to section."""
        parts = path.upper().split('/', 1)
        candidates = [s for s in self.subsections if s.name == parts[0]]
        if len(candidates) == 0:
            s = InputSection(name=parts[0])
            self.subsections.append(s)
            candidates = [s]
        elif len(candidates) != 1:
            raise Exception('Multiple %s sections found ' % parts[0])

        key = line.split()[0].upper()
        if len(parts) > 1:
            candidates[0].add_keyword(parts[1], line, unique)
        elif key == '_SECTION_PARAMETERS_':
            if candidates[0].params is not None:
                msg = 'Section parameter of section %s already set' % parts[0]
                raise Exception(msg)
            candidates[0].params = line.split(' ', 1)[1].strip()
        else:
            old_keys = [k.split()[0].upper() for k in candidates[0].keywords]
            if unique and key in old_keys:
                msg = 'Keyword %s already present in section %s'
                raise Exception(msg % (key, parts[0]))
            candidates[0].keywords.append(line)

    def get_subsection(self, path):
        """Finds a subsection"""
        parts = path.upper().split('/', 1)
        candidates = [s for s in self.subsections if s.name == parts[0]]
        if len(candidates) > 1:
            raise Exception('Multiple %s sections found ' % parts[0])
        if len(candidates) == 0:
            s = InputSection(name=parts[0])
            self.subsections.append(s)
            candidates = [s]
        if len(parts) == 1:
            return candidates[0]
        return candidates[0].get_subsection(parts[1])


def parse_input(inp):
    """Parses the given CP2K input string"""
    root_section = InputSection('CP2K_INPUT')
    section_stack = [root_section]

    for line in inp.split('\n'):
        line = line.split('!', 1)[0].strip()
        if len(line) == 0:
            continue

        if line.upper().startswith('&END'):
            s = section_stack.pop()
        elif line[0] == '&':
            parts = line.split(' ', 1)
            name = parts[0][1:]
            if len(parts) > 1:
                s = InputSection(name=name, params=parts[1].strip())
            else:
                s = InputSection(name=name)
            section_stack[-1].subsections.append(s)
            section_stack.append(s)
        else:
            section_stack[-1].keywords.append(line)

    return root_section


def find_substring_array(lines, sub):
    """Finds the last occurrence of the given substring in the given array of lines"""
    loc = np.flatnonzero(np.core.defchararray.find(lines, sub) != -1)[-1]
    return loc


def read_energy(lines, sub='ENERGY|'):
    """Reads the energy from the given array of lines using substring matching"""
    loc = find_substring_array(lines, sub)
    energy = float(lines[loc].split(":")[-1])
    # convert to eV
    energy *= Ha
    return energy


def get_number_of_atoms(lines, sub="- Atoms:"):
    """Reads the number of atoms from the given array of lines using substring matching"""
    loc = find_substring_array(lines, sub)
    return int(lines[loc].split(":")[-1])


def read_forces(lines, sub="# Atom   Kind   Element"):
    """Reads the forces from the given array of lines using substring matching"""
    # Get number of atoms
    n = get_number_of_atoms(lines)
    loc = find_substring_array(lines, sub)
    forces = np.zeros((n, 3), dtype=float)
    for i in range(n):
        # Extract the forces
        forces[i] = lines[loc + i + 1].split()[3:]
        # convert to eV/A
        forces[i] *= Ha / Bohr
    return forces


def read_stress(lines, sub="STRESS|                        x"):
    """Reads the stress from the given array of lines using substring matching"""
    loc = find_substring_array(lines, sub)
    stress = np.zeros((3, 3), dtype=float)
    for i in range(3):
        stress[i] = lines[loc + i + 1].split()[2:]
    # convert to eV/A^3
    # stress *= Ha / Bohr**3
    stress /= 160.21766208
    # stress = np.array([float(x) for x in stress.split()]).reshape(3, 3)
    # Convert 3x3 stress tensor to Voigt form as required by ASE
    stress = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                       stress[1, 2], stress[0, 2], stress[0, 1]])
    stress = -1.0 * stress
    # stress = np.array([float(x) for x in line.split()]).reshape(3, 3)
    assert np.all(stress == np.transpose(stress))  # should be symmetric
    return stress
