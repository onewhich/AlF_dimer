import numpy as np
from ase import units

class generate_representation:

    def __init__(self, systems):
        if type(systems) == type([]):
            self.coordinates = [i_sys.get_positions() for i_sys in systems]
            self.atomic_numbers = [i_sys.get_atomic_numbers() for i_sys in systems]
            try:
                self.forces = [i_sys.get_forces() for i_sys in systems]
            except:
                pass
        else:
            self.coordinates = [systems.get_positions()]
            self.atomic_numbers = [systems.get_atomic_numbers()]
            try:
                self.forces = [systems.get_forces()]
            except:
                pass
        
        self.n_elements = 86  # def2-TZVPP basis set can cover first 86 elements.
        self.generate_bond_length_lib()

    def Cartesian(self, centralize=True):
        Cartesian_representation = []
        if centralize:
            #centers = []
            for i_coord in self.coordinates:
                center = np.sum(i_coord, axis=0)/len(i_coord)
                i_coord -= center
                #centers.append(center)
                Cartesian_representation.append(i_coord.flatten())
            return Cartesian_representation #, centers
        else:
            for i_coord in self.coordinates:
                Cartesian_representation.append(i_coord.flatten())
            return Cartesian_representation

    def inverse_r_exp_r(self, relative_r=True):
        if relative_r:
            bond_length_lib = self.bond_length_lib
        else:
            bond_length_lib = np.ones((self.n_elements+1,self.n_elements+1))
        rep_inverse_r_exp_r = []
        for i_sys in range(len(self.coordinates)):
            rep_sys = []
            pairs_rij = {}
            for i, i_coord in enumerate(self.coordinates[i_sys][:-1]):
                for k, j_coord in enumerate(self.coordinates[i_sys][i+1:]):
                    j = k+i+1
                    r_ref = bond_length_lib[self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    r_ij = np.linalg.norm(i_coord - j_coord) / r_ref
                    pair = [self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    pair.sort()
                    if not str(pair) in pairs_rij:
                        pairs_rij[str(pair)] = [r_ij]
                    else:
                        pairs_rij[str(pair)] += [r_ij]
            sorted_key = sorted(pairs_rij)
            for key in sorted_key:
                pairs_rij[key].sort()
                for r_ij in pairs_rij[key]:
                    inverse_r = 1 / r_ij
                    exp_r = np.exp(-r_ij)
                    rep_sys.append(inverse_r)
                    rep_sys.append(exp_r)
            rep_sys = np.array(rep_sys)
            rep_inverse_r_exp_r.append(rep_sys)
        rep_inverse_r_exp_r = np.array(rep_inverse_r_exp_r)
        return rep_inverse_r_exp_r

    def inverse_r_exp_r_unsorted(self, relative_r=True):
        if relative_r:
            bond_length_lib = self.bond_length_lib
        else:
            bond_length_lib = np.ones((self.n_elements+1,self.n_elements+1))
        rep_inverse_r_exp_r = []
        for i_sys in range(len(self.coordinates)):
            rep_sys = []
            for i, i_coord in enumerate(self.coordinates[i_sys][:-1]):
                for k, j_coord in enumerate(self.coordinates[i_sys][i+1:]):
                    j = k+i+1
                    r_ref = bond_length_lib[self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    r_ij = np.linalg.norm(i_coord - j_coord) / r_ref
                    inverse_r = 1 / r_ij
                    exp_r = np.exp(-r_ij)
                    rep_sys.append(inverse_r)
                    rep_sys.append(exp_r)
            rep_sys = np.array(rep_sys)
            rep_inverse_r_exp_r.append(rep_sys)
        rep_inverse_r_exp_r = np.array(rep_inverse_r_exp_r)
        return rep_inverse_r_exp_r

    def inverse_r(self, relative_r=True):
        if relative_r:
            bond_length_lib = self.bond_length_lib
        else:
            bond_length_lib = np.ones((self.n_elements+1,self.n_elements+1))
        rep_inverse_r_exp_r = []
        for i_sys in range(len(self.coordinates)):
            rep_sys = []
            for i, i_coord in enumerate(self.coordinates[i_sys][:-1]):
                for k, j_coord in enumerate(self.coordinates[i_sys][i+1:]):
                    j = k+i+1
                    r_ref = bond_length_lib[self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    r_ij = np.linalg.norm(i_coord - j_coord) / r_ref
                    inverse_r = 1 / r_ij
                    rep_sys.append(inverse_r)
            rep_sys = np.array(rep_sys)
            rep_inverse_r_exp_r.append(rep_sys)
        rep_inverse_r_exp_r = np.array(rep_inverse_r_exp_r)
        return rep_inverse_r_exp_r

    def distance_matrix(self, relative_r=False, bond_length_cutoff=True, cutoff_factor=1.4):
        if relative_r:
            bond_length_lib = self.bond_length_lib
        else:
            bond_length_lib = np.ones((self.n_elements+1,self.n_elements+1))
        if bond_length_cutoff:
            cutoff = cutoff_factor * bond_length_lib
        distance_matrix = []
        for i_sys in range(len(self.coordinates)):
            rep_sys = []
            pairs_rij = {}
            for i, i_coord in enumerate(self.coordinates[i_sys][:-1]):
                for k, j_coord in enumerate(self.coordinates[i_sys][i+1:]):
                    j = k+i+1
                    r_ref = bond_length_lib[self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    r_ij = np.linalg.norm(i_coord - j_coord) / r_ref
                    if bond_length_cutoff:
                        if r_ij > cutoff[self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]:
                            r_ij = 0
                    pair = [self.atomic_numbers[i_sys][i], self.atomic_numbers[i_sys][j]]
                    pair.sort()
                    if not str(pair) in pairs_rij:
                        pairs_rij[str(pair)] = [r_ij]
                    else:
                        pairs_rij[str(pair)] += [r_ij]
            sorted_key = sorted(pairs_rij)
            for key in sorted_key:
                pairs_rij[key].sort()
                for r_ij in pairs_rij[key]:
                    rep_sys.append(r_ij)
            distance_matrix.append(np.array(rep_sys))
        return np.array(distance_matrix)

    def generate_bond_length_lib(self):
        '''Bond length calculated from dimer at CCSD/def2-TZVPP level.'''
        self.bond_length_lib=np.ones((self.n_elements+1,self.n_elements+1))
        self.bond_length_lib[1,1] = 0.7423 # H-H
        self.bond_length_lib[1,6] = 1.1195 # H-C
        self.bond_length_lib[1,8] = 0.9687 # H-O
        self.bond_length_lib[6,6] = 1.3849 # C-C
        self.bond_length_lib[6,8] = 1.1272 # C-O
        self.bond_length_lib[8,8] = 1.2005 # O-O
        self.bond_length_lib[9,9] = 1.3952 # F-F
        self.bond_length_lib[9,13] = 1.6608 # F-Al
        self.bond_length_lib[13,13] = 2.5727 # Al-Al
        for i in range(len(self.bond_length_lib[1:-1])):
            for j in range(len(self.bond_length_lib[i+1:])):
                self.bond_length_lib[j,i] = self.bond_length_lib[i,j]

    # the following method is not correct to represent forces
    def forces_representation(self):
        '''The forces are represented on relative atomic positions as basis functions.
        These basis functions can form a complete set, when the force can be decomposed into the linear combination of the basis functions (relative coordinates).
        f_1 = c2 (r_2 - r_1) / ||r_2 - r_1|| + c3 (r_3 - r_1) / ||r_3 - r_1|| + ...
        f_2 = c1 (r_1 - r_2) / ||r_1 - r_2|| + c3 (r_3 - r_2) / ||r_3 - r_2|| + ...
        c1 = f_2 dot (r_1 - r_2) / ||r_1 - r_2||
        c2 = f_1 dot (r_2 - r_1) / ||r_2 - r_1||
        f_1, f_2: the forces of atom 1 and atom 2.
        c1, c2, c3: the coefficients of basis functions.
        r_1, r_2, r_3: the Cartesian coordinates of atom 1,2,3.
        '''
        atom_indices = [np.array(range(len(coordinate))) for coordinate in self.coordinates ]
        basis_functions = []
        basis_norms = []
        for i, i_coordinate in enumerate(self.coordinates):
            sys_basis = []
            sys_norms = []
            for i_atom in atom_indices[i]:
                basis = i_coordinate - i_coordinate[i_atom]
                basis = np.delete(basis, i_atom, axis=0)
                sys_basis.append(basis)
                sys_norms.append(np.linalg.norm(basis, axis=-1))
            basis_functions.append(np.array(sys_basis))
            basis_norms.append(np.array(sys_norms))
        
        forces_coefficient = []
        for i, i_force in enumerate(self.forces):
            sys_coefficient = []
            for i_atom in atom_indices[i]:
                coefficient = np.dot(i_force[i_atom], basis_functions[i][i_atom].T) / basis_norms[i][i_atom]
                sys_coefficient.append(coefficient)
            forces_coefficient.append(np.array(sys_coefficient))
        
        return forces_coefficient, basis_functions, basis_norms

    @staticmethod
    def generate_Al2F2_representation(Al2F2, representation_name="inverse_r_exp_r"):
        """
            Al2F2: ASE Atoms object, with positions information of Al2F2
            representation_name: Name of the structural representation:
                                 Values:
                                       inverse_r_exp_r: 1/r_{ij}, 1/R_{ij}, exp(-r_{ij}), exp(-R), where R is the largest AlF-AlF distance.
                                       inverse_r_exp_r_relative: Same as inverse_r_exp_r, except r_{ij}=r_{ij}/r_{ij,optimized}, 
                                                                 where the interatomic distances are normalized by the equilibrium interatomic
                                                                 distances of diatomic molecule ij, optimized by ab intio methods, e.g. CCSD(T).
        """
        Al1 = Al2F2.get_positions()[0]# / units.Bohr # change length in AA to Bohr
        F2 = Al2F2.get_positions()[1] #/ units.Bohr
        Al3 = Al2F2.get_positions()[2]# / units.Bohr
        F4 = Al2F2.get_positions()[3] #/ units.Bohr

        r_Al1_F2 = np.linalg.norm(Al1 - F2)
        r_Al3_F4 = np.linalg.norm(Al3 - F4)
        r_Al1_F4 = np.linalg.norm(Al1 - F4)
        r_Al3_F2 = np.linalg.norm(Al3 - F2)
        r_Al1_Al3 = np.linalg.norm(Al1 - Al3)
        r_F2_F4 = np.linalg.norm(F2 - F4)
        R1 = np.linalg.norm((Al1 + F2)/2 - (Al3 + F4)/2)
        R2 = np.linalg.norm((Al1 + F4)/2 - (Al3 + F2)/2)
        R = max(R1, R2)
        if(representation_name == "inverse_r_exp_r"):

            inverse_r_Al1_F2 = 1.0/r_Al1_F2
            inverse_r_Al3_F4 = 1.0/r_Al3_F4
            inverse_r_Al1_F4 = 1.0/r_Al1_F4
            inverse_r_Al3_F2 = 1.0/r_Al3_F2
            inverse_r_Al1_Al3 = 1.0/r_Al1_Al3
            inverse_r_F2_F4 = 1.0/r_F2_F4
            inverse_R = 1.0/R
            exp_r_Al1_F2 = np.exp(-r_Al1_F2)
            exp_r_Al3_F4 = np.exp(-r_Al3_F4)
            exp_r_Al1_F4 = np.exp(-r_Al1_F4)
            exp_r_Al3_F2 = np.exp(-r_Al3_F2)
            exp_r_Al1_Al3 = np.exp(-r_Al1_Al3)
            exp_r_F2_F4 = np.exp(-r_F2_F4)
            exp_R = np.exp(-R)
        elif(representation_name == "inverse_r_exp_r_relative"):

            # Optimized bond lengths (CCSD(T)/avqz)
            r_F2 = 1.413030165 / units.Bohr # From AA to Bohr
            r_Al2 = 2.568675944 / units.Bohr
            r_AlF = 1.668734698 / units.Bohr
            inverse_r_Al1_F2 = 1.0/r_Al1_F2 * r_AlF
            inverse_r_Al3_F4 = 1.0/r_Al3_F4 * r_AlF
            inverse_r_Al1_F4 = 1.0/r_Al1_F4 * r_AlF
            inverse_r_Al3_F2 = 1.0/r_Al3_F2 * r_AlF
            inverse_r_Al1_Al3 = 1.0/r_Al1_Al3 * r_Al2
            inverse_r_F2_F4 = 1.0/r_F2_F4 * r_F2
            inverse_R = 1.0/R
            exp_r_Al1_F2 = np.exp(-r_Al1_F2 / r_AlF)
            exp_r_Al3_F4 = np.exp(-r_Al3_F4 / r_AlF)
            exp_r_Al1_F4 = np.exp(-r_Al1_F4 / r_AlF)
            exp_r_Al3_F2 = np.exp(-r_Al3_F2 / r_AlF)
            exp_r_Al1_Al3 = np.exp(-r_Al1_Al3 / r_Al2)
            exp_r_F2_F4 = np.exp(-r_F2_F4 / r_F2)
            exp_R = np.exp(-R)
        else:
            print("Error: Representation not implemented:", representation_name)

        features_invr_AlF = np.array([inverse_r_Al1_F2,inverse_r_Al3_F4, inverse_r_Al1_F4, inverse_r_Al3_F2])
        features_invr_Al2_F2 = np.array([inverse_r_Al1_Al3, inverse_r_F2_F4, inverse_R])
        features_expr_AlF = np.array([exp_r_Al1_F2,exp_r_Al3_F4, exp_r_Al1_F4, exp_r_Al3_F2])
        features_expr_Al2_F2 = np.array([exp_r_Al1_Al3,exp_r_F2_F4, exp_R])

        representation = np.concatenate((np.sort(features_invr_AlF),
                                  features_invr_Al2_F2,
                                  np.sort(features_expr_AlF),
                                  features_expr_Al2_F2))
                                  
                
    
        return np.array(representation)

