import numpy as np
import pickle
from copy import deepcopy
from structure_io import read_xyz_traj, write_xyz_traj
import math
from ase.calculators.calculator import Calculator
from representation import generate_Al2F2_representation
from sklearn.gaussian_process import GaussianProcessRegressor, kernels



class ml_potential(Calculator):
    '''
    This class implements loading machine learned many body potential and returning the potential as the output of the Cartesian coordinates input.
    INPUT:
    trained_ml_potential: trained model in scikit-learn format and loaded by pickle.
    #model_features: (numpy npy file) dictionary including the name of the features used in machine learning, loaded by numpy.
    #model_labels: (numpy npy file) dictionary including the name of the labels used in machine learning, loaded by numpy.
    system: ase.Atoms object
    '''

    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    def __init__(self, restart=None, ignore_bad_restart=False, label='ml_potential', atoms=None, command=None, **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart=ignore_bad_restart, label=label, atoms=atoms, command=command, **kwargs)

        self.current_step = None
        self.log_morest = kwargs['log_file']
        self.if_print_uncertainty = kwargs['ml_parameters']['ml_print_uncertainty']
        self.if_fd_forces = kwargs['ml_parameters']['ml_fd_forces']
        if self.if_fd_forces:
            self.fd_displacement = kwargs['ml_parameters']['fd_displacement']
        try:
            self.trained_ml_potential = kwargs['ml_parameters']['ml_potential_model']
            self.ml_potential = pickle.load(open(self.trained_ml_potential, 'rb'))
        except:
            if 'ml_training_set' in kwargs['ml_parameters']:
                self.training_set = read_xyz_traj(kwargs['ml_parameters']['ml_training_set'])
                self.log_morest.write('Trained ML model has not beed indicated. The ML model will be trained from training set.\n')
                self.ml_potential = self.train_ml_potential()
                if 'ml_gpr_noise_level_bounds' in kwargs['ml_parameters']:
                    tmp_noise_level_bounds = kwargs['ml_parameters']['ml_gpr_noise_level_bounds']
                    if type(tmp_noise_level_bounds) == float:
                        self.noise_level_bounds = np.array([tmp_noise_level_bounds, 1.])
                    elif type(tmp_noise_level_bounds) == np.ndarray:
                        self.noise_level_bounds = np.array([tmp_noise_level_bounds[0], tmp_noise_level_bounds[1]])
                else:
                    self.noise_level_bounds = np.array([1e-7, 1.])
            else:
                raise Exception('ML model or training set can not be read. Please specify the name.')

    def calculate(self, *args, **kwargs):
        Calculator.calculate(self, *args, **kwargs)
        self.results['energy'], self.results['forces'] = self.get_potential_forces(self.atoms)

    def get_ml_potential(self, system_list):
        if type(system_list) != list:
            raise ValueError
        representation_list = [generate_Al2F2_representation(i_system) for i_system in system_list]
        if self.if_fd_forces:
            ml_energy, ml_energy_std = self.ml_potential.predict(representation_list, return_std=True)
            ml_energy = np.array(ml_energy)
            ml_energy_std = np.array(ml_energy_std)
            return ml_energy, ml_energy_std
        else:
            ml_energy_forces, ml_energy_forces_std = self.ml_potential.predict(representation_list, return_std=True)
            ml_energy = np.array(ml_energy_forces[:,0])
            ml_forces = np.array(ml_energy_forces[:,1:]).reshape(-1,3)
            ml_energy_std = np.array(ml_energy_forces_std[:,0])
            ml_forces_std = np.array(ml_energy_forces_std[:,1:]).reshape(-1,3)
            return ml_energy, ml_energy_std, ml_forces, ml_forces_std

    def get_potential_forces(self, system):
        if type(system) == list:
            system = system[0]
        if self.if_fd_forces:
            system_list = [system]
            n_atoms = system.get_global_number_of_atoms()
            forces = []
            for i in range(n_atoms):
                for j in range(3):
                    new_system = deepcopy(system)
                    coordinates = new_system.get_positions()
                    coordinates[i,j] = coordinates[i,j] + self.fd_displacement
                    new_system.set_positions(coordinates)
                    system_list.append(new_system)
            # Get the predictions of energy and uncertainty
            energy_list, energy_std_list = self.get_ml_potential(system_list)
            #print("Energy:", energy_list)
            #print("Energy std:", energy_std_list)
            energy_0 = energy_list[0][0]
            energy_std_0 = energy_std_list[0]
            if self.if_print_uncertainty:
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy: "+str(energy_0)+"\n")
                self.log_morest.write("Current ML energy uncertainty: "+str(energy_std_0)+"\n\n")
            for i,i_energy in enumerate(energy_list[1:]):
                force_value = -1*(i_energy - energy_0)/self.fd_displacement
                forces.append(force_value)
            forces = np.array(forces)
            self.potential_energy = energy_0
            self.forces = forces.reshape(n_atoms, 3)
        else:
            potential_energy, potential_energy_std, forces, forces_std = self.get_ml_potential(system)
            if self.if_print_uncertainty:
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy: "+str(potential_energy)+"\n")
                self.log_morest.write("Current ML energy uncertainty: "+str(potential_energy_std)+"\n\n")
            self.potential_energy = potential_energy
            self.forces = forces
        return self.potential_energy, self.forces
    
    def get_current_step(self, current_step):
        self.current_step = current_step

    @staticmethod
    def RMSE(true, pred):
        true = true.flatten()
        pred = pred.flatten()
        RMSE_value = 0.0
        N = 0
        #print(true, len(true))
        for i in range(len(true)):
            if(math.isnan(true[i])):
                continue
            if(math.isnan(pred[i])):
                continue
            RMSE_value += (pred[i] - true[i]) ** 2
            N += 1
        RMSE_value = math.sqrt(RMSE_value/N)
        return RMSE_value

    def train_ml_potential(self):
        """system_list: The trajectory for training set"""
        #self.log_morest.write("Model is training.\n")
        if len(self.training_set) < 1:
            raise Exception('The training set has no system.')
        x_train = [generate_Al2F2_representation(i_system) for i_system in self.training_set]
        np.savetxt('training_set_representation',x_train)

        potential_energy_list = np.array([i_system.get_potential_energy() for i_system in self.training_set])
        y_train = potential_energy_list
        np.savetxt('training_set_label',y_train)

        gpr_kernel=kernels.Matern(nu=2.5)*kernels.DotProduct(sigma_0=10)  + kernels.WhiteKernel(noise_level=0.1, \
                                                                            noise_level_bounds=(self.noise_level_bounds[0],self.noise_level_bounds[1]))
        self.log_morest.write("Training set:\n\tShape of feature: "+str(np.shape(x_train))+"\n")
        self.log_morest.write("\tShape of label: "+str(np.shape(y_train))+"\n")
        gpr = GaussianProcessRegressor(kernel=gpr_kernel,normalize_y=True)
        gpr.fit(x_train, y_train)
        with open(self.trained_ml_potential,'wb') as trained_model_file:
            pickle.dump(gpr, trained_model_file, protocol=4)
        self.log_morest.write("The trained kernel: "+str(gpr.kernel_)+"\n")

        y_train_pred, y_train_pred_std = gpr.predict(x_train, return_std=True)
        self.log_morest.write("Training RMSE: "+str(self.RMSE(y_train, y_train_pred))+"\n")
        self.log_morest.write("Training uncertainty: "+str(np.average(y_train_pred_std))+"\n")
        self.log_morest.write("Median training uncertainty: "+str(np.median(y_train_pred_std))+"\n\n")

        return gpr

