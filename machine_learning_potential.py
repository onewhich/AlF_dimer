import numpy as np
import pickle
from copy import deepcopy
import math
from ase.io import read
from ase.calculators.calculator import Calculator
from representation import generate_Al2F2_representation
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_absolute_error as MAE


class ml_potential(Calculator):
    '''
    Train the machine-learning potential, and implement to a ASE calculator. Predictions of energies can be made by the calculator.
    If required, forces are calculated by finite difference of energies.
    '''

    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    def __init__(self, restart=None, ignore_bad_restart=False, label='ml_potential', atoms=None, command=None, **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart=ignore_bad_restart, label=label, atoms=atoms, command=command, **kwargs)

        self.current_step = None
        self.log = kwargs['log_file']
        self.fd_displacement = kwargs['ml_parameters']['fd_displacement']
        try:
            self.trained_ml_potential = kwargs['ml_parameters']['ml_potential_model']
            self.ml_potential = pickle.load(open(self.trained_ml_potential, 'rb'))
        except:
            if 'ml_training_set' in kwargs['ml_parameters']:
                self.training_set = read(kwargs['ml_parameters']['ml_training_set'], index=':', format='extxyz')
                if 'ml_gpr_noise_level_bounds' in kwargs['ml_parameters']:
                    tmp_noise_level_bounds = kwargs['ml_parameters']['ml_gpr_noise_level_bounds']
                    if type(tmp_noise_level_bounds) == float:
                        self.noise_level_bounds = np.array([tmp_noise_level_bounds, 1.])
                    elif type(tmp_noise_level_bounds) == np.ndarray:
                        self.noise_level_bounds = np.array([tmp_noise_level_bounds[0], tmp_noise_level_bounds[1]])
                else:
                    self.noise_level_bounds = np.array([1e-7, 1.])
                self.log.write('Trained ML model has not beed indicated. The ML model will be trained from training set.\n')
                self.ml_potential = self.train_ml_potential()
            else:
                raise Exception('ML model or training set can not be read. Please specify the name.')

    def calculate(self, *args, **kwargs):
        """
        Set ASE calculator.
        """
        Calculator.calculate(self, *args, **kwargs)
        self.results['energy'], self.results['forces'] = self.get_potential_forces(self.atoms)

    def get_ml_energy(self, system_list):
        """
        Get the machine-learned potential energies of the configurations.

        Input:
        system_list: The training set as a trajectory

        Returns:
        ml_energy: Energies predicted by the machine-learning model
        ml_energy_std: Uncertainty of energy predictions

        """
        if type(system_list) != list:
            raise ValueError

        # Generate the structural representation
        representation_list = [generate_Al2F2_representation(i_system) for i_system in system_list]

        # Predict the potential energy by the machine-learning potential
        ml_energy, ml_energy_std = self.ml_potential.predict(representation_list, return_std=True)
        ml_energy = np.array(ml_energy)
        ml_energy_std = np.array(ml_energy_std)
        return ml_energy, ml_energy_std

    def get_potential_forces(self, system):
        """
        Calculate the energy and force of a system.

        Input:
        system: An ASE Atoms object

        Returns:
        potential_energy: Potential energy of the system predicted by the machine-learning model
        forces: Forces of the system from finite difference
        """

        if type(system) == list:
            system = system[0]
        system_list = [system]
        n_atoms = system.get_global_number_of_atoms()
        forces = []

        # Generate the coodinates for finite difference
        for i in range(n_atoms):
            for j in range(3):
                new_system = deepcopy(system)
                coordinates = new_system.get_positions()
                coordinates[i,j] = coordinates[i,j] + self.fd_displacement
                new_system.set_positions(coordinates)
                system_list.append(new_system)
        # Get the predictions of energy and uncertainty, for the system and also finite difference
        energy_list, energy_std_list = self.get_ml_energy(system_list)
        energy_0 = energy_list[0]
        energy_std_0 = energy_std_list[0]
        #self.log.write("Current sampling step: "+str(self.current_step)+"\n")
        #self.log.write("Current ML energy: "+str(energy_0)+"\n")
        
        # Calculation of force from finite difference of energy
        for i,i_energy in enumerate(energy_list[1:]):
            force_value = -1*(i_energy - energy_0)/self.fd_displacement
            forces.append(force_value)
        forces = np.array(forces)
        self.potential_energy = energy_0
        self.forces = forces.reshape(n_atoms, 3)
        return self.potential_energy, self.forces
    
    def get_current_step(self, current_step):
        self.current_step = current_step

    @staticmethod
    def RMSE(true, pred):
        true = np.array(true).flatten()
        pred = np.array(pred).flatten()
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
        """
        Train the machine-learning model for potential energy

        Returns:
        The Gaussian-process regression (GPR) model of potential energy surface

        """
        if len(self.training_set) < 1:
            raise Exception('The training set is empty.')

        # Generate the structural representation for the configurations
        x_train = [generate_Al2F2_representation(i_system) for i_system in self.training_set]
        # np.savetxt('training_set_representation',x_train)

        # Get the labels for the training
        potential_energy_list = np.array([i_system.get_potential_energy() for i_system in self.training_set])
        y_train = potential_energy_list
        # np.savetxt('training_set_energies',y_train)

        # Set the GPR kernel
        # The core is the Matern(5/2) kernel, but the dot product kernel can also improve a bit the prediction (~<10%)
        # The noise level should be set according to the noise of the training set, which is typically very small for quantum chemistry predictions.
        gpr_kernel=kernels.Matern(nu=2.5)*kernels.DotProduct(sigma_0=10)  + kernels.WhiteKernel(noise_level=0.1, \
                                                                            noise_level_bounds=(self.noise_level_bounds[0],self.noise_level_bounds[1]))

        self.log.write("Training set:\n\tShape of feature: "+str(np.shape(x_train))+"\n")
        self.log.write("\tShape of label: "+str(np.shape(y_train))+"\n")
        self.log.flush()

        gpr = GaussianProcessRegressor(kernel=gpr_kernel,normalize_y=True)
        
        # Train the model
        #print("Training model...", flush=True)
        gpr.fit(x_train, y_train)
        self.log.write("Model trained.\n")
        self.log.write("The trained kernel: "+str(gpr.kernel_)+"\n")
        self.log.flush()

        # Make preidctions for the training set
        y_train_pred, y_train_pred_std = gpr.predict(x_train, return_std=True)
        self.log.write("Training MAE: " + str(MAE(y_train, y_train_pred)/4.0)+" eV/atom\n")
        self.log.write("Training RMSE: "+ str(self.RMSE(y_train, y_train_pred)/4.0)+" eV/atom\n")
        self.log.write("Training uncertainty: " + str(np.average(y_train_pred_std))+"\n")
        self.log.write("Median training uncertainty: " + str(np.median(y_train_pred_std))+"\n\n")
        self.log.flush()
       

        # Write the model
        with open(self.trained_ml_potential,'wb') as trained_model_file:
            pickle.dump(gpr, trained_model_file, protocol=4)
        self.log.write("Trained PES model saved to "+str(self.trained_ml_potential)+"\n")
        return gpr

