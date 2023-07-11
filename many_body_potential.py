import numpy as np
import pickle
from copy import deepcopy
from ase import units
from structure_io import read_xyz_traj, write_xyz_traj
import subprocess
import os
import math
from collective_variable import collective_variables
from ase.calculators.calculator import Calculator, FileIOCalculator
from representation import generate_representation
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
        self.if_fd_forces = True
        self.fd_displacement = kwargs['ml_parameters']['fd_displacement']
        if kwargs['ml_parameters']['ml_additional_features'] == None:
            self.additional_features = None
        else:
            self.additional_features = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features'])
        if kwargs['ml_parameters']['ml_additional_features_min'] == None:
            self.additional_features_min = None
        else:
            self.additional_features_min = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features_min'])
        if kwargs['ml_parameters']['ml_additional_features_max'] == None:
            self.additional_features_max = None
        else:
            self.additional_features_max = collective_variables(CVs_list=kwargs['ml_parameters']['ml_additional_features_max'])
        try:
            self.trained_ml_potential = kwargs['ml_parameters']['ml_potential_model']
            self.ml_potential = pickle.load(open(self.trained_ml_potential, 'rb'))
        except:
            raise Exception('ML model can not be read. Please specify the name.')

    def calculate(self, *args, **kwargs):
        Calculator.calculate(self, *args, **kwargs)
        self.results['energy'], self.results['forces'] = self.get_potential_forces(self.atoms)

    def get_ml_potential(self, system_list):
        #if type(system_list) != list:
        #    raise ValueError
        #representation_list = [generate_representation.generate_Al2F2_representation(i_system) for i_system in system_list]
        representation_list = generate_representation(system_list).inverse_r_exp_r()
        if self.additional_features == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features.generate_CVs_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
        if self.additional_features_min == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features_min.generate_CV_min_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
        if self.additional_features_max == None:
            representation_list = representation_list
        else:
            addional_features_list = self.additional_features_max.generate_CV_max_list(system_list)
            representation_list = np.hstack((representation_list,addional_features_list))
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
            # Determine if the energy need to be calculated on the fly
            if self.if_active_learning and (energy_std_0 > self.energy_uncertainty_tolerance):
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy uncertainty is larger than tolerance(="+str(self.energy_uncertainty_tolerance)+"): "+str(energy_std_0)+"\n")
                self.log_morest.write("The relevant ML predicted potential energy: "+str(energy_0)+"\n")
                self.log_morest.write("Current system:\n")
                chemical_symbols = system.get_chemical_symbols()
                coordinates = system.get_positions()
                for i in range(len(coordinates)):
                    self.log_morest.write(chemical_symbols[i]+" "+str(coordinates[i][0])+" "+str(coordinates[i][1])+" "+str(coordinates[i][2])+"\n")
                #return float('nan'), float('nan')
                # If the ML energy has too large uncertainty, call ab initio calculations
                self.potential_energy, self.forces = self.ab_initio_potential.get_potential_forces(system)
                self.log_morest.write("The relevant ab initio potential energy: "+str(self.potential_energy)+"\n")
                write_xyz_traj(self.filename_training_set, system)
                self.training_set = read_xyz_traj(self.filename_training_set)
                self.log_morest.write("The current system has been added to the training set.\n\n")
                self.appending_set_counter += 1
                if self.appending_set_counter == self.appending_set_number:
                    self.log_morest.write("Start to train a new model:\n")
                    self.ml_potential = self.train_ml_potential()
                    self.appending_set_counter = 0
            else:
                if self.if_print_uncertainty:
                    self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                    self.log_morest.write("Current ML energy uncertainty: "+str(energy_std_0)+"\n\n")
                for i,i_energy in enumerate(energy_list[1:]):
                    force_value = -1*(i_energy - energy_0)/self.fd_displacement
                    forces.append(force_value)
                forces = np.array(forces)
                self.potential_energy = energy_0
                self.forces = forces.reshape(n_atoms, 3)
                #print('Predicted energy: ',energy_0)
                #print('Std error of the predicted energy: ',energy_std_0)
                #print('\n')
        else:
            potential_energy, potential_energy_std, forces, forces_std = self.get_ml_potential(system)
            #TODO: the RMSE of forces prediction is not used for judgment
            if self.if_active_learning and (potential_energy_std > self.energy_uncertainty_tolerance):
                self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
                self.log_morest.write("Current ML energy uncertainty is larger than tolerance(="+str(self.energy_uncertainty_tolerance)+"): "+str(potential_energy_std)+"\n")
                self.log_morest.write("The relevant ML predicted potential energy: "+str(potential_energy)+"\n")
                self.log_morest.write("Current ML forces uncertainty is: "+str(forces_std.flatten())+"\n")
                self.log_morest.write("The relevant ML predicted forces: "+str(forces.flatten())+"\n")
                self.log_morest.write("Current system:\n")
                chemical_symbols = system.get_chemical_symbols()
                coordinates = system.get_positions()
                for i in range(len(coordinates)):
                    self.log_morest.write(chemical_symbols[i]+" "+str(coordinates[i][0])+" "+str(coordinates[i][1])+" "+str(coordinates[i][2])+"\n")
                # If the ML energy has too large uncertainty, call ab initio calculations
                self.potential_energy, self.forces = self.ab_initio_potential.get_potential_forces(system)
                self.log_morest.write("The relevant ab initio potential energy: "+str(self.potential_energy)+"\n")
                write_xyz_traj(self.filename_training_set, system)
                self.training_set = read_xyz_traj(self.filename_training_set)
                self.log_morest.write("The current system has been added to the training set.\n\n")
                self.appending_set_counter += 1
                if self.appending_set_counter == self.appending_set_number:
                    self.log_morest.write("Start to train a new model:\n")
                    self.ml_potential = self.train_ml_potential()
                    self.appending_set_counter = 0
            else:
                if self.if_print_uncertainty:
                    self.log_morest.write("Current sampling step: "+str(self.current_step)+"\n")
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
        x_train = generate_representation(self.training_set).inverse_r_exp_r()
        if self.additional_features == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features.generate_CVs_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
        if self.additional_features_min == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features_min.generate_CV_min_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
        if self.additional_features_max == None:
            x_train = x_train
        else:
            addional_features_list = self.additional_features_max.generate_CV_max_list(self.training_set)
            x_train = np.hstack((x_train,addional_features_list))
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

