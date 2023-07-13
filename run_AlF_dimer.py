import numpy as np
from ase.io import read
from machine_learning_potential import ml_potential
from sklearn.metrics import mean_absolute_error as MAE

"""
Generate the machine-learning potential and make predictions for the test set

The training/test sets are read from data/traj_train.xyz and data/traj_test.xyz, respectively.
The training and test sets should be trajectory files in extxyz format.
The training/test predictions are output to 
The summary of the results can be printed to Alf_dimer.log.

Parameters:
fd_displacement: Displacement (in Angstrom) of the finite-difference calculation of the force
ml_gpr_noise_level_bounds: Boundary of the Gaussian process regression white noise kernal 
                           It should be very small (e.g. 1e-7) if the training set is generated by high-accuracy quantum chemistry methods.

ml_potential_model: The trained potential energy model
                    If the model has been trained already, then the trained model can be read and used for predictions.
                    If not, then it will be created.

"""

training_set_filename = 'data/traj_train.xyz'
test_set_filename = 'data/traj_test.xyz'
log_filename = 'AlF_dimer.log'
test_set_prediction_filename = 'AlF_dimer_test_set_prediction.txt'
trained_ml_potential_model = 'trained_ml_potential_model.pkl'

log_file = open(log_filename,'w')
test_set = read(test_set_filename, index=':', format='extxyz')

ml_parameters = {'fd_displacement': 0.0025, 
                 'ml_potential_model': trained_ml_potential_model,
                 'ml_training_set': training_set_filename,
                 'ml_gpr_noise_level_bounds': 1e-07}
ml_calculator = ml_potential(ml_parameters = ml_parameters, log_file = log_file)

# Train/read the model and make predictions 
test_set_prediction = []
test_set_true = []
for i,i_sys in enumerate(test_set):
    # Read the potential energy (if exist) from the trajectory
    test_set_true.append(i_sys.get_potential_energy())
    # In the first loop, the ml_potential will be trained/read during initialization. Then it will be used for later predictions.
    ml_calculator.get_current_step(i)
    i_sys.calc = ml_calculator
    # Make test-set prediction
    energy = i_sys.get_potential_energy()
    test_set_prediction.append(energy)

log_file.write("\nTest-set MAE: " + str(MAE(test_set_true, test_set_prediction)/4.0) + " eV/atom")
log_file.write("\nTest-set RMSE: " + str(ml_potential.RMSE(test_set_true, test_set_prediction)/4.0) + "eV/atom")

np.savetxt(test_set_prediction_filename, test_set_prediction)




