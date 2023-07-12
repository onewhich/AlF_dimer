from ase.io import read
from many_body_potential import ml_potential

test_set_name = 'traj_test.xyz'

ml_parameters = {'ml_print_uncertainty': True, 'ml_fd_forces': True, 'fd_displacement': 0.0025, 
                 'ml_potential_model': 'trained_ml_potential_model.pkl',
                 'ml_training_set': 'traj_train.xyz'}
log_file = 'AlF_dimer.log'

test_set = read(test_set_name, index=':', format='extxyz')

for i,i_sys in enumerate(test_set):
    ml_calculator = ml_potential(ml_parameters = ml_parameters, log_file = log_file)
    ml_calculator.get_current_step(i)
    i_sys.calc = ml_calculator
    print(i, i_sys.get_potential_energy())
