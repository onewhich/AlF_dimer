from many_body_potential import ml_potential

ml_parameters = {'ml_print_uncertainty': True, 'ml_fd_forces': True, 'fd_displacement': 0.0025, 
                 'ml_potential_model': 'trained_ml_potential_model.pkl',
                 'ml_training_set': 'training_set.xyz'}
log_file = 'AlF_dimer.log'