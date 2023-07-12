import numpy as np

def generate_Al2F2_representation(Al2F2, representation_name="inverse_r_exp_r_relative"):
    """
        Al2F2: ASE Atoms object, with positions information of Al2F2
        representation_name: Name of the structural representation:
                             Values:
                                   inverse_r_exp_r: 1/r_{ij}, 1/R_{ij}, exp(-r_{ij}), exp(-R), where R is the largest AlF-AlF distance.
                                   inverse_r_exp_r_relative: Same as inverse_r_exp_r, except r_{ij}=r_{ij}/r_{ij,optimized}, 
                                                             where the interatomic distances are normalized by the equilibrium interatomic
                                                             distances of diatomic molecule ij, optimized by ab intio methods, e.g. CCSD(T).
    """
    Al1 = Al2F2.get_positions()[0] # change length in AA to Bohr
    F2 = Al2F2.get_positions()[1]
    Al3 = Al2F2.get_positions()[2]
    F4 = Al2F2.get_positions()[3]
    
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
        r_F2 = 1.413030165
        r_Al2 = 2.568675944
        r_AlF = 1.668734698
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

