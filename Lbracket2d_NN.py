import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Fix OpenMP conflict (PyTorch vs NumPy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add current dir to path to find src
sys.path.append(os.getcwd())

from src import Problem, FEMSolver, FunctionEvaluator, Optimizer_Neural
from src.utils import plot_density, compute_filter_matrix, plot_stress, plot_history

prob = Problem()
# run_mode options:
# 0 = Minimize compliance, Constraint: volume
# 1 = Minimize volume, Constraint: max stress
# 2 = Minimize max stress, Constraint: volume
# 3 = Minimize volume, Constraint: local ALM stress
run_mode = 1
prob.opt.options.max_iter = 200 # Longer run for projection check
# Objective and Constraints
if run_mode == 1:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['stress']
    prob.opt.options.outputs_path = 'outputs/Lbracket_Stress'
    prob.opt.functions.objective_scale = 20
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05  # Move limit
elif run_mode == 0:
    prob.opt.parameters.objective_type = 'compliance'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.4 
    prob.opt.options.outputs_path = 'outputs/Lbracket_Compliance'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.2  # Move limit     
elif run_mode == 2:
    prob.opt.parameters.objective_type = 'stress'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.3 
    prob.opt.options.outputs_path = 'outputs/Lbracket_MinStress'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05
elif run_mode == 3:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['local_stress']
    prob.opt.options.outputs_path = 'outputs/Lbracket_LocalALM'
    prob.opt.functions.objective_scale = 20
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05
# =================================
# ------- NN Hyperparameters ------
prob.opt.nn_params.use_tounn_logic = True
prob.opt.nn_params.hidden_dim = 32
prob.opt.nn_params.num_layers = 4
prob.opt.nn_params.vol_penal_min = 1
prob.opt.nn_params.vol_penal_max = 20
prob.opt.nn_params.activation = 'ReLU'
prob.opt.nn_params.fourier_scale = 1.0
# =================================
prob.opt.nn_params.use_fourier = 0
prob.opt.nn_params.feature_type =  'positional'
prob.opt.nn_params.learning_rate = 0.01
prob.opt.nn_params.optimizer_type = 'Adam'
# =================================
prob.fe.mesh_input.type = '2DLbracket'
prob.fe.mesh_input.L_side = 100.0
prob.fe.mesh_input.L_cutout = 60.0
prob.fe.mesh_input.L_element_size = 1 # Mesh size
prob.opt.options.move_limit = 0.05  # Move limit 
# Material
prob.fe.material.E = 1.0 # Young's modulus
prob.fe.material.nu = 0.3 # Poisson's ratio
# Valid keys: 'p-norm', 'mrf'
prob.opt.parameters.aggregation_type = 'p-norm'
prob.opt.parameters.aggregation_parameter = 12
# Adaptive Constraint Scaling (ACS) should be used with P-norm
prob.opt.parameters.ACS.use = True if run_mode == 1 else False
# Projection Settings
prob.opt.parameters.projection.use = False
prob.opt.parameters.projection.type = 'heaviside'
prob.opt.parameters.projection.eta = 0.5
prob.opt.parameters.projection.beta_init = 1.0
prob.opt.parameters.projection.beta_final = 30.0
# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.filter_radius_factor = 0.0
prob.opt.stress_needed = 1 if run_mode in [1, 2, 3] else 0 # Enable stress calculation check
prob.opt.parameters.slimit = 2.4 # Stress limit
prob.opt.parameters.init_dens = 0.5 # Initial density
# Output Settings
prob.opt.options.plot = True
prob.opt.options.show_plot = True
# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.slimit = 2.4 
solver = FEMSolver(prob)
prob.solver = solver 
prob.fe.mesh_input.bcs_file = 'setup_lbracket_bcs'
solver.init_FE()
optimizer = Optimizer_Neural(prob)
optimizer.optimization_loop() 
