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
run_mode = 0 # Default compliance minimization

prob.opt.options.max_iter = 400

# Objective and Constraints
if run_mode == 1:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['stress']
    prob.opt.options.outputs_path = 'outputs/Bridge2d_NN_Stress'
    prob.opt.functions.objective_scale = 20
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05
elif run_mode == 0:
    prob.opt.parameters.objective_type = 'compliance'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.45  # Match TOuNN default vf for TipCantilever
    prob.opt.options.outputs_path = 'outputs/Bridge2d_NN_Compliance'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.2
elif run_mode == 2:
    prob.opt.parameters.objective_type = 'stress'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.45 
    prob.opt.options.outputs_path = 'outputs/Bridge2d_NN_MinStress'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05
elif run_mode == 3:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['local_stress']
    prob.opt.options.outputs_path = 'outputs/Bridge2d_NN_LocalALM'
    prob.opt.functions.objective_scale = 20
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.05

# =================================
# ------- NN Hyperparameters ------
prob.opt.nn_params.use_tounn_logic = True
prob.opt.nn_params.hidden_dim = 20
prob.opt.nn_params.num_layers = 5
prob.opt.nn_params.vol_penal_min = 1
prob.opt.nn_params.vol_penal_max = 20
prob.opt.nn_params.activation = 'ReLU'
prob.opt.nn_params.fourier_scale = 1.0
prob.opt.nn_params.use_fourier = 0
prob.opt.nn_params.feature_type =  'positional'
prob.opt.nn_params.learning_rate = 0.01 # 0.01 matches TOuNN
prob.opt.nn_params.optimizer_type = 'Adam'
# =================================

# Setup Mesh for TipCantilever (Match TOuNN)
prob.fe.mesh_input.type = 'generate'
prob.fe.mesh_input.box_dimensions = [60.0, 30.0]
prob.fe.mesh_input.elements_per_side = [60, 30]

prob.opt.options.move_limit = 0.05

# Material (Match TOuNN)
prob.fe.material.E = 1.0
prob.fe.material.nu = 0.3

# Aggregation and Projection Settings
prob.opt.parameters.aggregation_type = 'p-norm'
prob.opt.parameters.aggregation_parameter = 12
prob.opt.parameters.ACS.use = True if run_mode == 1 else False
prob.opt.parameters.projection.use = False
prob.opt.parameters.projection.type = 'heaviside'
prob.opt.parameters.projection.eta = 0.5
prob.opt.parameters.projection.beta_init = 1.0
prob.opt.parameters.projection.beta_final = 30.0

# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.filter_radius_factor = 0.0
prob.opt.stress_needed = 1 if run_mode in [1, 2, 3] else 0
prob.opt.parameters.slimit = 2.4
prob.opt.parameters.init_dens = 0.45

# Output Settings
prob.opt.options.plot = True
prob.opt.options.show_plot = True

solver = FEMSolver(prob)
prob.solver = solver 
prob.fe.mesh_input.bcs_file = 'setup_bridge2d_bcs'
solver.init_FE()
optimizer = Optimizer_Neural(prob)
optimizer.optimization_loop()
