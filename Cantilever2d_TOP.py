import numpy as np
import sys
import os

# Fix OpenMP conflict (PyTorch vs NumPy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add current dir to path to find src
sys.path.append(os.getcwd())

from src import Problem, FEMSolver, FunctionEvaluator, Optimizer_global, Optimizer_local
from src.utils import plot_density, compute_filter_matrix, plot_stress, plot_history

# ===============================================
prob = Problem()

# run_mode options:
# 0 = Minimize compliance, Constraint: volume
# 1 = Minimize volume, Constraint: max stress
# 2 = Minimize max stress, Constraint: volume
# 3 = Minimize volume, Constraint: local ALM stress
run_mode = 0 # Match TOuNN default compliance minimization

# Objective and Constraints
prob.opt.options.max_iter = 100
if run_mode == 1:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['stress']
    prob.opt.options.outputs_path = 'outputs/Cantilever2d_Stress'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.02
    prob.opt.parameters.slimit = 2.4 
    prob.opt.parameters.relaxation_param = 0.5
    prob.opt.parameters.ACS.use = True
    prob.opt.parameters.ACS.alpha_osc = 0.8
    prob.opt.parameters.ACS.alpha_no_osc = 1
    prob.opt.parameters.ACS.c = []
elif run_mode == 0:
    prob.opt.parameters.objective_type = 'compliance'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.6  # TOuNN default vf for TipCantilever
    prob.opt.options.outputs_path = 'outputs/Cantilever2d_Compliance'
    prob.opt.functions.objective_scale = 1.0e-2
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.2
elif run_mode == 2:
    prob.opt.parameters.objective_type = 'stress'
    prob.opt.parameters.constraint_types = ['volume']
    prob.opt.parameters.target_volume = 0.6 
    prob.opt.options.outputs_path = 'outputs/Cantilever2d_MinStress'
    prob.opt.functions.objective_scale = 1.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.02
elif run_mode == 3:
    prob.opt.parameters.objective_type = 'volume'
    prob.opt.parameters.constraint_types = ['local_stress']
    prob.opt.options.outputs_path = 'outputs/Cantilever2d_LocalALM'
    prob.opt.functions.objective_scale = 40.0
    prob.opt.functions.constraint_scale = 1.0
    prob.opt.options.move_limit = 0.02
    prob.opt.parameters.slimit = 2.4 
    prob.opt.parameters.vol_penal_min = 0.1
    prob.opt.parameters.vol_penal_max = 5.0

# ===============================================
# Setup Mesh for TipCantilever (Match TOuNN)
prob.fe.mesh_input.type = 'generate'
prob.fe.mesh_input.box_dimensions = [60.0, 30.0]
prob.fe.mesh_input.elements_per_side = [60, 30]

# Material (Match TOuNN)
prob.fe.material.E = 1.0 # Young's modulus
prob.fe.material.nu = 0.3 # Poisson's ratio

# Aggregation settings
prob.opt.parameters.aggregation_type = 'p-norm'
prob.opt.parameters.aggregation_parameter = 10
prob.opt.parameters.ACS.use = True if run_mode == 1 else False

# Projection Settings
prob.opt.parameters.projection.use = True
prob.opt.parameters.projection.type = 'heaviside'
prob.opt.parameters.projection.eta = 0.5
prob.opt.parameters.projection.beta_init = 1.0
prob.opt.parameters.projection.beta_final = 30.0

# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.filter_radius_factor = 2.5

prob.opt.stress_needed = 1 if run_mode in [1, 2, 3] else 0
prob.opt.parameters.slimit = 2.4
prob.opt.parameters.init_dens = 0.6 # Initial density matching volume fraction

# MMA Parameters
prob.opt.parameters.mma.version = 2007
n_cons = len(prob.opt.parameters.constraint_types)
prob.opt.parameters.mma.c = 1000.0 * np.ones(n_cons)
prob.opt.parameters.mma.d = 1.0 * np.ones(n_cons)
prob.opt.parameters.mma.a0 = 1.0
prob.opt.parameters.mma.a = np.zeros(n_cons)
           
# Output Settings
prob.opt.options.plot = True
prob.opt.options.show_plot = True

solver = FEMSolver(prob)
prob.solver = solver 

# Setup BCs
prob.fe.mesh_input.bcs_file = 'setup_cantilever2d_bcs'

# Init FE
solver.init_FE()

# Compute centroids and filter matrix
centroids = np.mean(prob.fe.coords[prob.fe.elem_node], axis=1)
radius = 2.5 * prob.fe.max_elem_side
prob.opt.H = compute_filter_matrix(centroids.T, radius)

# Initialize Optimizer
if run_mode == 3:
    optimizer = Optimizer_local(prob)
else:
    optimizer = Optimizer_global(prob)

# Run Optimization
optimizer.optimization_loop()
