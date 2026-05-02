
"""
Script to verify gradients for L-bracket example
"""
import sys
import os

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.struct_prob import Problem
from src.optim_mma import Optimizer
from src.verification import verify_gradients

# =======================================
# Setup problem exactly like lbracket_2d.py
prob = Problem()

# Mesh Gen
prob.fe.mesh_input.type = '2DLbracket'
prob.fe.mesh_input.L_side = 100.0
prob.fe.mesh_input.L_cutout = 60.0
prob.fe.mesh_input.L_element_size = 5 # Mesh size

# Material
prob.fe.material.E = 1.0
prob.fe.material.nu = 0.3

# P-norm and ACS settings (as currently active)
prob.opt.parameters.aggregation_type = 'p-norm'
prob.opt.parameters.interpolation_type = 'modified_simp'
prob.opt.parameters.rho_min = 1e-3

# Projection Settings
prob.opt.parameters.projection.use = True
prob.opt.parameters.projection.type = 'heaviside'
prob.opt.parameters.projection.eta = 0.5
prob.opt.parameters.projection.beta_init = 1.0
prob.opt.parameters.projection.beta_final = 30.0

prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.filter_radius_factor = 2.5
prob.opt.parameters.relaxation_param = 2.5

# Configure BCs for init_FE
prob.fe.mesh_input.bcs_file = 'setup_lbracket_bcs'

# Init Optimizer
optimizer = Optimizer(prob)

# Ensure n_node is set (it should be by initialize -> generate_mesh)

print("Running Gradient Verification...")

# NOTE: Only running finite difference check. No optimization loop.
# Check 20 variables
output_path = 'outputs/2DLbracket/gradients'
verify_gradients(optimizer, num_checks=20, step=1e-6, output_dir=output_path)

print(f"Check complete. See plots in '{output_path}'.")

