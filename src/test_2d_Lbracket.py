
import sys
import os

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

# Add current dir to path to find src
sys.path.append(os.getcwd())

from src import Problem, FEMSolver, FunctionEvaluator
from src.utils import compute_filter_matrix, plot_density, plot_stress

prob = Problem()
prob.fe.make_fd_check = True
prob.fe.fd_step_size = 1e-8 # Step size for FD check
# Mesh
prob.fe.mesh_input.type = '2DLbracket'
prob.fe.mesh_input.L_side = 100.0
prob.fe.mesh_input.L_cutout = 60.0
prob.fe.mesh_input.L_element_size = 2 # Mesh size

# Material
prob.fe.material.E = 1.0 # Young's modulus
prob.fe.material.nu = 0.3 # Poisson's ratio

# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.slimit = 2.4 

solver = FEMSolver(prob)
prob.solver = solver # Link solver to problem for FunctionEvaluator usage



# Setup BCs for L-bracket via init_FE
prob.fe.mesh_input.bcs_file = 'setup_lbracket_bcs'

# Use init_FE
solver.init_FE()

# Compute centroids
centroids = np.mean(prob.fe.coords[prob.fe.elem_node], axis=1)
# Initialize filter matrix H (radius factor 2.5)
radius = 2.5 * prob.fe.max_elem_side
prob.opt.H = compute_filter_matrix(centroids.T, radius)

# For testing, use full density
x = 0.5*np.ones(prob.fe.n_elem)

# 2. Assemble Stiffness and Solve via FE_analysis
solver.FE_analysis(x, analysis_type='primal')

u = prob.fe.U

# Evaluate
evaluator = FunctionEvaluator(prob)
vf, grad_vf = evaluator.compute_volume_fraction(x)
c, grad_c = evaluator.compute_compliance(x, u)

# Norm of Gradients
norm_grad_vf = np.linalg.norm(grad_vf)
norm_grad_c = np.linalg.norm(grad_c)
print(f"Gradient norm of volume fraction: {norm_grad_vf}")
print(f"Gradient norm of compliance: {norm_grad_c}")
# Stress calculated inside compute_max_stress_violation
g, grad_g = evaluator.compute_max_stress_violation(x, u)
norm_grad_g = np.linalg.norm(grad_g)

print(f"Gradient norm of stress violation: {norm_grad_g}")

if hasattr(prob.fe, 'svm') and prob.fe.svm is not None:
    vm_stress = prob.fe.svm.flatten() # Flatten for plotting
else:
    vm_stress = np.zeros(prob.fe.n_elem)

print("-" * 30)
print(f"Results for {prob.fe.mesh_input.type}:")
print(f"  Volume Fraction: {vf:.4f}")
print(f"  Compliance:      {c:.4e}")
print(f"  Max Von Mises:   {np.max(vm_stress):.4e}")
print("-" * 30)

plot_density(prob.fe.coords, prob.fe.elem_node, x, show=True)
plot_stress(prob.fe.coords, prob.fe.elem_node, vm_stress, show=True)
plt.show()  # Keep plots open
