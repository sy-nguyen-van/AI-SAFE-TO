import sys
import os
import numpy as np
import src
sys.path.append(os.getcwd())

from src.struct_prob import Problem
from src.fem import FEMSolver
from src.functions import FunctionEvaluator
from src.setup_bcs import setup_cantilever3d_bcs

prob = Problem()
# 3D Generate
prob.fe.mesh_input.type = 'generate'
# Dimensions matching MATLAB example: 20x4x2
prob.fe.mesh_input.box_dimensions = [20.0, 4.0, 2.0]
# Coarse mesh for speed: [20, 4, 2] -> 160 elements?
# MATLAB: [160 32 16] is very fine. Let's use simpler [20, 4, 2] or even coarser for quick check
# [10, 2, 1] -> element size 2x2x2
# [20, 4, 2] -> element size 1x1x1
prob.fe.mesh_input.elements_per_side = [20, 4, 2]

# Material
prob.fe.material.E = 1.0 # Young's modulus
prob.fe.material.nu = 0.3 # Poisson's ratio
prob.fe.material.rho_min = 1e-3 # Minimum density in void region
prob.fe.material.nu_void = 0.3 # Poisson ratio of the void material

# Valid keys: 'p-norm', 'mrf'
prob.opt.parameters.aggregation_type = 'p-norm'

# Adaptive Constraint Scaling (ACS) should be used with P-norm
prob.opt.parameters.ACS.use = True

# Projection Settings
prob.opt.parameters.projection.use = False
prob.opt.parameters.projection.type = 'heaviside'
prob.opt.parameters.projection.eta = 0.5
prob.opt.parameters.projection.beta_init = 1.0
prob.opt.parameters.projection.beta_final = 25.0

# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.filter_radius_factor = 2.5
prob.opt.options.max_iter = 10 # Longer run for projection check
prob.opt.stress_needed = True # Enable stress calculation check
prob.opt.parameters.slimit = 2.4 # Guess

# Use SIMP
prob.opt.parameters.interpolation_type = 'modified_SIMP'
prob.opt.parameters.penalization_param = 3.0

print("Initializing Solver (3D)...")
solver = FEMSolver(prob)
solver.initialize()

print(f"Mesh: {prob.fe.n_elem} elements, {prob.fe.n_node} nodes.")
assert prob.fe.dim == 3, "Dimension should be 3"

print("Setting up BCs...")
setup_cantilever3d_bcs(prob)
print(f"Fixed DOFs: {len(prob.fe.fixed_dofs)}")
print(f"Force Norm: {np.linalg.norm(prob.fe.F):.4e}")

print("Assembling Stiffness...")
x = np.ones(prob.fe.n_elem) # Solid
K = solver.assemble_stiffness(x, 3.0, 1e-3)

print("Solving...")
u = solver.solve(K, prob.fe.F, prob.fe.fixed_dofs)

evaluator = FunctionEvaluator(prob)
c, _ = evaluator.compute_compliance(x, u)
print(f"Compliance: {c:.4e}")

# Check displacement at tip
# Load applied at (L, H/2, 0)
# Find node near tip
L, H, W = 20.0, 4.0, 2.0
tol = 1e-6
# Tip center
tip_id = -1
min_d = 1e9
coords = prob.fe.coords
for i in range(prob.fe.n_node):
    d = (coords[i,0]-L)**2 + (coords[i,1]-H/2)**2 + (coords[i,2]-0)**2
    if d < min_d:
        min_d = d
        tip_id = i

print(f"Node at tip ({coords[tip_id]}) displacement:")
print(f"Ux: {u[3*tip_id]:.4e}")
print(f"Uy: {u[3*tip_id+1]:.4e}") # Should be negative (down)
print(f"Uz: {u[3*tip_id+2]:.4e}")

if u[3*tip_id+1] < -1e-5:
    print("SUCCESS: Tip deflects downwards as expected.")
else:
    print("FAILURE: Tip displacement unexpected or too small.")

# Stress Check
vm, stress = evaluator.compute_stress(x, u)
print(f"Max VM Stress: {np.max(vm):.4e}")
if vm.shape[0] == prob.fe.n_elem:
    print("SUCCESS: Stress shape correct.")
else:
    print("FAILURE: Stress shape mismatch.")


# 


solver = FEMSolver(prob)
solver.initialize()



# For testing, use full density
x = 0.5*np.ones(prob.fe.n_elem)


# Setup BCs for L-bracket or default rect
setup_cantilever3d_bcs(prob)
# Assemble stiffness
p = prob.opt.parameters.penalization_param
rho_min = prob.opt.parameters.rho_min
K = solver.assemble_stiffness(x, p, rho_min)
prob.fe.K = K 

# Solve
u = solver.solve(K, prob.fe.F, prob.fe.fixed_dofs)
np.linalg.norm(u)
# Evaluate
evaluator = FunctionEvaluator(prob)
vf, _ = evaluator.compute_volume_fraction(x)
c, _ = evaluator.compute_compliance(x, u)
vm_stress, _ = evaluator.compute_stress(x, u)

print("-" * 30)
print(f"Results for {prob.fe.mesh_input.type}:")
print(f"  Volume Fraction: {vf:.4f}")
print(f"  Compliance:      {c:.4e}")
print(f"  Max Von Mises:   {np.max(vm_stress):.4e}")
print("-" * 30)

print(f"L-bracket: {prob.fe.n_node} nodes, {prob.fe.n_elem} elems")