
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current dir to path to find src
sys.path.append(os.getcwd())

from src import Problem, FEMSolver
from src.fem_torch import TorchFEM
from src.utils import plot_density, plot_stress

print(">>> Testing L-bracket with Differentiable FEM (TorchFEM)")

# 1. Setup Problem Config (Same as before)
prob = Problem()

# Mesh
prob.fe.mesh_input.type = '2DLbracket'
prob.fe.mesh_input.L_side = 100.0
prob.fe.mesh_input.L_cutout = 60.0
prob.fe.mesh_input.L_element_size = 2.0 # Mesh size
prob.fe.mesh_input.bcs_file = 'setup_lbracket_bcs' # Ensure BCs are loaded

# Material
prob.fe.material.E = 1.0 # Young's modulus
prob.fe.material.nu = 0.3 # Poisson's ratio

# Optim Params
prob.opt.parameters.penalization_param = 3.0
prob.opt.parameters.slimit = 2.4 

# 2. Initialize Standard FEM (Required for Mesh Generation)
print("   Initializing Mesh...")
solver_std = FEMSolver(prob)
solver_std.init_FE() # Generates mesh, sets prob.fe

# 3. Initialize TorchFEM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using Device: {device}")

# Enable double precision for stability
torch.set_default_dtype(torch.float64)

ad_fem = TorchFEM(prob, device=device)

# 4. Design Variable (Density) with Gradients
# x = 0.5 * ones
x_np = 0.5 * np.ones(prob.fe.n_elem)
x = torch.tensor(x_np, dtype=torch.float64, device=device, requires_grad=True)

# 5. Connect Physics (Forward Pass)
print("   Running Forward Analysis...")
penal = prob.opt.parameters.penalization_param

# K(x) * u = F
u = ad_fem.solve(x, penal=penal)

# 6. Compute Objectives & Gradients via AD
print("   Computing Objectives & Gradients...")

# A. Volume Fraction
# Need to retain graph? No, we can re-run or just do separate passes if we want distinct grads.
# We'll compute them sequentially and clear grads or accumulate?
# Let's do clear grad approach for norms.

# --- Volume ---
if x.grad is not None: x.grad.zero_()
vol_frac = torch.sum(x * ad_fem.elem_vol) / torch.sum(ad_fem.elem_vol)
vol_frac.backward(retain_graph=True) # Retain for other calcs if shared graph

vf_val = vol_frac.item()
grad_vf_norm = x.grad.norm().item()
x.grad.zero_() # Reset

# --- Compliance ---
c = ad_fem.compute_compliance(u, x)
c.backward(retain_graph=True)

c_val = c.item()
grad_c_norm = x.grad.norm().item()
x.grad.zero_() # Reset

# --- Stress ---
# compute_max_stress returns (aggregated_stress, true_max_stress)
# We differentiate the aggregated stress P-norm
s_pn, s_max = ad_fem.compute_max_stress(u, x, penal=penal, q=0.5)

# If we want gradient of "Stress Violation" or just Stress P-Norm?
# User snippet computed gradient of "max stress violation".
# Since stress violation is usually relu(s/limit - 1), but let's just do gradient of P-norm for indication.
# The snippet used `evaluator.compute_max_stress_violation`.

# Let's match the violation gradient concept:
# limit = 2.4
limit = prob.opt.parameters.slimit
# Approximate violation using s_pn (differentiable)
# This might differ slightly from FunctionEvaluator which uses analytic aggregation derivatives

g_viol = s_pn / limit - 1.0 # Simple scaling
g_viol.backward()

# Note: If g_viol < 0, grad is propogated. ReLU would kill it.
# FunctionEvaluator usually aggregates the *positive* parts via P-Norm.

g_val = g_viol.item() # This is the P-norm violation attempt
grad_g_norm = x.grad.norm().item()

# 7. Print Results
print("-" * 30)
print(f"Results for {prob.fe.mesh_input.type} (TorchFEM AD):")
print(f"  Volume Fraction: {vf_val:.4f}")
print(f"  |Grad| VF:       {grad_vf_norm:.4e}")
print(f"  Compliance:      {c_val:.4e}")
print(f"  |Grad| C:        {grad_c_norm:.4e}")
print(f"  Max Von Mises:   {s_max.item():.4e}") # Computed inside function
print(f"  Agg Stress Viol: {g_val:.4f}")
print(f"  |Grad| S_viol:   {grad_g_norm:.4e}")
print("-" * 30)

# 8. Plotting
# Move to CPU/Numpy
x_plot = x.detach().cpu().numpy()

# Need to compute stress field for plotting
if hasattr(ad_fem, 'compute_von_mises'):
     vm_stress_tensor = ad_fem.compute_von_mises(u, x, penal=penal, q=0.5)
     vm_stress = vm_stress_tensor.detach().cpu().numpy().flatten()
else:
     vm_stress = np.zeros(prob.fe.n_elem)

plot_density(prob.fe.coords, prob.fe.elem_node, x_plot, show=True)
plot_stress(prob.fe.coords, prob.fe.elem_node, vm_stress, show=True)
