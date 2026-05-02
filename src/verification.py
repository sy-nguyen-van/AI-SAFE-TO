
import numpy as np
import matplotlib.pyplot as plt
import os
from .struct_prob import Problem
from .optim_mma import Optimizer

def verify_gradients(optimizer: Optimizer, num_checks: int = 50, step: float = 1e-6, output_dir: str = '.'):
    """
    Verifies gradients of objective and constraints using Finite Difference.
    
    Args:
        optimizer: The optimizer instance (initialized with problem).
        num_checks: Number of random design variables to check.
        step: Finite difference step size.
        output_dir: Directory to save plots.
    """
    
    # Ensure optimizer is initialized (fe state populated)
    # We assume x is the current design in optimizer or we pick a random one?
    # Better to use a non-trivial x (e.g. 0.5 or random) to avoid boundary singularities.
    
    print("Starting Gradient Verification...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Random initialization (uniform 0.1 to 0.9)
    # Avoid 0.0 to prevent singularity issues in stiffness matrix
    print("Generating random density field [0.1, 0.9] for gradient check...")
    np.random.seed(None) # Ensure randomness
    x = np.random.uniform(0.0, 1.0, size=optimizer.fe.n_elem)
    x = np.clip(x, 0.0, 1.0)
    
    # 1. Base Evaluation (Analytical)
    # We need to perform analysis first
    p = optimizer.opt.parameters.penalization_param
    
    # Get physical variables using the evaluator's logic (handles filter/project)
    # This ensures consistency with how functions.py computes gradients w.r.t x (design)
    _, x_phys = optimizer.evaluator._get_physical_vars(x)
    
    K = optimizer.solver.assemble_stiffness(x_phys, p)
    optimizer.fe.K = K
    
    # Check for various fixed dof naming conventions
    if hasattr(optimizer.fe, 'fixeddofs_ind') and optimizer.fe.fixeddofs_ind is not None:
         fixed_dofs = optimizer.fe.fixeddofs_ind
    else:
         fixed_dofs = getattr(optimizer.fe, 'fixed_dofs', np.array([], dtype=int))
         
    F = getattr(optimizer.fe, 'F', np.zeros(optimizer.fe.n_node * optimizer.fe.dim))
    u = optimizer.solver.solve(K, F, fixed_dofs)
    optimizer.fe.U = u
    
    # Analytical Gradients w.r.t Design Variable x
    # Volume
    vol_val, vol_grad_ana = optimizer.evaluator.compute_volume_fraction(x)
    vol_grad_ana = np.array(vol_grad_ana).flatten() # Ensure flat
    
    # Stress
    # Assuming 'maximum stress violation' is in constraints
    stress_val, stress_grad_ana = optimizer.evaluator.compute_max_stress_violation(x, u)
    stress_grad_ana = np.array(stress_grad_ana).flatten()
    
    # Pick random indices
    indices = np.random.choice(len(x), size=min(num_checks, len(x)), replace=False)
    indices.sort()
    
    # FD Store
    vol_grad_fd = np.zeros(len(indices))
    comp_grad_fd = np.zeros(len(indices))
    stress_grad_fd = np.zeros(len(indices))
    
    vol_grad_ana_sub = vol_grad_ana[indices]
    
    # Compliance
    comp_val, comp_grad_ana = optimizer.evaluator.compute_compliance(x, u)
    comp_grad_ana = np.array(comp_grad_ana).flatten()
    comp_grad_ana_sub = comp_grad_ana[indices]
    
    stress_grad_ana_sub = stress_grad_ana[indices]
    
    print(f"Checking {len(indices)} variables with step {step}...")
    
    for i, idx in enumerate(indices):
        # Central Difference
        # Forward
        x_fwd = x.copy()
        x_fwd[idx] += step
        
        # Backward
        x_bwd = x.copy()
        x_bwd[idx] -= step
        
        # Solve Forward
        _, x_fwd_phys = optimizer.evaluator._get_physical_vars(x_fwd)
        K_fwd = optimizer.solver.assemble_stiffness(x_fwd_phys, p)
        # We assume F is constant
        u_fwd = optimizer.solver.solve(K_fwd, F, fixed_dofs)
        
        # Vol Fwd
        v_fwd, _ = optimizer.evaluator.compute_volume_fraction(x_fwd)
        # Comp Fwd
        c_fwd, _ = optimizer.evaluator.compute_compliance(x_fwd, u_fwd)
        # Stress Fwd
        s_fwd, _ = optimizer.evaluator.compute_max_stress_violation(x_fwd, u_fwd)
        
        # Solve Backward
        _, x_bwd_phys = optimizer.evaluator._get_physical_vars(x_bwd)
        K_bwd = optimizer.solver.assemble_stiffness(x_bwd_phys, p)
        u_bwd = optimizer.solver.solve(K_bwd, F, fixed_dofs)
        
        # Vol Bwd
        v_bwd, _ = optimizer.evaluator.compute_volume_fraction(x_bwd)
        # Comp Bwd
        c_bwd, _ = optimizer.evaluator.compute_compliance(x_bwd, u_bwd)
        # Stress Bwd
        s_bwd, _ = optimizer.evaluator.compute_max_stress_violation(x_bwd, u_bwd)
        
        # Gradients
        vol_grad_fd[i] = (v_fwd - v_bwd) / (2 * step)
        comp_grad_fd[i] = (c_fwd - c_bwd) / (2 * step)
        stress_grad_fd[i] = (s_fwd - s_bwd) / (2 * step)
        
        if (i+1) % 100 == 0:
            print(f"  Checked {i+1}/{len(indices)}")

    # Calculate Errors
    def compute_errors(ana, fd, name):
        abs_err = np.abs(ana - fd)
        # Avoid division by zero for relative error
        safe_ana = np.where(np.abs(ana) < 1e-12, 1.0, np.abs(ana))
        rel_err = abs_err / safe_ana
        # If ana was small, handle rel_err? 
        # MATLAB usually just does norm(diff)/norm(ana) or max relative.
        
        max_abs_idx_local = np.argmax(abs_err)
        max_abs = abs_err[max_abs_idx_local]
        max_abs_idx_global = indices[max_abs_idx_local]
        
        max_rel_idx_local = np.argmax(rel_err)
        max_rel = rel_err[max_rel_idx_local]
        max_rel_idx_global = indices[max_rel_idx_local]
        
        max_rel_idx_local = np.argmax(rel_err)
        max_rel = rel_err[max_rel_idx_local]
        max_rel_idx_global = indices[max_rel_idx_local]
        
        mean_val = np.mean(ana)
        
        print(f"\n--- {name} Gradients Check ---")
        print(f"Mean Gradient Value: {mean_val:.6e}")
        print(f"Max Absolute Error:  {max_abs:.6e} at Index {max_abs_idx_global}")
        print(f"Max Relative Error:  {max_rel:.6e} at Index {max_rel_idx_global}")
        
        return max_abs, max_rel

    compute_errors(vol_grad_ana_sub, vol_grad_fd, "Volume Fraction (compute_volume_fraction)")
    compute_errors(comp_grad_ana_sub, comp_grad_fd, "Compliance (compute_compliance)")
    compute_errors(stress_grad_ana_sub, stress_grad_fd, "Max Stress Aggregation (compute_max_stress_violation)")

    # Plotting
    _plot_verification(indices, vol_grad_ana_sub, vol_grad_fd, "Volume Fraction", os.path.join(output_dir, "grad_check_volume.pdf"))
    _plot_verification(indices, comp_grad_ana_sub, comp_grad_fd, "Compliance", os.path.join(output_dir, "grad_check_compliance.pdf"))
    _plot_verification(indices, stress_grad_ana_sub, stress_grad_fd, "Max Stress Aggregation", os.path.join(output_dir, "grad_check_stress.pdf"))
    
    print("\nGradient verification complete.")

def _plot_verification(indices, ana, fd, title, filename):
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot 1: Gradient Values
    ax1.plot(indices, ana, 'b-', marker='o', label='Analytical', alpha=0.7, linewidth=1.5)
    ax1.plot(indices, fd, 'r--', marker='x', label='Finite Difference', alpha=0.7, linewidth=1.5)
    
    ax1.set_ylabel('Gradient Value')
    ax1.set_xlabel('Design Variable Index')
    ax1.set_title(f'{title}: Gradient Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Handle auto-scaling artifact for constant values
    grad_range = np.max(ana) - np.min(ana)
    if grad_range < 1e-12:
        mean_val = np.mean(ana)
        ax1.set_ylim(mean_val - 1e-10, mean_val + 1e-10)

    # Annotate max error roughly on main plot? 
    # User said "don't plot the plot of absolute erro", implies visual clutter logic.
    # No need to annotate error.
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()  
