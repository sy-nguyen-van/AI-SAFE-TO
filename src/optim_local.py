
import numpy as np
import os
import sys
from typing import Optional, List, Dict, Any
from .struct_prob import Problem
from .utils import compute_filter_matrix, plot_density, plot_stress, plot_history, compute_heaviside_projection
from mmapy import mmasub, kktcheck
from .functions import FunctionEvaluator
from .fem import FEMSolver # Import FEMSolver
class Optimizer_local: # Optimizer class
    def __init__(self, problem: Problem): # Initialize with Problem instance
        self.problem = problem # Problem instance
        self.fe = problem.fe # FEM instance
        self.opt = problem.opt # Optimization instance        
        # Initialize FEM        
        self.solver = FEMSolver(problem) # Initialize FEMSolver
        self.solver.initialize() # Initialize FEM
        self.problem.solver = self.solver # Set solver in problem        
        # Initialize Evaluator
        self.evaluator = FunctionEvaluator(problem) # Initialize Evaluator        
        # Filter
        self.H = None # Filter matrix
        self.init_filter() # Initialize filter
        
        # ALM initialization
        n_cons = len(self.opt.parameters.constraint_types)
        self.lambda_cons = np.zeros(n_cons)
        self.mu_cons = np.full(n_cons, getattr(self.opt.parameters, 'vol_penal_min', 1.0))
        
    # (Removed linear penalty update _update_penalty to follow PolyStress adaptive ALM logic)
    def init_filter(self): # Initialize filter calculations
        if self.opt.H is not None: # If filter matrix is already initialized
            self.H = self.opt.H
            return            
        r = self.opt.parameters.filter_radius_factor * self.fe.max_elem_side # Filter radius
        print(f"Filter Radius: {r}")
        self.H = compute_filter_matrix(self.fe.centroids, r, self.fe.max_elem_side)
        self.opt.filter_radius = r
        self.opt.H = self.H        
    # _project and _project_gradient replaced by utils.compute_heaviside_projection
    def _evaluate_all(self, x):
        """Helper to evaluate obj, constr, and sensitivities."""
        opt = self.opt
        fe = self.fe
        params = opt.parameters        
        # 1. Project
        beta = params.projection.current_beta
        # Initialize if missing
        if beta == 0.0:
             params.projection.current_beta = params.projection.beta_init
             beta = params.projection.beta_init        
        # Calculate x_phys
        # Filter
        if self.H is not None:
             x_tilde = self.H.dot(x)
        else:
             x_tilde = x             
        # Projection
        if params.projection.use:
             x_phys, _ = compute_heaviside_projection(x_tilde, beta, params.projection.eta)
        else:
             x_phys = x_tilde             
        # Store state for functions.py
        self.opt.x_current = x
        self.opt.x_tilde = x_tilde
        self.opt.x_phys = x_phys             
        # Solve
        K = self.solver.assemble_stiffness(x_phys, params.penalization_param, fe.material.rho_min)
        fe.K = K
        # Use processed fixed DOFs if available, else fallback to input attribute
        fixed_dofs = getattr(fe, 'fixeddofs_ind', getattr(fe, 'fixed_dofs', []))
        F = getattr(fe, 'F', None)
        u = self.solver.solve(K, F, fixed_dofs,'primal')
        fe.U = u        
        # Objective params.penalization_param.'objective_type' = 'volume' or 'compliance'
        obj_type = getattr(params, 'objective_type', 'volume')
        if obj_type == 'volume':
             f0val, df0dx = self.evaluator.compute_volume_fraction(x)
        elif obj_type == 'compliance':
             f0val, df0dx = self.evaluator.compute_compliance(x, u)
        else:
             f0val, df0dx = 0.0, np.zeros_like(x)             
        # Constraints
        n_cons = len(params.constraint_types)
        fval = np.zeros(n_cons)
        dfdx = np.zeros((n_cons, len(x)))        
        for i, c_type in enumerate(params.constraint_types):
             if c_type == 'stress':
                 val, grad = self.evaluator.compute_max_stress_violation(x, u)
                 fval[i] = val
                 dfdx[i, :] = grad
             elif c_type == 'local_stress':
                 mu_val = self.mu_cons[i]
                 val, grad = self.evaluator.compute_local_stress_penalty(x, u, mu_val)
                 fval[i] = val
                 dfdx[i, :] = grad
             elif c_type == 'volume':
                 target_vol = getattr(params, 'target_volume', 0.3)
                 v, dv = self.evaluator.compute_volume_fraction(x)
                 fval[i] = v / target_vol - 1.0
                 dfdx[i, :] = dv / target_vol
        
        return f0val, df0dx, fval, dfdx, x_phys, u
    def optimization_loop(self):
        """
        Main optimization loop matching runmma.m logic.
        """
        opt = self.opt
        fe = self.fe
        params = opt.parameters
        out_dir = opt.options.outputs_path
        # Before running optimization, make sure H is initialized
        if self.H is None:
            raise ValueError("Filter matrix H is not initialized. Please call init_filter() first.")
        # Initialize Partitioning (BCs and Indexing)
        self.solver.FE_init_partitioning()
        
        # Initialize History
        opt.history = {
            'x': [], 'fval': [], 'fconsval': [], 'grf': [], 'norm_dfdx': []
        }
        if opt.stress_needed:
            opt.history['gstress'] = []
            opt.history['true_stress_max'] = []

        # Initialize Lower/Upper Bounds
        lb = np.zeros(fe.n_elem)
        ub = np.ones(fe.n_elem)
        
        if params.penalization_scheme in ['SIMP', 'RAMP']:
            lb[:] = fe.material.rho_min
        elif params.penalization_scheme in ['modified_SIMP', 'modified_RAMP']:
            lb[:] = fe.material.rho_min / 100.0
        else:
            print("Warning: Unrecognized penalization scheme.")
            return

        # Optimization configuration
        n_cons = len(params.constraint_types)
        n_dv = fe.n_elem # ndv
        opt.n_dv = n_dv
        
        # x initialization
        if opt.dv is None:
             opt.dv = params.init_dens * np.ones(n_dv)
        x = opt.dv.copy()
        xold1 = x.copy()
        xold2 = x.copy()        
        # Move Limits
        ml_step = opt.options.move_limit * np.abs(ub - lb)        
        # MMA Asymptotes
        low = lb.copy(); upp = ub.copy()        
        # MMA Constants
        mma_params = params.mma
        # Ensure arrays
        if mma_params.c is None:
             mma_c = 1000.0 * np.ones(n_cons)
        else:
             mma_c = mma_params.c             
        if mma_params.d is None:
             mma_d = np.ones(n_cons)
        else:
             mma_d = mma_params.d             
        if mma_params.a is None:
             mma_a = np.zeros(n_cons)
        else:
             mma_a = mma_params.a             
        mma_a0 = mma_params.a0
        
        # Initial Evaluation
        iter_k = 0        
        # Evaluate
        f0val, df0dx, fval, dfdx, x_phys, u = self._evaluate_all(x)        
        # Print Initial
        max_cons = np.max(np.maximum(fval, 0.0)) if n_cons > 0 else 0.0
        print(f"It. {iter_k}, Obj= {f0val:.5e}, ConsViol = {max_cons:.5e}")        
        # Save History
        opt.history['fval'].append(f0val)
        opt.history['fconsval'].append(fval.copy()) # Store array
        opt.history['x'].append(x.copy())
        if opt.stress_needed:
             opt.history['gstress'].append(opt.approx_h_max)
             opt.history['true_stress_max'].append(opt.true_stress_max)             
        # Plot Initial
        if opt.options.plot:
             output_path = getattr(opt.options, 'outputs_path', '.')
             if output_path != '.' and not os.path.exists(output_path):
                 os.makedirs(output_path, exist_ok=True)
             plot_density(fe.coords, fe.elem_node, x_phys, os.path.join(output_path, 'density_iter.png'), show=True)             
        # Init stopping values
        obj_change = 10.0 * opt.options.obj_tol
        # GRF: Global Resource Function (Monitoring measure)
        grf = 4.0 * np.dot(x * (1.0 - x), fe.elem_vol) / np.sum(fe.elem_vol)
        opt.grf = grf
        opt.history['grf'].append(grf)                
        # Linear Beta Update
        beta_min = params.projection.beta_init
        beta_max = params.projection.beta_final
        # === MAIN MMA LOOP ===
        while iter_k < opt.options.max_iter:
            iter_k += 1            
            # Impose Move Limits
            # step_fac logic
            if opt.stress_needed:
                 # Check true stress max for move limit relaxation
                 if np.all(opt.true_stress_max <= 1.2 * params.slimit):
                     step_fac = 2.0
                 else:
                     step_fac = 1.0
            else:
                 step_fac = 1.0                 
            mlb = np.maximum(lb, x - step_fac * ml_step)
            mub = np.minimum(ub, x + step_fac * ml_step)
            
            # Projection Continuation (Beta Update)
            if params.projection.use:                
                # Formula: coeff = beta_min + iter * ((beta_max - beta_min) / max_iter)
                new_beta = beta_min + iter_k * ((beta_max - beta_min) / opt.options.max_iter)
                
                # Cap at beta_max (though formula naturally reaches it at max_iter)
                if new_beta > beta_max:
                    new_beta = beta_max
                    
                params.projection.current_beta = new_beta            
            # Adaptive Constraint Scaling (ACS)
            if params.aggregation_type == 'p-norm' and opt.stress_needed:             
                if params.ACS.use:
                    if iter_k <= 3:
                        cc = 1.0
                    else:
                        c_hist = params.ACS.c
                        # iter 1 based in matlab. iter-3 means last 3.
                        # If len(c_hist) >= 3
                        c_old3 = c_hist[-3]
                        c_old2 = c_hist[-2]
                        c_old1 = c_hist[-1]                        
                        if (c_old3 - c_old2) * (c_old2 - c_old1) < 0:
                            alpha = params.ACS.alpha_osc
                        else:
                            alpha = params.ACS.alpha_no_osc                            
                        # cc = true_h_max / approx_h_max
                        cc_new = opt.true_h_max / opt.approx_h_max
                        cc = alpha * cc_new + (1.0 - alpha) * c_old1                        
                    params.ACS.c.append(cc)
                else:
                    cc = 1.0                    
                fval = cc * (fval + 1.0) - 1.0
                dfdx = cc * dfdx                     
            # Function scaling
            f0val = opt.functions.objective_scale * f0val
            df0dx = df0dx * opt.functions.objective_scale
            # Handle constraint scaling with broadcasting
            c_scale_input = opt.functions.constraint_scale
            if np.isscalar(c_scale_input):
                 c_scale = np.full(n_cons, c_scale_input)
            else:
                 c_scale = np.array(c_scale_input)
                 if c_scale.ndim == 0:
                      c_scale = np.full(n_cons, c_scale.item())
                 elif c_scale.shape[0] != n_cons:
                      c_scale = np.ones(n_cons)                      
            fval = c_scale * fval
            # Scale rows of dfdx: (n_cons, 1) * (n_cons, n_dv)
            dfdx = c_scale[:, np.newaxis] * dfdx            
            
            # For ALM, add penalty to objective and dummy out constraint for MMA
            for i, c_type in enumerate(params.constraint_types):
                if c_type == 'local_stress':
                    f0val += fval[i]
                    df0dx += dfdx[i, :]
                    fval[i] = -1.0
                    dfdx[i, :] = 0.0

            # Solve MMA Subproblem
            # mmapy mmasub expects (n, 1) for vectors
            x_col = x.reshape(-1, 1)
            mlb_col = mlb.reshape(-1, 1)
            mub_col = mub.reshape(-1, 1)
            xold1_col = xold1.reshape(-1, 1)
            xold2_col = xold2.reshape(-1, 1)
            df0dx_col = df0dx.reshape(-1, 1)
            fval_col = fval.reshape(-1, 1)            
            # Execute MMA subproblem
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
                     n_cons, n_dv, iter_k, x_col, mlb_col, mub_col, xold1_col, xold2_col,
                     f0val, df0dx_col, fval_col, dfdx,
                     low, upp, mma_a0, mma_a.reshape(-1, 1), mma_c.reshape(-1, 1), mma_d.reshape(-1, 1), 
                     move=1.0                 )
            # Update Design
            xold2 = xold1.copy()
            xold1 = x.copy()
            x = xmma.flatten()
            opt.dv = x # Update in struct
            # Update Eval
            f0val, df0dx, fval, dfdx, x_phys, u = self._evaluate_all(x)           
            
            # ---------------------------------------------
            # PolyStress ALM Adaptive Updates (Outer Iteration)
            # ---------------------------------------------
            alm_update_interval = 10 # Default for MMA if not specified
            if hasattr(opt, 'nn_params'):
                alm_update_interval = getattr(opt.nn_params, 'alm_update_interval', 10)
            
            if iter_k % alm_update_interval == 0:
                tau = 0.5
                gamma = 1.5
                mu_max = getattr(opt.parameters, 'vol_penal_max', 20.0)
                
                for i, c_type in enumerate(opt.parameters.constraint_types):
                    if c_type == 'local_stress':
                        if hasattr(opt, 'local_h_e') and hasattr(opt, 'mu_local') and hasattr(opt, 'lam_local'):
                            # Element-wise violation v_e
                            v_e = np.maximum(0.0, opt.local_h_e)
                            
                            # Initialize previous violations if first update
                            if not hasattr(opt, 'v_prev_local'):
                                opt.v_prev_local = v_e.copy()
                                
                            # Find elements where violation did NOT decrease sufficiently
                            bad_convergence = (v_e > tau * opt.v_prev_local) & (v_e > 1e-6)
                            
                            # Increase mu_e for those elements
                            opt.mu_local[bad_convergence] = np.minimum(mu_max, gamma * opt.mu_local[bad_convergence])
                            
                            # Update lambda_e for all elements
                            opt.lam_local = np.maximum(0.0, opt.lam_local + opt.mu_local * opt.local_h_e)
                            
                            # Save current violation for next check
                            opt.v_prev_local = v_e.copy()
                    else:
                        # Standard global ALM constraints
                        scale_i = (opt.functions.constraint_scale[i]
                                   if isinstance(opt.functions.constraint_scale, (list, tuple, np.ndarray))
                                   else opt.functions.constraint_scale)
                        scaled_c_val = fval[i] * scale_i
                        v_i = max(0.0, scaled_c_val)
                        
                        if not hasattr(opt, 'v_prev_global'):
                            opt.v_prev_global = np.zeros(len(opt.parameters.constraint_types))
                            
                        # If violation didn't decrease enough
                        if v_i > tau * opt.v_prev_global[i] and v_i > 1e-6:
                            self.mu_cons[i] = min(mu_max, gamma * self.mu_cons[i])
                            
                        # Update global multiplier
                        self.lambda_cons[i] = max(0.0, self.lambda_cons[i] + self.mu_cons[i] * scaled_c_val)
                        opt.v_prev_global[i] = v_i
            
            # KKT Check
            # Reshape for kktcheck
            df0dx_col = df0dx.reshape(-1, 1)
            fval_col = fval.reshape(-1, 1)
            # dfdx is (m, n)
            lb_col = lb.reshape(-1, 1)
            ub_col = ub.reshape(-1, 1)
            residu, kktnorm, residumax = kktcheck(
                n_cons, n_dv, x_mma_col := x.reshape(-1, 1), ymma, zmma, lam, xsi, eta, mu, zet, s,
                lb_col, ub_col, df0dx_col, fval_col, dfdx, mma_a0, mma_a.reshape(-1, 1), mma_c.reshape(-1, 1), mma_d.reshape(-1, 1)
            )            
            # Output Screen            
            out_const = fval
            if opt.stress_needed:    
                if 'local_stress' in opt.parameters.constraint_types:
                    out_const = opt.true_h_max
                else:
                    out_const = opt.true_h_max - 1        
            print(f"It. {iter_k}, Obj= {f0val:.5e}, ConsViol = {np.max(out_const):.5e}, KKT-norm = {kktnorm:.5e}, Obj change = {obj_change:.5e}")
            # Save Output
            # We skip .mat save for now, assume python workflow uses artifacts or pickle if needed.
            # History Update
            opt.history['fval'].append(f0val)
            opt.history['fconsval'].append(fval.copy())
            opt.history['x'].append(x.copy())
            if opt.stress_needed:
                 opt.history['gstress'].append(opt.approx_h_max)
                 opt.history['true_stress_max'].append(opt.true_stress_max)
                 opt.history['norm_dfdx'].append(np.max(np.abs(dfdx)))
            # Check Obj Convergence
            if iter_k > 1:
                 # ((f_new - f_old) / f_old)
                 f_old = opt.history['fval'][-2]
                 obj_change = (f0val - f_old) / f_old
                 if abs(obj_change) < opt.options.obj_tol:
                     print("Objective function convergence tolerance satisfied.")
            # Check GRF
            grf = 4.0 * np.dot(x * (1.0 - x), fe.elem_vol) / np.sum(fe.elem_vol)
            opt.grf = grf
            opt.history['grf'].append(grf)            
            # Stopping Criteria
            if abs(obj_change) < opt.options.obj_tol and grf < opt.options.max_GRF:
                 # Check continuation
                 if (params.continuation and cont_ended) or not params.continuation:
                     print("Satisfied stopping criteria.")
                     break                     
            # Iterative Plotting
            # Plot current x_phys
            plot_density(self.fe.coords, self.fe.elem_node, x_phys, os.path.join(out_dir, 'TOP_density_iter.png'), show=True, title=f"Iteration {iter_k}")
        # Save Final Density Plot
        f0val, df0dx, fval, dfdx, x_phys, u = self._evaluate_all(x) 
        # Ensure stress is available for plotting
        if getattr(self.fe, 'svm', None) is None:
             self.evaluator.compute_max_stress_violation(x, u)
        
        relaxed_stress = self.fe.svm.flatten()
        vol_frac,_ = self.evaluator.compute_volume_fraction(x)
        
        plot_density(self.fe.coords, self.fe.elem_node, x_phys, 
                    filename=os.path.join(out_dir, 'TOP_density.png'), 
                    show=False, title=f'MMA; $v_f = {vol_frac:.2f}$',
                    H=self.opt.H, penal=self.opt.parameters.penalization_param)
        # Save files
        plot_stress(self.fe.coords, self.fe.elem_node, relaxed_stress, 
            filename=os.path.join(out_dir, 'TOP_stress.png'), show=False)
        plot_history(opt.history, filename=os.path.join(out_dir, 'TOP_history.png'), show=False)
        print(f"Saved plots to '{out_dir}'")