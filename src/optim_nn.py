"""
Neural Network - Based Topology Optimization
--------------------------------------------
This module performs density-based topology optimization where the design
field ρ(x) is represented by a neural network instead of explicit element
variables. Gradients from the FEM (compliance, stress, volume) are injected
into PyTorch autograd using a custom wrapper, enabling backpropagation through
the NN weights.
"""

import os
import torch
import torch.optim as optim
import numpy as np
from .struct_prob import Problem
from .utils import compute_filter_matrix, plot_density, plot_stress, plot_history, compute_heaviside_projection
from .functions import FunctionEvaluator
from .fem import FEMSolver # Import FEMSolver
from .nn_torch import DensityNetwork, TOuNN_Network
# =============================================================================
# External Gradient → Autograd Bridge
# =============================================================================
class ExternalFunction(torch.autograd.Function):
    """
    This class wraps FEM-computed scalar and gradient values into a differentiable
    PyTorch graph. It **does not** compute gradients itself — instead, it inserts
    the externally computed df/dρ so PyTorch can flow gradients backward through
    the NN parameters.

    f(ρ)   ≈ FEM evaluation (compliance / volume / stress)
    df/dρ  = sensitivity array from FEM

    During forward():
        store gradient tensor so backward() can return it.

    During backward():
        grad_output = upstream derivative (chain rule)
        return grad_output * grad_tensor  ← autograd propagation
    """

    @staticmethod
    def forward(ctx, input_tensor, value, gradient):
        # Convert numpy gradients to proper torch tensor on correct device
        if not torch.is_tensor(gradient):
            grad_tensor = torch.from_numpy(gradient).to(input_tensor.device, input_tensor.dtype)
        else:
            grad_tensor = gradient.to(input_tensor.device, input_tensor.dtype)

        # Must match input_tensor shape (N elements)
        grad_tensor = grad_tensor.reshape_as(input_tensor)

        # Save gradient for backward()
        ctx.save_for_backward(grad_tensor)

        # Return scalar (loss term) as proper torch tensor
        return value if torch.is_tensor(value) else torch.tensor(
            value,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Extract stored gradient
        (grad_tensor,) = ctx.saved_tensors
        # Chain rule: ∂L/∂ρ = grad_output * df/dρ
        return grad_output * grad_tensor, None, None
# =============================================================================
# NN-Based Optimizer Class
# =============================================================================
class Optimizer_Neural:
    """
    Neural-Network topology optimization framework.

    Method:
        Neural network predicts density ρ(x) for each element centroid
        FEM computes objective + constraint values and sensitivities df/dρ
        Custom autograd injects these sensitivities
        Backpropagation → update NN weights
        Augmented-Lagrangian / penalty enforces constraints

    Good for:
        - stress-constrained TO
        - differentiable optimization without MMA
    """
    def __init__(self, problem: Problem, network=None):
        self.problem = problem # Problem instance
        self.fe = problem.fe # FEM instance
        self.opt = problem.opt # Optimization instance        
        # Initialize FEM        
        self.solver = FEMSolver(problem) # Initialize FEMSolver
        self.solver.initialize() # Initialize FEM
        self.problem.solver = self.solver # Set solver in problem        
        # Initialize Evaluator
        self.evaluator = FunctionEvaluator(problem) # Initialize Evaluator        
        # Filter - Disabled for NN (Neural Network provides implicit filtering)
        self.H = None
        self.opt.H = None
        # -----------------------------
        # Build Neural Network
        # -----------------------------
        nn_cfg = problem.opt.nn_params
        if network is None:            
            if getattr(nn_cfg, 'use_tounn_logic', False):
                self.net = TOuNN_Network(
                    input_dim=self.fe.dim,
                    hidden_dim=nn_cfg.hidden_dim,
                    num_layers=nn_cfg.num_layers
                )
            else:
                self.net = DensityNetwork(
                    input_dim=self.fe.dim,
                    hidden_dim=nn_cfg.hidden_dim,
                    num_layers=nn_cfg.num_layers,
                    activation=nn_cfg.activation,
                    use_fourier=getattr(nn_cfg, "use_fourier", False),
                    fourier_scale=getattr(nn_cfg, "fourier_scale", 10.0),
                    feature_type=getattr(nn_cfg, "feature_type", 'fourier'),
                )
        else:
            self.net = network

        # Move NN to device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # Print NN info
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f" Neural Network Initialized: {total_params} trainable parameters")

        # -----------------------------------------
        # Prepare NN input: element centroid coords
        # -----------------------------------------
        centroids = self._compute_centroids()

        # Normalize domain to [-1,1] range
        if getattr(nn_cfg, 'use_tounn_logic', False):
            self.centroids_norm = centroids
        else:
            min_c = np.min(self.fe.coords, axis=0)
            max_c = np.max(self.fe.coords, axis=0)
            self.centroids_norm = 2.0 * (centroids - min_c) / (max_c - min_c) - 1.0

        # Tensor used at every iteration (no grad needed)
        self.centroids_tensor = torch.tensor(
            self.centroids_norm, dtype=torch.float32, device=self.device
        )

        # --------------------------------------------------------
        # Optimizer & LR scheduler
        # --------------------------------------------------------
        self.learning_rate = nn_cfg.learning_rate
        opt_type = getattr(nn_cfg, 'optimizer_type', 'Adam')
        
        if opt_type == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        elif opt_type == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
        elif opt_type == 'LBFGS':
            self.optimizer = optim.LBFGS(self.net.parameters(), lr=self.learning_rate, max_iter=20)
        else:
            print(f"Warning: Unknown optimizer type '{opt_type}'. Defaulting to Adam.")
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
            
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        # --------------------------------------------------------
        # Augmented-Lagrangian Parameters
        # --------------------------------------------------------
        n_cons = len(problem.opt.parameters.constraint_types)
        self.lambda_cons = np.zeros(n_cons)           # optional: true AL λ multipliers
        self.mu_cons = np.full(n_cons, nn_cfg.vol_penal_min)  # quadratic penalty weight
        self.obj0 = None                              # used for objective normalization
        # Initialize History
        self.history = {'x': [], 'fval': [], 'fconsval': [], 'grf': [], 'loss': []}
    # =========================================================================
    # Filter
    # =========================================================================
    def init_filter(self): # Initialize filter calculations
        if self.H is not None: # If filter matrix is already initialized
            return            
        r = self.opt.parameters.filter_radius_factor * self.fe.max_elem_side # Filter radius
        print(f"Filter Radius: {r}")
        self.H = compute_filter_matrix(self.fe.centroids, r, self.fe.max_elem_side)
        self.opt.filter_radius = r
        self.opt.H = self.H  
    # =========================================================================
    # FEM Wrapper
    # =========================================================================
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
        fixed_dofs = fe.fixeddofs_ind
        F = fe.F
        u = self.solver.solve(K, F, fixed_dofs,'primal')
        fe.U = u        
        # Objective params.penalization_param.'objective_type' = 'volume', 'compliance', or 'stress'
        obj_type = params.objective_type
        if obj_type == 'volume':
             f0val, df0dx = self.evaluator.compute_volume_fraction(x)
        elif obj_type == 'compliance':
             f0val, df0dx = self.evaluator.compute_compliance(x, u)
        elif obj_type == 'stress':
             f0val, df0dx = self.evaluator.compute_max_stress_violation(x, u)
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
                 target_vol = params.target_volume
                 v, dv = self.evaluator.compute_volume_fraction(x)
                 fval[i] = v / target_vol - 1.0
                 dfdx[i, :] = dv / target_vol
        
        return f0val, df0dx, fval, dfdx, x_phys, u

    # =========================================================================
    def _create_external_tensor(self, rho_tensor, value, gradient, scale=1.0):
        """
        Wrap FEM scalar + gradient into differentiable autograd node.
        """
        return ExternalFunction.apply(rho_tensor, value * scale, gradient * scale)

    def _update_penalty(self, epoch, max_iter):
        """
        Increase quadratic penalty weight μ from min → max over iterations.
        """
        nn_p = self.problem.opt.nn_params
        t = epoch / (max_iter - 1)
        self.mu_cons[:] = nn_p.vol_penal_min + (nn_p.vol_penal_max - nn_p.vol_penal_min) * t

    def _compute_centroids(self):
        """
        Compute centroid of each element:
            centroid = average of its node coordinates
        """
        nodes = self.fe.coords               # (n_nodes × dim)
        elems = self.fe.elem_node            # (n_elem × nodes_per_elem)
        el_coords = nodes[elems]             # (n_elem × nodes_per_elem × dim)
        return np.mean(el_coords, axis=1)    # (n_elem × dim)
    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    def optimization_loop(self):
        print("====== Starting Neural-Network Topology Optimization ======")
        opt = self.opt
        fe = self.fe
        params = opt.parameters
        out_dir = opt.options.outputs_path
        # No explicit density filter needed for Neural Network optimization
        # Initialize Partitioning (BCs and Indexing)
        self.solver.FE_init_partitioning()
        # Evaluate
        # x init
        if opt.dv is None:
             opt.dv = params.init_dens * np.ones(fe.n_elem)
        x = opt.dv.copy()
        f0val, df0dx, fval, dfdx, x_phys, u = self._evaluate_all(x)  
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
        self.grf = grf
        self.history['grf'].append(grf)                   
        # ******* MAIN MMA LOOP *******
        # Linear Beta Update
        beta_min = params.projection.beta_init
        beta_max = params.projection.beta_final
        
        # TOuNN logic initialization
        if getattr(opt.nn_params, 'use_tounn_logic', False):
            tounn_alpha = 0.1
            tounn_alpha_inc = 0.05
            tounn_alpha_max = 100.0
            # TOuNN penalization continuation
            params.penalization_param = 2.0
            tounn_penal_inc = 0.01
            tounn_penal_max = 4.0
        
        for epoch in range(opt.options.max_iter):
            # (Removed linear penalty update to follow PolyStress adaptive ALM logic)
            
            # Projection Continuation (Beta Update)
            if params.projection.use:                
                new_beta = beta_min + epoch * ((beta_max - beta_min) / opt.options.max_iter)                
                if new_beta > beta_max:
                    new_beta = beta_max
                params.projection.current_beta = new_beta

            # ---------------------------------------------
            #  NN predicts density at element centroids
            # ---------------------------------------------
            rho_tensor = self.net(self.centroids_tensor).squeeze()     # shape = (n_elem,)
            rho_np = rho_tensor.detach().cpu().numpy()                 # used by FEM
            # ---------------------------------------------
            #  FEM physics evaluation
            # ---------------------------------------------
            f0val, df0drho, fval, dfdx, x_phys, u = self._evaluate_all(rho_np)            
            # Normalize objective scale once
            if self.obj0 is None:
                self.obj0 = f0val
                if opt.stress_needed == 1:
                    self.obj0 = 1.0

            # Inject objective term into autograd with normalization
            # Scale is divided by the initial objective value (obj0) to balance the gradients
            norm_scale = opt.functions.objective_scale / (abs(self.obj0) + 1e-12)
            loss_obj = self._create_external_tensor(
                rho_tensor, f0val, df0drho, scale=norm_scale
            )
            loss = loss_obj

            # ---------------------------------------------
            #  Constraint contributions
            # ---------------------------------------------
            constraint_status = []
            for i, c_type in enumerate(opt.parameters.constraint_types):
                if c_type == 'volume' and getattr(opt.nn_params, 'use_tounn_logic', False):
                    # TOuNN calculates volume natively in PyTorch
                    target_vol = params.target_volume if hasattr(params, 'target_volume') else 0.5
                    volConstraint = (torch.mean(rho_tensor) / target_vol) - 1.0
                    loss += tounn_alpha * torch.pow(volConstraint, 2)
                    constraint_status.append(f"VolViol={volConstraint.item():+.4f}")
                else:
                    scale_i = (opt.functions.constraint_scale[i]
                               if isinstance(opt.functions.constraint_scale, (list, tuple, np.ndarray))
                               else opt.functions.constraint_scale)
    
                    fval_tensor = self._create_external_tensor(rho_tensor, fval[i], dfdx[i], scale_i)
                    constraint_status.append(f"ConsViol={fval[i]:+.4f}")
                    mu = self.mu_cons[i]
                    lam = self.lambda_cons[i]
    
                    if c_type == 'local_stress':
                        # fval_tensor is ALREADY the fully computed local ALM penalty P_AL.
                        # Add it directly to PyTorch loss (it carries its own FEM gradients)
                        loss += fval_tensor
                    else:
                        # Inequality Augmented Lagrangian: L = 1/(2μ) * (max(0, λ + μ*fval)^2 - λ^2)
                        active_constraint = torch.relu(lam + mu * fval_tensor)
                        penalty = (active_constraint**2 - lam**2) / (2.0 * mu)             
                        loss += penalty
            
            if getattr(opt.nn_params, 'use_tounn_logic', False):
                tounn_alpha = min(tounn_alpha_max, tounn_alpha + tounn_alpha_inc)

            # ---------------------------------------------
            #  Backpropagate & optimize NN weights
            # ---------------------------------------------
            self.optimizer.zero_grad()
            loss.backward()
            if getattr(opt.nn_params, 'use_tounn_logic', False):
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.1)
            else:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            if not getattr(opt.nn_params, 'use_tounn_logic', False):
                self.scheduler.step()       # optional — could update every 20 iters
            
            # TOuNN penalization continuation update
            if getattr(opt.nn_params, 'use_tounn_logic', False):
                params.penalization_param = min(tounn_penal_max, params.penalization_param + tounn_penal_inc)

            # ---------------------------------------------
            # PolyStress ALM Adaptive Updates (Outer Iteration)
            # ---------------------------------------------
            alm_update_interval = getattr(opt.nn_params, 'alm_update_interval', 10)
            if (epoch + 1) % alm_update_interval == 0:
                tau = 0.5
                gamma = 1.5
                mu_max = opt.nn_params.vol_penal_max
                
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
                        # Standard global ALM constraints (like volume)
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

            # Print progress
            print(
                f"[{epoch:03d}]  Loss={loss.item():.4e},  Obj={f0val:.4f},  "
                f"{'  '.join(constraint_status)},  "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            self.history['fval'].append(f0val)
            self.history['fconsval'].append(fval.copy())
            self.history['x'].append(rho_np)
            self.history['loss'].append(loss.item())
            
            # Check GRF
            grf = 4.0 * np.dot(rho_np * (1.0 - rho_np), self.fe.elem_vol) / np.sum(self.fe.elem_vol)
            self.grf = grf
            self.history['grf'].append(grf)

            # Iterative Plotting
            if getattr(opt.nn_params, 'use_tounn_logic', False):
                x_plot = x_phys ** params.penalization_param
            else:
                x_plot = x_phys
            plot_density(self.fe.coords, self.fe.elem_node, x_plot, 
            os.path.join(out_dir, 'NN_density_iter.png'), show=True, title=f"Iteration {epoch}")
            
        # Save Final Density Plot
        f0val, df0dx, fval, dfdx, x_phys, u = self._evaluate_all(rho_np) 
        
        # Ensure stress is available for plotting
        if getattr(self.fe, 'svm', None) is None:
             self.evaluator.compute_max_stress_violation(rho_np, u)
             
        relaxed_stress = self.fe.svm.flatten()
        vol_frac,_ = self.evaluator.compute_volume_fraction(rho_np)
        
        if getattr(opt.nn_params, 'use_tounn_logic', False):
            x_plot = x_phys ** params.penalization_param
        else:
            x_plot = x_phys
        
        plot_density(self.fe.coords, self.fe.elem_node, x_plot, 
                    filename=os.path.join(out_dir, 'NN_density.png'), 
                    show=False, title=f'NN; $v_f = {vol_frac:.2f}$',
                    H=self.opt.H, penal=self.opt.parameters.penalization_param)
        # Save files
        plot_stress(self.fe.coords, self.fe.elem_node, relaxed_stress, 
            filename=os.path.join(out_dir, 'NN_stress.png'), show=False)
        plot_history(self.history, filename=os.path.join(out_dir, 'NN_history.png'), show=False)
        print(f"Saved plots to '{out_dir}'")
        
