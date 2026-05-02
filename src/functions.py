# Importing required modules
import numpy as np
from typing import Tuple, Dict
from .struct_prob import Problem
from .utils import compute_filter_matrix, compute_heaviside_projection, compute_simp, compute_modified_simp
# FunctionEvaluator class
class FunctionEvaluator: # FunctionEvaluator class
    def __init__(self, problem: Problem): # Initialize with Problem instance
        self.problem = problem # Problem instance
        self.fe = problem.fe # FEM instance
        self.opt = problem.opt # Optimizer instance
        
    def _get_physical_vars(self, x: np.ndarray): # Get physical variables
        # Density Filter
        if self.opt.parameters.filter_type == 'density' and self.opt.H is not None: # Check if density filter is used
             x_tilde = self.opt.H.dot(x)
        else:
             x_tilde = x             
        # Projection
        if self.opt.parameters.projection.use: # Check if Heaviside projection is used
             beta = self.opt.parameters.projection.current_beta # Current beta value
             x_phys, _ = compute_heaviside_projection(x_tilde, beta, self.opt.parameters.projection.eta) # Projected design variables
        else:
             x_phys = x_tilde # No projection
             
        return x_tilde, x_phys # Return filtered and projected design variables

    def compute_volume_fraction(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns volume fraction and its gradient w.r.t design variable x."""
        x_tilde, x_phys = self._get_physical_vars(x) # Convolution filter    
        vol = self.fe.elem_vol # Element volumes
        total_vol = np.sum(vol) # Total volume
        vf = np.dot(x_phys, vol) / total_vol # Volume fraction
        # Base sensitive w.r.t x_phys
        dv_dphys = vol / total_vol # Base sensitive w.r.t x_phys
        # Chain Rule
        grad = dv_dphys        
        if self.opt.parameters.projection.use: # Check if Heaviside projection is used
             beta = self.opt.parameters.projection.current_beta
             _, d_proj = compute_heaviside_projection(x_tilde, beta, self.opt.parameters.projection.eta)
             grad = grad * d_proj
        # Filter Derivative (Chain rule for density, Heuristic for sensitivity)
        if self.opt.H is not None: # Check if density filter is used
             grad = self.opt.H.T.dot(grad)                 
        return vf, grad # Return volume fraction and its gradient
        
    def compute_compliance(self, x: np.ndarray, u: np.ndarray) -> Tuple[float, np.ndarray]: # Compute compliance and its gradient
        """
        Computes the mean compliance and its sensitivities.
        c = dot(U, P)
        grad_c = H^T * (dc_dx * dpen)
        """
        x_tilde, x_phys = self._get_physical_vars(x) # Get physical variables
        
        # 1. Compute Compliance c = U . P
        F = self.fe.F # Load vector
        load_total = F[:, None] if F.ndim == 1 else F # Load vector

        if u.ndim == 1: u = u[:, None]
        if load_total.ndim == 1: load_total = load_total[:, None]        
        # Ensure dimensions align for dot product
        if load_total.shape[1] != u.shape[1]:
             # If mismatch, assume single load vector applies to all U columns (unlikely for linear statics unless multi-BC)
             # or simply take sum. Correct phys is trace(U^T P).
             # Element-wise product and sum handles both (n, 1) and (n, nloads) cases if broadcasting works or dimensions match
             pass
             
        c_i = u * load_total # Compliance
        c = np.sum(c_i) # Total compliance
        # 2. Sensitivity
        p = self.opt.parameters.penalization_param # Penalization parameter
        Ke = self.fe.Ke # Stiffness matrix        
        # Compute d_penalty/d_x
        interp_type = self.opt.parameters.interpolation_type # Interpolation type
        rho_min = self.opt.parameters.rho_min # Minimum density
        if interp_type == 'modified_simp': # Modified SIMP
             _, dpen_rho_e = compute_modified_simp(x_phys, p, rho_min) # Derivative of penalty w.r.t density
        else: # SIMP
             _, dpen_rho_e = compute_simp(x_phys, p) # Derivative of penalty w.r.t density              
        # Ue: (n_elem, n_elem_dof, n_loads)
        # Using advanced indexing
        Ue = u[self.fe.edof_mat.astype(int), :] # (n_elem, n_edof, n_loads)        
        # Step 1: Ke @ Ue
        Ke_Ue = np.einsum('nij, njk -> nik', Ke, Ue) # (n_elem, n_edof, n_loads)    
        # Step 2: Ue . (Ke Ue)
        Dc_Dpenalized_elem_dens = np.sum(-1*Ue * Ke_Ue, axis=(1, 2)) # sum over dofs and loads
        grad = Dc_Dpenalized_elem_dens * dpen_rho_e # Gradient of compliance w.r.t density      
        # Chain Rule for Projection and Filter
        if self.opt.parameters.projection.use: # Check if Heaviside projection is used
             beta = self.opt.parameters.projection.current_beta # Current beta value
             _, d_proj = compute_heaviside_projection(x_tilde, beta, self.opt.parameters.projection.eta) # Projected design variables
             grad = grad * d_proj # Chain Rule for projection
        if self.opt.H is not None: # Check if density filter is used
             grad = self.opt.H.T.dot(grad) # Chain Rule for density filter
        return c, grad # Return compliance and its gradient
       
    def _smooth_max(self, x: np.ndarray, p: float, form_def: str) -> Tuple[float, np.ndarray]: # Smooth max function
        """
        Computes smooth approximation of max(x).
        Matches MATLAB smooth_max.m
        Returns: S, dSdx
        """
        if form_def == 'p-norm': # p-norm
            sum_pow = np.sum(x ** p) # Sum of x^p
            S = (sum_pow ** (1.0 / p)) # Smooth max
            dSdx = (x / S) ** (p - 1) # Derivative of smooth max
            return S, dSdx # Return smooth max and its gradient
        elif form_def == 'KS': # KS
            mx = np.max(x) # Maximum value
            epxm = np.exp(p * (x - mx)) # Exponential of p times (x - mx)
            sum_epxm = np.sum(epxm) # Sum of exponential
            S = mx + np.log(sum_epxm) / p # Smooth max
            dSdx = epxm / sum_epxm # Derivative of smooth max
            return S, dSdx # Return smooth max and its gradient
            
        elif form_def == 'LKS': # LKS
             # max(x) + log(sum(exp)/N)/p
             mx = np.max(x)
             epxm = np.exp(p * (x - mx))
             sum_epxm = np.sum(epxm)
             N = x.size
             S = mx + np.log(sum_epxm / N) / p # Smooth max
             dSdx = epxm / sum_epxm # Derivative of smooth max
             return S, dSdx # Return smooth max and its gradient
        elif form_def == 'softmax': # softmax
            mx = np.max(x) # Maximum value
            epxm = np.exp(p * (x - mx)) # Exponential of p times (x - mx)
            sum_epxm = np.sum(epxm) # Sum of exponential
            S = np.sum(x * epxm) / sum_epxm # Smooth max
            dSdx = epxm * (1.0 + p * (x - S)) / sum_epxm # Derivative of smooth max
            return S, dSdx # Return smooth max and its gradient
        else: # Unknown form
            raise ValueError(f"Unknown smooth_max form: {form_def}")

    def compute_max_stress_violation(self, x: np.ndarray, u: np.ndarray) -> Tuple[float, np.ndarray]: # Compute maximum stress violation
        """Computes aggregated maximum stress violation using P-norm or MRF w.r.t design variable x."""        
        x_tilde, x_phys = self._get_physical_vars(x) # Get physical variables
        # ============
        se = np.zeros((self.fe.n_elem, self.fe.nloads)) # Elementwise stress violation
        C = self.fe.C_mat # Elasticity matrix
        V = self.fe.V       # V matrix for stress calculation
        # Get material properties
        p = self.opt.parameters.penalization_param # Penalization parameter
        rho_min = self.fe.material.rho_min # Minimum density
        dpen_rho_e = (1.0 - rho_min) * p * ((x_phys + 1e-12) ** (p - 1.0)) # Derivative of penalty w.r.t density
        for iload in range(self.fe.nloads): # Loop over loads
            Ul = self.fe.U[:, iload] # (n_dof,)
            Ue = Ul[self.fe.edof_mat.astype(int)] # (n_elem, n_dof)
            eps = np.einsum('nij, nj -> ni', self.fe.B0e, Ue) # (n_elem, 3)
            sigma = eps @ C.T 
            temp = sigma @ V
            val = np.sum(sigma * temp, axis=1)
            se[:, iload] = np.sqrt(np.maximum(val, 0)) # elementwise von mises stress
        # Elementwise relaxation
        q = self.opt.parameters.relaxation_param # Power law for stress relaxation
        re = x_phys ** q # Elementwise relaxation
        dredrhof = q * ((x_phys + 1e-12) ** (q - 1.0)) # gradient of relaxation
        limit = self.opt.parameters.slimit
        se_relaxed = se * re[:, None] # stress relaxation
        self.fe.svm = se_relaxed # Store relaxed stress
        P = self.opt.parameters.aggregation_parameter # aggregation parameter
        h = se_relaxed / limit # Elementwise stress violation
        dhds = np.ones_like(h) / limit # gradient of stress violation
        phi = h.copy() # Smooth max input
        dphidh = np.ones_like(h)  
        g, dgdphi = self._smooth_max(phi.reshape(-1, 1), P, form_def='p-norm') # Smooth max
        dgdphi = dgdphi.reshape(h.shape) # gradient of smooth max
        g = g - 1.0 # Smooth max constraint        
        # SENSITIVITY  
        self.fe.dJdu = np.zeros((self.fe.n_dof, self.fe.nloads)) # Initialize adjoint force
        CV = C @ V # Precompute CV
        # start loop over load cases
        for iload in range(self.fe.nloads):
             se_safe = se[:, iload].copy()  # Elementwise stress relaxation
             A = dgdphi[:, iload] * dphidh[:, iload] * dhds[:, iload] * re # (n_elem,)
             A = A / (se_safe + 1e-12) # (n_elem,) avoid division by zero             
             # Recalculate sigma for this load case
             Ul = self.fe.U[:, iload] # displacement vector for this load case
             Ue = Ul[self.fe.edof_mat.astype(int)] # elementwise displacement
             eps = np.einsum('nij, nj -> ni', self.fe.B0e, Ue) # strain vector for this load case
             sigma_l = eps @ C.T # stress vector for this load case             
             term = sigma_l @ CV.T # 
             v_elem = np.einsum('nij, ni -> nj', self.fe.B0e, term) #  
             fe_adj = -1.0 * A[:, None] * v_elem #  
             np.add.at(self.fe.dJdu[:, iload], self.fe.edof_mat.astype(int).flatten(), fe_adj.flatten()) # Add to adjoint force

        # Solve Adjoint
        Lambda = self.problem.solver.solve(self.fe.K, self.fe.dJdu, self.fe.fixeddofs_ind, analysis_type = 'adjoint')
        # Sensitivity Computation (lTdku)
        lTdku = np.zeros((self.fe.n_elem, self.fe.nloads)) # Initialize lTdku
        for iload in range(self.fe.nloads):
             Le = Lambda[self.fe.edof_mat.astype(int), iload] # adjoint solution
             Ue = self.fe.U[self.fe.edof_mat.astype(int), iload] # elementwise displacement
             Ke_Ue = np.einsum('nij, nj -> ni', self.fe.Ke, Ue) # 
             val = np.einsum('ni, ni -> n', Le, Ke_Ue)
             lTdku[:, iload] = dpen_rho_e * val # sensitivity
             
        # Final Summation
        grad_g = np.zeros(self.fe.n_elem) # Initialize gradient
        for iload in range(self.fe.nloads):
            term1 = dgdphi[:, iload] * dphidh[:, iload] * dhds[:, iload] * (se[:, iload] * dredrhof) # 
            grad_g += term1 + lTdku[:, iload]             
        # Filter Logic
        # Chain Rule for Projection and Filter
        if self.opt.parameters.projection.use: # Check use of Heaviside projection
             beta = self.opt.parameters.projection.current_beta
             _, d_proj = compute_heaviside_projection(x_tilde, beta, self.opt.parameters.projection.eta)
             grad_g = grad_g * d_proj # Apply projection chain rule
        
        if self.opt.H is not None: # Check if filter is active
             grad_g = self.opt.H.T.dot(grad_g) # Apply filter chain rule
          
        # Store results
        self.opt.approx_h_max = g + 1.0 # Approximate h_max
        self.opt.true_h_max = np.max(h) # True h_max
        self.opt.true_stress_max = np.max(se_relaxed) # True stress max
        self.opt.grad_stress = grad_g # Gradient of stress
        return g, grad_g # Return constraint and gradient
    def compute_local_stress_penalty(self, x: np.ndarray, u: np.ndarray, mu: float) -> Tuple[float, np.ndarray]:
        """Computes the ALM penalty and gradient directly on local element stresses (no p-norm)."""
        x_tilde, x_phys = self._get_physical_vars(x)
        
        se = np.zeros((self.fe.n_elem, self.fe.nloads))
        C = self.fe.C_mat
        V = self.fe.V
        p = self.opt.parameters.penalization_param
        rho_min = self.fe.material.rho_min
        dpen_rho_e = (1.0 - rho_min) * p * ((x_phys + 1e-12) ** (p - 1.0))
        
        for iload in range(self.fe.nloads):
            Ul = self.fe.U[:, iload]
            Ue = Ul[self.fe.edof_mat.astype(int)]
            eps = np.einsum('nij, nj -> ni', self.fe.B0e, Ue)
            sigma = eps @ C.T 
            temp = sigma @ V
            val = np.sum(sigma * temp, axis=1)
            se[:, iload] = np.sqrt(np.maximum(val, 0))
            
        q = self.opt.parameters.relaxation_param
        re = x_phys ** q
        dredrhof = q * ((x_phys + 1e-12) ** (q - 1.0))
        limit = self.opt.parameters.slimit
        se_relaxed = se * re[:, None]
        self.fe.svm = se_relaxed
        
        h = se_relaxed / limit - 1.0
        
        # Initialize local ALM multipliers if they don't exist
        if not hasattr(self.opt, 'lam_local') or self.opt.lam_local is None:
            self.opt.lam_local = np.zeros(h.shape)
            
        lam_local = self.opt.lam_local
        
        # Store for the optimizer to update at the end of the epoch
        self.opt.local_h_e = h.copy()
        
        # Local active constraint: c_e = max(0, lambda_e + mu * h_e)
        c_e = np.maximum(0.0, lam_local + mu * h)
        
        # Scalar Penalty: P = sum(1/(2*mu) * (c_e^2 - lambda_e^2))
        P_AL = np.sum((c_e**2 - lam_local**2) / (2.0 * mu))
        
        # Derivative of P_AL w.r.t h_e is exactly c_e
        dP_dh = c_e
        
        self.fe.dJdu = np.zeros((self.fe.n_dof, self.fe.nloads))
        CV = C @ V
        
        for iload in range(self.fe.nloads):
             se_safe = se[:, iload].copy()
             # A = dP_dh * dh_ds * ds_relax
             # dh_ds = 1 / limit
             A = dP_dh[:, iload] * (1.0 / limit) * re
             A = A / (se_safe + 1e-12)
             
             Ul = self.fe.U[:, iload]
             Ue = Ul[self.fe.edof_mat.astype(int)]
             eps = np.einsum('nij, nj -> ni', self.fe.B0e, Ue)
             sigma_l = eps @ C.T
             term = sigma_l @ CV.T
             v_elem = np.einsum('nij, ni -> nj', self.fe.B0e, term)
             fe_adj = -1.0 * A[:, None] * v_elem
             np.add.at(self.fe.dJdu[:, iload], self.fe.edof_mat.astype(int).flatten(), fe_adj.flatten())

        Lambda = self.problem.solver.solve(self.fe.K, self.fe.dJdu, self.fe.fixeddofs_ind, analysis_type='adjoint')
        
        lTdku = np.zeros((self.fe.n_elem, self.fe.nloads))
        for iload in range(self.fe.nloads):
             Le = Lambda[self.fe.edof_mat.astype(int), iload]
             Ue = self.fe.U[self.fe.edof_mat.astype(int), iload]
             Ke_Ue = np.einsum('nij, nj -> ni', self.fe.Ke, Ue)
             val = np.einsum('ni, ni -> n', Le, Ke_Ue)
             lTdku[:, iload] = dpen_rho_e * val
             
        grad_P = np.zeros(self.fe.n_elem)
        for iload in range(self.fe.nloads):
            # local relaxation derivative term
            term1 = dP_dh[:, iload] * (1.0 / limit) * (se[:, iload] * dredrhof)
            grad_P += term1 + lTdku[:, iload]
            
        if self.opt.parameters.projection.use:
             beta = self.opt.parameters.projection.current_beta
             _, d_proj = compute_heaviside_projection(x_tilde, beta, self.opt.parameters.projection.eta)
             grad_P = grad_P * d_proj
             
        if self.opt.H is not None:
             grad_P = self.opt.H.T.dot(grad_P)
             
        # Store results for display
        self.opt.true_stress_max = np.max(se_relaxed)
        self.opt.true_h_max = np.max(h)
        self.opt.approx_h_max = np.max(h)
        self.opt.grad_stress = grad_P
        
        return P_AL, grad_P
    # ============================================================
