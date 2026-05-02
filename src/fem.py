import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
from .struct_prob import Problem
from .utils import compute_simp, compute_modified_simp

class FEMSolver: # FEMSolver class
    def __init__(self, problem: Problem): # Initialize with Problem instance
        self.problem = problem # Problem instance
        self.fe = problem.fe # FEM instance
        
    def initialize(self): # Initialize FE
        self.init_FE() # Call init_FE

    def init_FE(self): # Initialize the Finite Element structure
        """
        Initialize the Finite Element structure.
        """        
        # Switch mesh input type
        mesh_type = self.fe.mesh_input.type # Mesh type
        
        if mesh_type == 'generate': # Generate mesh
            self.generate_mesh() # Generate mesh
        elif mesh_type == 'read-home-made':
            raise NotImplementedError("read-home-made not implemented yet.")
        elif mesh_type == 'read-gmsh':
             raise NotImplementedError("read-gmsh not implemented yet.")
        elif mesh_type == '2DLbracket': # L-bracket mesh
            self.gen_lbracket_mesh() # Generate L-bracket mesh
        elif mesh_type == 'double2DLb':
            raise NotImplementedError("double2DLb not implemented yet.")
        elif mesh_type == 'V-frame':
             raise NotImplementedError("V-frame not implemented yet.")
        else:
            raise ValueError(f"Unidentified mesh input type: {mesh_type}")
            
        # Ensure n_dof is set
        self.fe.n_dof = self.fe.n_node * self.fe.dim # Number of degrees of freedom
        
        # Compute element info
        self.compute_element_info() # Compute element info
        
        # Setup boundary conditions
        if hasattr(self.fe.mesh_input, 'bcs_file') and self.fe.mesh_input.bcs_file: # Check if bcs_file exists
             bcs = self.fe.mesh_input.bcs_file # BCS file
             if callable(bcs): # Check if callable
                 bcs(self.problem) # Call BCS function
             elif bcs == 'setup_lbracket_bcs': # L-bracket BCS
                  from .setup_bcs import setup_lbracket_bcs # Import setup_lbracket_bcs
                  setup_lbracket_bcs(self.problem) # Call setup_lbracket_bcs
             else:
                  print(f"Warning: bcs_file '{bcs}' not recognized or actionable in init_FE.")
        
        # initialize the fixed/free partitioning scheme:
        self.FE_init_partitioning() # Initialize partitioning
        
        # assemble the boundary conditions
        self.FE_assemble_BC() # Assemble BCs
        
        # compute elastic coefficients
        self.compute_constitutive_matrix() # Compute C matrix
        
        # compute the element stiffness matrices
        self.compute_element_stiffness() # Compute Ke
    
    def generate_mesh(self): # Generates a rectangular/cuboid mesh
        """Generates a rectangular/cuboid mesh."""
        dims = self.fe.mesh_input.box_dimensions # Box dimensions
        els = self.fe.mesh_input.elements_per_side # Elements per side
        
        if not dims or not els: # Check if dimensions are provided
            raise ValueError("box_dimensions and elements_per_side must be provided for 'generate' mesh type.")
            
        self.fe.dim = len(dims) # Dimension
        self.fe.n_elem = int(np.prod(els)) # Total elements
        n_node = int(np.prod([e + 1 for e in els])) # Total nodes
        self.fe.n_node = n_node # Store number of nodes
        
        # Create nodal coordinates
        coords_list = [np.linspace(0, d, e + 1) for d, e in zip(dims, els)] # Coordinates list
        self.fe.max_elem_side = np.max([d / e for d, e in zip(dims, els)]) # Max element side
        
        if self.fe.dim == 2: # 2D Case
            xx, yy = np.meshgrid(coords_list[0], coords_list[1]) # default is usually 'xy', xx varies along dim 1, yy along dim 0? 
            self.fe.coords = np.column_stack((xx.flatten('F'), yy.flatten('F'))) # Nodal coordinates
            
        elif self.fe.dim == 3:
            xx, yy, zz = np.meshgrid(coords_list[0], coords_list[1], coords_list[2])
            self.fe.coords = np.column_stack((xx.flatten('F'), yy.flatten('F'), zz.flatten('F')))
        else:
             raise NotImplementedError("Only 2D and 3D are supported.")
            
        # Connectivity
        # Standard Q4 or Hex8
        nelx, nely = els[0], els[1]
        
        if self.fe.dim == 2:            
            # Python:
            row = np.arange(1, nely + 1).reshape(-1, 1) # (nely, 1)
            col = np.arange(1, nelx + 1).reshape(1, -1) # (1, nelx)            
            idx_row = np.arange(nely).reshape(-1, 1)
            idx_col = np.arange(nelx).reshape(1, -1)
            
            n1 = idx_row + idx_col * (nely + 1)
            n2 = idx_row + (idx_col + 1) * (nely + 1)
            n3 = n2 + 1
            n4 = n1 + 1
            
            # elem_mat = [n1(:), n2(:), n3(:), n4(:)]
            # Flatten 'F' to match the column-major ordering of elements in MATLAB logic
            self.fe.elem_node = np.column_stack((
                n1.flatten('F'), n2.flatten('F'), n3.flatten('F'), n4.flatten('F')
            )).astype(int)
            
        elif self.fe.dim == 3:
            nelx, nely, nelz = els[0], els[1], els[2]
            
            idx_row = np.arange(nely).reshape(-1, 1, 1) # y
            idx_col = np.arange(nelx).reshape(1, -1, 1) # x
            idx_lay = np.arange(nelz).reshape(1, 1, -1) # z
            
            # Node index at (r, c, l) (0-based)
            # n = r + c * ny_nodes + l * ny_nodes * nx_nodes
            
            n1 = idx_row     + idx_col * ny_nodes      + idx_lay * (ny_nodes * nx_nodes)
            n2 = idx_row     + (idx_col + 1) * ny_nodes + idx_lay * (ny_nodes * nx_nodes)
            n3 = n2 + 1
            n4 = n1 + 1
            
            n5 = n1 + (ny_nodes * nx_nodes)
            n6 = n2 + (ny_nodes * nx_nodes)
            n7 = n3 + (ny_nodes * nx_nodes)
            n8 = n4 + (ny_nodes * nx_nodes)
            
            # Vectorized listing
            self.fe.elem_node = np.column_stack((
                n1.flatten('F'), n2.flatten('F'), n3.flatten('F'), n4.flatten('F'),
                n5.flatten('F'), n6.flatten('F'), n7.flatten('F'), n8.flatten('F')
            )).astype(int)

    def gen_lbracket_mesh(self): # Generates L-bracket mesh
        """Generates L-bracket mesh."""
        W = self.fe.mesh_input.L_side # Width
        S = self.fe.mesh_input.L_cutout # Cutout size
        h = self.fe.mesh_input.L_element_size # Element size
        
        if not all([W, S, h]): # Check if inputs are valid
             raise ValueError("L_side, L_cutout, and L_element_size must be provided.")
             
        if (W % h > 1e-9) or (S % h > 1e-9): # Check divisions
             print("Warning: Element size is not an exact divisor.")
             
        self.fe.max_elem_side = h # Store element size
             
        # Nodes
        xcoords = np.arange(0, W + h/2, h)
        ycoords = np.arange(0, W + h/2, h)
        
        nodes = []
        node_map = {} # (x, y) -> index
        inode = 0
        
        for x in xcoords:
            for y in ycoords:
                # Cutout check: Top-Right is void.
                # x > W-S and y > W-S
                # Be careful with floating point
                if (x > (W - S + 1e-9)) and (y > (W - S + 1e-9)):
                    continue
                
                nodes.append([x, y])
                inode += 1
                
        self.fe.coords = np.array(nodes) # Store nodes
        self.fe.n_node = inode # Number of nodes
        
        elems = [] # Initialize elements list
        
        # Grid indices (0-based)
        nx = len(xcoords)
        ny = len(ycoords)
        
        # Number of nodes in a FULL column
        nnode_W = ny 
        
        # Build map
        tol = 1e-6
        coord_to_id = {}
        for idx, node in enumerate(nodes):
             k = (int(round(node[0]/tol)), int(round(node[1]/tol)))
             coord_to_id[k] = idx
             
        # Generate elements
        # Loop over grid cells
        # x_i, y_j. Element is [x_i, x_i+1] x [y_j, y_j+1]
        
        for i in range(len(xcoords) - 1):
            for j in range(len(ycoords) - 1):
                x = xcoords[i]
                y = ycoords[j]
                
                # Check if this element exists (centroid not in cutout)
                xc = x + h/2
                yc = y + h/2
                if (xc > (W - S + 1e-9)) and (yc > (W - S + 1e-9)):
                    continue                
                # Let's find IDs
                def get_id(xx, yy):
                    k = (int(round(xx/tol)), int(round(yy/tol)))
                    return coord_to_id[k]
                
                try:
                    n1 = get_id(x, y)
                    n2 = get_id(x + h, y)
                    n3 = get_id(x + h, y + h)
                    n4 = get_id(x, y + h)
                    elems.append([n1, n2, n3, n4])
                except KeyError:
                    # Should not happen if logic is consistent
                    raise RuntimeError("Element node not found.")
                    
        self.fe.elem_node = np.array(elems, dtype=int)
        self.fe.n_elem = len(elems)
        self.fe.dim = 2


    def compute_element_info(self): # Computes element volumes, centroids, etc.
        """Computes element volumes, centroids, etc."""
        # Vectorized implementation
        dim = self.fe.dim # Dimension
        coords = self.fe.coords # (Nn, dim)
        elems = self.fe.elem_node # (Ne, nodes_per_elem)
        # Gather coords for all elements
        el_coords = coords[elems] # Element coordinates
        
        # Centroids: mean of nodes
        self.fe.centroids = np.mean(el_coords, axis=1).T # (dim, Ne)
        # Volumes
        if dim == 2: # 2D Case
            self.fe.V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]]) # V matrix for stress computation
            n1 = el_coords[:, 0, :]
            n2 = el_coords[:, 1, :]
            n3 = el_coords[:, 2, :]
            n4 = el_coords[:, 3, :]
            
            d1 = n3 - n1
            d2 = n4 - n2
            
            # 2D cross product: x1*y2 - x2*y1
            cross = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0] # Cross product
            self.fe.elem_vol = 0.5 * np.abs(cross) # Element volume
            
            # max element side/diag
            def dist(a, b): return np.linalg.norm(a - b, axis=1)
            
            s1 = dist(n2, n1)
            s2 = dist(n3, n2)
            s3 = dist(n4, n3)
            s4 = dist(n1, n4)
            d1_len = dist(n3, n1)
            d2_len = dist(n4, n2)
            
            self.fe.max_elem_side = np.max([s1, s2, s3, s4])
            # self.fe.max_elem_diag = np.max([d1_len, d2_len]) # if needed
            
        elif dim == 3:
            self.fe.V = np.array([
                [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
                [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
            ])
            n1 = el_coords[:, 0, :]
            n2 = el_coords[:, 1, :]
            n4 = el_coords[:, 3, :]
            n5 = el_coords[:, 4, :]
            
            dx = n2[:, 0] - n1[:, 0]
            dy = n4[:, 1] - n1[:, 1]
            dz = n5[:, 2] - n1[:, 2]
            
            self.fe.elem_vol = np.abs(dx * dy * dz) # Element volume
            self.fe.max_elem_side = np.max([dx.max(), dy.max(), dz.max()]) # Max element side
            
        # Compute edof_mat (Topology)
        n_elem = self.fe.n_elem
        elems = self.fe.elem_node
        
        if dim == 2:
            n_dof_per_node = 2
            nodes_per_elem = 4
            edof = n_dof_per_node * nodes_per_elem
            
            dof_indices = np.zeros((n_elem, edof), dtype=int) # DOF indices
            for i in range(nodes_per_elem):
                dof_indices[:, 2*i] = 2 * elems[:, i]
                dof_indices[:, 2*i+1] = 2 * elems[:, i] + 1
            
            self.fe.edof_mat = dof_indices # Store DOF indices
            
        elif dim == 3:
            n_dof_per_node = 3
            nodes_per_elem = 8
            edof = n_dof_per_node * nodes_per_elem
            
            dof_indices = np.zeros((n_elem, edof), dtype=int)
            for i in range(nodes_per_elem):
                for k in range(3):
                    dof_indices[:, 3*i + k] = 3 * elems[:, i] + k
            
            self.fe.edof_mat = dof_indices

    def compute_constitutive_matrix(self): # Computes the elasticity matrix C
        """Computes the elasticity matrix C."""
        E = self.fe.material.E # Young's modulus
        nu = self.fe.material.nu # Poisson's ratio
        dim = self.fe.dim # Dimension
        
        if dim == 2: # 2D Case
            # Plane Stress
            self.fe.C_mat = (E / (1 - nu**2)) * np.array([ # Elasticity matrix
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
            
            # Von Mises matrix V for 2D Plane Stress
            self.fe.V = np.array([ # V matrix
                [1.0, -0.5, 0.0],
                [-0.5, 1.0, 0.0],
                [0.0, 0.0, 3.0]
            ])
            
        elif dim == 3:
            # Isotropic 3D: 6x6 matrix
            factor = E / ((1 + nu) * (1 - 2 * nu))
            c1 = 1 - nu
            c2 = nu
            c3 = (1 - 2 * nu) / 2
            
            C = np.zeros((6, 6))
            C[0:3, 0:3] = c2
            np.fill_diagonal(C[0:3, 0:3], c1)
            C[3, 3] = c3
            C[4, 4] = c3
            C[5, 5] = c3
            
            self.fe.C_mat = factor * C
            
            # Von Mises V for 3D
            V = np.zeros((6, 6))
            V[0:3, 0:3] = -0.5
            np.fill_diagonal(V, 1.0)
            V[3, 3] = 3.0
            V[4, 4] = 3.0
            V[5, 5] = 3.0
            
            self.fe.V = V
        else:
             raise ValueError("Dimension must be 2 or 3")

    def compute_element_stiffness(self): # Computes element stiffness matrices Ke
        """Computes element stiffness matrices Ke (fully solid)."""
        if self.fe.C_mat is None: # Check if C matrix exists
            self.compute_constitutive_matrix() # Compute C matrix
            
        dim = self.fe.dim # Dimension
        coords = self.fe.coords # Coordinates
        elems = self.fe.elem_node # Elements
        n_elem = self.fe.n_elem # Number of elements
        C = self.fe.C_mat # Elasticity matrix
        
        # Gauss quadrature
        gauss_pt = np.array([-1, 1]) / np.sqrt(3)
        weights = np.array([1, 1])
        
        if dim == 2: # 2D Case
            n_dof_per_node = 2 # DOFs per node
            nodes_per_elem = 4 # Nodes per element
            edof = n_dof_per_node * nodes_per_elem # Total DOFs per element
            
            Ke = np.zeros((n_elem, edof, edof)) # Stiffness matrices
            B0e = np.zeros((n_elem, 3, edof)) # Strain-disp at centroid
            
            # Shape function derivatives in parent coordinates
            def get_G0_N(xi, eta): # Derivatives for 4-node Quad element (Q4)
                return 0.25 * np.array([
                    [eta - 1, 1 - eta, 1 + eta, -eta - 1],
                    [xi - 1, -xi - 1, 1 + xi, 1 - xi]
                ])

            el_coords = coords[elems]
            
            elem_vol = np.zeros(n_elem)
            # Loop over gauss points
            for xi, w_xi in zip(gauss_pt, weights): # Loop over xi
                for eta, w_eta in zip(gauss_pt, weights): # Loop over eta
                    
                    G0 = get_G0_N(xi, eta) # Shape function derivatives
                    J = np.matmul(G0, el_coords) # Jacobian
                    det_J = np.linalg.det(J) # Determinant of Jacobian
                    inv_J = np.linalg.inv(J) # Inverse of Jacobian
                    
                    GN = np.matmul(inv_J, np.tile(G0, (n_elem, 1, 1))) # Shape function derivatives in global coords
                    
                    B = np.zeros((n_elem, 3, 8)) # B matrix
                    B[:, 0, 0::2] = GN[:, 0, :] # dN/dx
                    B[:, 1, 1::2] = GN[:, 1, :] # dN/dy
                    B[:, 2, 0::2] = GN[:, 1, :]
                    B[:, 2, 1::2] = GN[:, 0, :]
                    
                    wdJ = w_xi * w_eta * det_J # Weighted determinant
                    elem_vol += wdJ # Accumulate volume
                    CB = np.matmul(C, B) # C*B
                    BTCB = np.matmul(B.transpose(0, 2, 1), CB) # B^T * C * B
                    
                    Ke += wdJ[:, None, None] * BTCB # Accumulate stiffness
            
            self.fe.elem_vol = elem_vol
            
            # Compute B0 at centroid (0,0)
            G0_0 = get_G0_N(0, 0)
            J0 = np.matmul(G0_0, el_coords)
            inv_J0 = np.linalg.inv(J0)
            GN0 = np.matmul(inv_J0, np.tile(G0_0, (n_elem, 1, 1)))
            B0 = np.zeros((n_elem, 3, 8))
            B0[:, 0, 0::2] = GN0[:, 0, :]
            B0[:, 1, 1::2] = GN0[:, 1, :]
            B0[:, 2, 0::2] = GN0[:, 1, :]
            B0[:, 2, 1::2] = GN0[:, 0, :]
            
            self.fe.Ke = Ke # Store stiffness matrices
            self.fe.B0e = B0 # Store B0 matrices

        elif dim == 3:
            n_dof_per_node = 3
            nodes_per_elem = 8
            edof = n_dof_per_node * nodes_per_elem
            
            Ke = np.zeros((n_elem, edof, edof))
            B0e = np.zeros((n_elem, 6, edof)) # 3D Strain: 6 components
            
            # Gauss Quadrature 2x2x2
            gauss_pts = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
            weights = [1.0, 1.0]    
            
            def get_GN_3D(xi, eta, zeta): # Derivatives for 8-node Hex element (Hex8)
                node_signs = [
                    [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                    [-1,-1,1],  [1,-1,1],  [1,1,1],  [-1,1,1]
                ]
                GN = np.zeros((3, 8))
                for i in range(8):
                    xi_i, eta_i, zeta_i = node_signs[i]
                    GN[0, i] = 0.125 * xi_i * (1 + eta * eta_i) * (1 + zeta * zeta_i)
                    GN[1, i] = 0.125 * eta_i * (1 + xi * xi_i) * (1 + zeta * zeta_i)
                    GN[2, i] = 0.125 * zeta_i * (1 + xi * xi_i) * (1 + eta * eta_i)
                return GN

            # Element 0 Coords for Jacobian (Assuming Uniform)
            # Or vectorized over all elements
            el_coords = coords[elems] # (n_elem, 8, 3)
            
            # Loop
            # Ke_local accumulator? 
            # No, standard loop structure:
            Ke_accum = np.zeros((n_elem, edof, edof))
            
            for xi, w_xi in zip(gauss_pts, weights):
                for eta, w_eta in zip(gauss_pts, weights):
                    for zeta, w_zeta in zip(gauss_pts, weights):
                        G0 = get_GN_3D(xi, eta, zeta) # (3, 8)                     
                       
                        J = np.einsum('jk, nkl -> njl', G0, el_coords)
                        
                        det_J = np.linalg.det(J) # (N,)
                        inv_J = np.linalg.inv(J) # (N, 3, 3)``
                        
       
                        GN = np.matmul(inv_J, G0)
                        
                        # B matrix (N, 6, 24)
                        B = np.zeros((n_elem, 6, 24))
            
                        
                        # Automated fill
                        for i in range(8):
                            col = 3 * i
                            dN_dx = GN[:, 0, i]
                            dN_dy = GN[:, 1, i]
                            dN_dz = GN[:, 2, i]
                            
                            B[:, 0, col] = dN_dx   # eps_x
                            B[:, 1, col+1] = dN_dy # eps_y
                            B[:, 2, col+2] = dN_dz # eps_z
                            B[:, 3, col] = dN_dy   # gam_xy
                            B[:, 3, col+1] = dN_dx
                            B[:, 4, col+1] = dN_dz # gam_yz
                            B[:, 4, col+2] = dN_dy
                            B[:, 5, col] = dN_dz   # gam_zx
                            B[:, 5, col+2] = dN_dx
                            
                        # CB (N, 6, 24)
                        # C is (6,6). 
                        CB = np.matmul(self.fe.C_mat, B)
                        
                        # B^T C B
                        BTCB = np.matmul(B.transpose(0, 2, 1), CB) # (N, 24, 24)
                        
                        scale = w_xi * w_eta * w_zeta * det_J
                        Ke_accum += scale[:, None, None] * BTCB
                        
            # B0e centroid
            G0 = get_GN_3D(0.0, 0.0, 0.0)
            J = np.einsum('jk, nkl -> njl', G0, el_coords)
            inv_J = np.linalg.inv(J)
            GN = np.matmul(inv_J, np.tile(G0, (n_elem, 1, 1)))
            
            for i in range(8):
                col = 3 * i
                dN_dx = GN[:, 0, i]
                dN_dy = GN[:, 1, i]
                dN_dz = GN[:, 2, i]
                
                B0e[:, 0, col] = dN_dx
                B0e[:, 1, col+1] = dN_dy
                B0e[:, 2, col+2] = dN_dz
                B0e[:, 3, col] = dN_dy
                B0e[:, 3, col+1] = dN_dx
                B0e[:, 4, col+1] = dN_dz
                B0e[:, 4, col+2] = dN_dy
                B0e[:, 5, col] = dN_dz
                B0e[:, 5, col+2] = dN_dx

            self.fe.Ke = Ke_accum
            self.fe.B0e = B0e
        else:
             raise ValueError("Unsupported dimension")
    def FE_init_partitioning(self): # Initialize the fixed/free partitioning scheme
        n_global_dof = self.fe.dim*self.fe.n_node # Total DOFs
        self.fe.fixeddofs = np.zeros(n_global_dof, dtype=bool) # Fixed DOFs mask
        if hasattr(self.fe, 'fixed_dofs') and self.fe.fixed_dofs is not None: # Check if fixed_dofs exists
             self.fe.fixeddofs[self.fe.fixed_dofs] = True # Set fixed DOFs
        elif hasattr(self.fe, 'BC') and self.fe.BC is not None:
            if self.fe.dim == 2:
                fixed_indices = []
                
                nodes = self.fe.BC.disp_node
                dofs = self.fe.BC.disp_dof
                
                # Check 1 (x)
                mask_x = (dofs == 1)
                if np.any(mask_x):
                    idx_x = 2 * nodes[mask_x]
                    fixed_indices.append(idx_x)
                    
                # Check 2 (y)
                mask_y = (dofs == 2)
                if np.any(mask_y):
                    idx_y = 2 * nodes[mask_y] + 1
                    fixed_indices.append(idx_y)
                
                if fixed_indices:
                    all_fixed = np.concatenate(fixed_indices).astype(int)
                    self.fe.fixeddofs[all_fixed] = True
            else:
                # 3D
                nodes = self.fe.BC.disp_node
                dofs = self.fe.BC.disp_dof
                
                fixed_indices = []
                # 1->x (3*n), 2->y (3*n+1), 3->z (3*n+2)
                for d_chk, offset in zip([1, 2, 3], [0, 1, 2]):
                    mask = (dofs == d_chk)
                    if np.any(mask):
                        idx = 3 * nodes[mask] + offset
                        fixed_indices.append(idx)
                        
                if fixed_indices:
                    all_fixed = np.concatenate(fixed_indices).astype(int)
                    self.fe.fixeddofs[all_fixed] = True
        else:
             pass
        
        self.fe.freedofs = ~self.fe.fixeddofs # Free DOFs mask
        self.fe.fixeddofs_ind = np.where(self.fe.fixeddofs)[0] # Fixed DOFs indices
        self.fe.freedofs_ind = np.where(self.fe.freedofs)[0] # Free DOFs indices
        self.fe.n_free_dof = len(self.fe.freedofs_ind) # Number of free DOFs

        # Vectorized assembly indices
        # Ensure edof_mat is present (from compute_element_stiffness)
        if not hasattr(self.fe, 'edof_mat'):
             raise RuntimeError("edof_mat not found. Run compute_element_stiffness first.")
             
        edof = self.fe.edof_mat # (Ne, n_elem_dof)
        n_elem_dof = edof.shape[1]
        
        self.fe.iK = np.repeat(edof, n_elem_dof, axis=1).flatten()
        self.fe.jK = np.tile(edof, (1, n_elem_dof)).flatten()
        
    # =================
    def assemble_stiffness(self, densities: np.ndarray, penalization_param: float = 3.0, rho_min: float = 1e-3): # Assembles the global stiffness matrix
        """Assembles the global stiffness matrix (Modified SIMP)."""
        
        x = np.array(densities) # Density values
        penalized_rho_e = rho_min + (1.0 - rho_min) * (x ** penalization_param) # SIMP: Penalized densities
        
        # Ersatz material
        penalized_Ke = penalized_rho_e[:, None, None] * self.fe.Ke # Penalized stiffness matrices
        
        # Assemble global K (COO first)
        data = penalized_Ke.flatten() # Flatten data
        n_dof = self.fe.n_dof # Total DOFs
        
        K = sparse.coo_matrix((data, (self.fe.iK, self.fe.jK)), shape=(n_dof, n_dof)) # COO matrix
        K = K.tocsr() # Convert to CSR
        K = (K + K.T) / 2.0 # Symmetrize
        self.fe.K = K # Store global stiffness matrix
        return K # Return K
        
    def FE_analysis(self, densities: np.ndarray = None, penalization_param: float = 3.0, rho_min: float = 1e-3, analysis_type: str = 'primal'): # Assemble the global stiffness matrix and solve
        """
        Assemble the global stiffness matrix and solve the finite element analysis.
        Replaces MATLAB function FE_analysis.
        """
        # If densities provided, assemble K
        if densities is not None: # Check if densities are provided
             # If densities provided, update stiffness; else assume pre-assembled or uniform
             self.assemble_stiffness(densities, penalization_param, rho_min) # Assemble K
        
        if not hasattr(self.fe, 'K'): # Check if K exists
             raise RuntimeError("Stiffness matrix K not assembled using FE_analysis without arguments.")
             
        # Use fe.P as load vector if present, else fe.F
        load_vector = self.fe.P if self.fe.P is not None else self.fe.F # Load vector
        if load_vector is None: # Check if load vector exists
             raise RuntimeError("No load vector (P or F) found.")
             
        if self.fe.fixeddofs_ind is None: # Check if fixed DOFs are initialized
             self.FE_init_partitioning() # Initialize fixed DOFs
             
        # Call internal solve
        self.solve(self.fe.K, load_vector, self.fe.fixeddofs_ind, analysis_type) # Solve system

    def FE_assemble_BC(self): # Assembles prescribed displacements (U) and loads (P/rhs)
        """
        Assembles prescribed displacements (U) and loads (P/rhs).
        Replaces MATLAB function FE_assemble_BC.
        """
        # Initialize U
        self.fe.U = np.zeros((self.fe.n_dof, self.fe.nloads)) # Displacement vector
        
        # Prescribed Displacements
        if self.fe.BC is not None and self.fe.BC.n_pre_disp_dofs > 0: # Check if fixed DOFs exist
            nodes = self.fe.BC.disp_node # Fixed nodes
            dofs = self.fe.BC.disp_dof # Fixed DOFs
            values = self.fe.BC.disp_value # Fixed values
            
            # Vectorized assignment
            indices = self.fe.dim * nodes + (dofs - 1) # Calculate global DOF indices
            
            for i in range(len(indices)):
                idx = indices[i]
                val = values[i]
                self.fe.U[idx, :] = val # Assign fixed value

        # Prescribed Loads
        self.fe.P = np.zeros((self.fe.n_dof, self.fe.nloads)) # Load vector
        
        if self.fe.BC is not None:
             # My BC_Struct has flat arrays. force_id distinguishes cases.
             # I will use my `force_id` array to filter.
             
             if self.fe.BC.n_pre_force_dofs > 0:
                 nodes = self.fe.BC.force_node
                 dofs = self.fe.BC.force_dof
                 values = self.fe.BC.force_value
                 ids = self.fe.BC.force_id
                 
                 for icase in range(self.fe.nloads):
                     # Filter for current load case (icase + 1) using force_id
                     mask = (ids == (icase + 1))
                     if np.any(mask):
                         n = nodes[mask]
                         d = dofs[mask]
                         v = values[mask]
                         
                         idx = self.fe.dim * n + (d - 1)
                         self.fe.P[idx, icase] = v
                         
        # Map P to RHS for solver compatibility if needed
        # In solve(), we check logic.
        # Usually rhs = P set here.
        self.fe.rhs = self.fe.P[:, 0] # Default to first load case for non-multiload solvers or manage in solve later?
        # Optim loop handles multiload by slicing.
        # Actually in optim.py we pass F.
        
    def solve(self, K: sparse.csr_matrix, load_vector: np.ndarray, fixed_dofs: np.ndarray, analysis_type: str = 'primal') -> np.ndarray:
        """
        Solves Ku=f subject to fixed_dofs, following MATLAB structure.
        
        Args:
            K: Global stiffness matrix.
            load_vector: Force vector (F for primal) or dJdu (for adjoint).
            fixed_dofs: Indices of fixed degrees of freedom.
            analysis_type: 'primal' or 'adjoint'.
        """
        total_dofs = K.shape[0]
        
        p = fixed_dofs
        f = np.setdiff1d(np.arange(total_dofs), p)
        
        is_primal = (analysis_type.lower() == 'primal')
        is_adjoint = (analysis_type.lower() == 'adjoint')
        
        if not is_primal and not is_adjoint:
             print("Warning: Unrecognized FE analysis type.")
             
        # save the system RHS
        if is_primal:            
            self.fe.rhs = load_vector[f]
            self.fe.U = np.zeros((total_dofs, self.fe.nloads))
            self.fe.RF = np.zeros((total_dofs, self.fe.nloads))
        elif is_adjoint:
            self.fe.prhs = load_vector[f]
            self.fe.Lambda = np.zeros((total_dofs, self.fe.nloads))
            
        # Solve K_ff * u_f = rhs
        # Partition K for solution and reaction forces
        if is_primal:
             self.fe.Kpp = K[p, :][:, p]
             self.fe.Kfp = K[f, :][:, p]
        
        K_ff = K[f, :][:, f]       
        factor = splu(K_ff.tocsc()) 
        u_out = None
        if is_primal:           
            sol = factor.solve(self.fe.rhs)
            if sol.ndim == 1:
                sol = sol.reshape(-1, 1)           
            self.fe.U[f, :] = sol
            u_out = self.fe.U 

        elif is_adjoint:
            sol = factor.solve(self.fe.prhs)
            if sol.ndim == 1:
                sol = sol.reshape(-1, 1)
            self.fe.Lambda[f, :] = sol
            u_out = self.fe.Lambda 
        # ==============            
        if is_primal:
            # Compute reaction forces
            u_p = self.fe.U[self.fe.fixeddofs_ind,:]
            u_f = self.fe.U[self.fe.freedofs_ind,:]            
            reaction = self.fe.Kpp.dot(u_p) + self.fe.Kfp.T.dot(u_f)            
            # Update global force vector with reactions at fixed DOFs
            self.fe.RF[self.fe.fixeddofs_ind, :] = reaction            
        return u_out

