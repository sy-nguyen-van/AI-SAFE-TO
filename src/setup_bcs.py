import numpy as np

def setup_lbracket_bcs(problem):

    """
    Sets up boundary conditions for L-bracket matching user requirements.
    Populates fe.BC structure.
    """
    fe = problem.fe
    coords = fe.coords # (N, 2)
    # Ensure BC struct exists
    # Check if BC_Struct is available in core, need import inside or ensure it's loaded
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    l = fe.mesh_input.L_side
    c = fe.mesh_input.L_cutout
    tol = fe.max_elem_side / 1000.0
    
    # Coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    
    # TR_pt: Load Region
    # x > l-tol AND y > (7/8)*(l-c) - tol AND y < l-c + tol
    cond_x = x > (l - tol)
    cond_y_lower = y > ((7.0/8.0)*(l - c) - tol)
    cond_y_upper = y < ((l - c) + tol)
    
    TR_pt = np.where(cond_x & cond_y_lower & cond_y_upper)[0]
    
    # T_edge: Top Edge
    # y > l - tol
    T_edge = np.where(y > (l - tol))[0]
    
    # Load Case 1
    fe.nloads = 1
    net_mag1 = -3.0
    load_dir1 = 2 # y-direction (1=x, 2=y user convention)
    
    load_region1 = TR_pt
    if len(load_region1) > 0:
        load_mag1 = net_mag1 / len(load_region1)
    else:
        load_mag1 = 0.0
        print("Warning: No load nodes found (TR_pt is empty). check mesh.")
        
    # Build Load Matrix
    # We store as separate arrays in BC_Struct
    # User: [node, dir, mag, case]
    
    # For a single load case/region, easy to construct
    n_force = len(load_region1)
    fe.BC.n_pre_force_dofs = n_force
    fe.BC.force_node = load_region1 # np array of node indices
    fe.BC.force_dof = np.full(n_force, load_dir1, dtype=int)
    fe.BC.force_value = np.full(n_force, load_mag1)
    fe.BC.force_id = np.full(n_force, 1, dtype=int) # Load case 1
    
    # Displacement BCs
    disp_region = T_edge
    n_disp_pts = len(disp_region)
    
    # Constrain X (1) and Y (2)
    # disp_mat shape (2*n_disp_pts, 3)
    
    # Nodes repeated twice
    bc_nodes = np.tile(disp_region, 2)
    
    # Dirs: first set 1, second set 2
    bc_dirs = np.concatenate([np.ones(n_disp_pts, dtype=int), 2 * np.ones(n_disp_pts, dtype=int)])
    
    # Mags: all zero
    bc_mags = np.zeros(2 * n_disp_pts)
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Force Vector construction from BC (Optional if solver uses F directly, but good to populate fe.F)
    # In my fem.py implementation, I rely on fe.BC to set fixeddofs, but I usually expect fe.F to be set.
    # Existing fem.py does: F = fe.F if present.
    # So I should also populate fe.F based on fe.BC forces.
    
    # Initialize global force vector
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    
    # Apply forces
    # x-dof: 2*node, y-dof: 2*node+1
    # User dir 1->x, 2->y
    
    # Forces
    if n_force > 0:
        # Vectorized mapping
        # 1 -> 0 offset, 2 -> 1 offset
        offsets = fe.BC.force_dof - 1
        indices = 2 * fe.BC.force_node + offsets
        values = fe.BC.force_value
        
        # Add to F
        np.add.at(F, indices, values) # handle multiple loads on same dof if any
        
    fe.F = F
    fe.n_loads = 1


def setup_cantilever2d_bcs(problem):
    """
    Sets up BCs for 2D Cantilever Beam matching TOuNN (TipCantilever).
    """
    fe = problem.fe
    coords = fe.coords # (N, 2)
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    L = np.max(x)
    
    tol = 1e-6
    
    # L_edge: Left Edge (Fixed)
    L_edge = np.where(x < tol)[0]
    
    # Load Case 1
    fe.nloads = 1
    
    # Tip Cantilever: Load at bottom right corner (x=L, y=0)
    load_region1 = np.where((x > L - tol) & (y < tol))[0]
    
    if len(load_region1) > 0:
        load_mag1 = -1.0 # TOuNN uses -1
        load_dir1 = 2 # y-direction
        
        n_force = len(load_region1)
        fe.BC.n_pre_force_dofs = n_force
        fe.BC.force_node = load_region1
        fe.BC.force_dof = np.full(n_force, load_dir1, dtype=int)
        fe.BC.force_value = np.full(n_force, load_mag1 / n_force)
        fe.BC.force_id = np.full(n_force, 1, dtype=int)
    else:
        fe.BC.n_pre_force_dofs = 0
        fe.BC.force_node = np.array([], dtype=int)
        fe.BC.force_dof = np.array([], dtype=int)
        fe.BC.force_value = np.array([])
        fe.BC.force_id = np.array([], dtype=int)
        print("Warning: No load nodes found for 2D Cantilever.")

    # Displacement BCs
    disp_region = L_edge
    n_disp_pts = len(disp_region)
    
    # Constrain X (1) and Y (2)
    bc_nodes = np.tile(disp_region, 2)
    bc_dirs = np.concatenate([np.ones(n_disp_pts, dtype=int), 2 * np.ones(n_disp_pts, dtype=int)])
    bc_mags = np.zeros(2 * n_disp_pts)
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Initialize global force vector
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    
    if fe.BC.n_pre_force_dofs > 0:
        offsets = fe.BC.force_dof - 1
        indices = 2 * fe.BC.force_node + offsets
        values = fe.BC.force_value
        np.add.at(F, indices, values)
        
    fe.F = F
    fe.n_loads = 1


def setup_cantilever3d_bcs(problem):
    """
    Sets up BCs for 3D Cantilever Beam matching MatlabMRF/input_files/cantilever3d/COMP.
    """
    fe = problem.fe
    coords = fe.coords # (n_node, 3)
    
    if fe.dim != 3:
        raise ValueError("Problem must be 3D.")
        
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    L = np.max(x)
    H = np.max(y)
    W = np.max(z) # Z-depth
    
    tol = 1e-6
    
    # Node Sets
    # L_face: x = 0
    L_face_nodes = np.where(np.abs(x) < tol)[0]
    
    # R_face: x = L
    R_face_nodes = np.where(np.abs(x - L) < tol)[0]
    
    # K_face: Back face? Assuming z = 0
    K_face_nodes = np.where(np.abs(z) < tol)[0]
    
    # Load Region: Semi-circle on R_face centered at (L, H/2, 0)
    # MATLAB: MRK_pt. Mid-Right-Back edge?
    # If K_face is z=0, then (L, H/2, 0) is on the edge of R_face and K_face?
    # Actually (L, H/2, 0) is mid-height on the back edge of the tip.
    
    yc = H / 2.0
    zc = 0.0
    xc = L
    
    # Distance from center on the R_face
    # nodes on R_face are subset.
    # We check distance in 3D, but since x=L for all, it's distance in Y-Z plane.
    
    r = 0.5
    
    mrk_dist_sq = (x[R_face_nodes] - xc)**2 + \
                  (y[R_face_nodes] - yc)**2 + \
                  (z[R_face_nodes] - zc)**2
                  
    load_indices_local = np.where(mrk_dist_sq <= r**2 + tol)[0]
    load_nodes = R_face_nodes[load_indices_local]
    
    # Displacements
    # 1. Encastre at Left Face (x, y, z fixed)
    fixed_dofs = []
    for n in L_face_nodes:
        fixed_dofs.extend([3*n, 3*n+1, 3*n+2])
        
    # 2. Symmetry at Back Face (z fixed) -> K_face
    for n in K_face_nodes:
        # z-dof is 3*n + 2
        dof = 3*n + 2
        # Check if already added (intersection of L_face and K_face)
        # Using set for efficiency if needed, but list for now
        fixed_dofs.append(dof)
        
    # Remove duplicates and sort
    fe.fixed_dofs = np.unique(np.array(fixed_dofs, dtype=int))
    
    # Forces
    # Net magnitude -1.0 in y-direction (dim 1)
    net_mag = -1.0
    f_val = net_mag / len(load_nodes) if len(load_nodes) > 0 else 0.0
    
    if len(load_nodes) == 0:
        print("Warning: No load nodes found for 3D Cantilever!")
    
    # Global Force Vector
    # dim * n_node
    F = np.zeros(fe.n_node * 3)
    
    for n in load_nodes:
        # y direction -> 3*n + 1
        F[3*n + 1] = f_val
        
    fe.F = F
    fe.n_loads = 1


def setup_midcantilever2d_bcs(problem):
    """
    Sets up BCs for 2D Midcantilever matching TOuNN.
    """
    fe = problem.fe
    coords = fe.coords
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    L = np.max(x)
    H = np.max(y)
    tol = 1e-6
    
    # Fixed at x = 0
    fixed_region = np.where(x < tol)[0]
    n_disp_pts = len(fixed_region)
    
    bc_nodes = np.tile(fixed_region, 2)
    bc_dirs = np.concatenate([np.ones(n_disp_pts, dtype=int), 2 * np.ones(n_disp_pts, dtype=int)])
    bc_mags = np.zeros(2 * n_disp_pts)
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Load at x = L, y = H/2
    fe.nloads = 1
    load_region = np.where((x > L - tol) & (np.abs(y - H/2.0) < tol))[0]
    
    if len(load_region) > 0:
        n_force = len(load_region)
        fe.BC.n_pre_force_dofs = n_force
        fe.BC.force_node = load_region
        fe.BC.force_dof = np.full(n_force, 2, dtype=int)
        fe.BC.force_value = np.full(n_force, -1.0 / n_force)
        fe.BC.force_id = np.full(n_force, 1, dtype=int)
    else:
        fe.BC.n_pre_force_dofs = 0
        fe.BC.force_node = np.array([], dtype=int)
        fe.BC.force_dof = np.array([], dtype=int)
        fe.BC.force_value = np.array([])
        fe.BC.force_id = np.array([], dtype=int)
        print("Warning: No load nodes found for Midcantilever.")
        
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    if fe.BC.n_pre_force_dofs > 0:
        np.add.at(F, 2 * fe.BC.force_node + (fe.BC.force_dof - 1), fe.BC.force_value)
    fe.F = F
    fe.n_loads = 1

def setup_mbb2d_bcs(problem):
    """
    Sets up BCs for 2D MBB Beam (half-domain) matching TOuNN.
    """
    fe = problem.fe
    coords = fe.coords
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    L = np.max(x)
    H = np.max(y)
    tol = 1e-6
    
    # Symmetry at x = 0 (fixed x), Roller at x = L, y = 0 (fixed y)
    sym_region = np.where(x < tol)[0]
    roller_region = np.where((x > L - tol) & (y < tol))[0]
    
    n_sym = len(sym_region)
    n_roller = len(roller_region)
    
    bc_nodes = np.concatenate([sym_region, roller_region])
    bc_dirs = np.concatenate([np.ones(n_sym, dtype=int), 2 * np.ones(n_roller, dtype=int)])
    bc_mags = np.zeros(n_sym + n_roller)
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Load at x = 0, y = H
    fe.nloads = 1
    load_region = np.where((x < tol) & (y > H - tol))[0]
    
    if len(load_region) > 0:
        n_force = len(load_region)
        fe.BC.n_pre_force_dofs = n_force
        fe.BC.force_node = load_region
        fe.BC.force_dof = np.full(n_force, 2, dtype=int)
        fe.BC.force_value = np.full(n_force, -1.0 / n_force)
        fe.BC.force_id = np.full(n_force, 1, dtype=int)
    else:
        fe.BC.n_pre_force_dofs = 0
        fe.BC.force_node = np.array([], dtype=int)
        fe.BC.force_dof = np.array([], dtype=int)
        fe.BC.force_value = np.array([])
        fe.BC.force_id = np.array([], dtype=int)
        print("Warning: No load nodes found for MBB.")
        
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    if fe.BC.n_pre_force_dofs > 0:
        np.add.at(F, 2 * fe.BC.force_node + (fe.BC.force_dof - 1), fe.BC.force_value)
    fe.F = F
    fe.n_loads = 1

def setup_michell2d_bcs(problem):
    """
    Sets up BCs for 2D Michell Beam matching TOuNN.
    """
    fe = problem.fe
    coords = fe.coords
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    L = np.max(x)
    H = np.max(y)
    tol = 1e-6
    
    # Pin at x=0, y=0 (fixed x,y), Roller at x=L, y=0 (fixed y)
    pin_region = np.where((x < tol) & (y < tol))[0]
    roller_region = np.where((x > L - tol) & (y < tol))[0]
    
    n_pin = len(pin_region)
    n_roller = len(roller_region)
    
    bc_nodes = np.concatenate([pin_region, pin_region, roller_region])
    bc_dirs = np.concatenate([
        np.ones(n_pin, dtype=int), 2 * np.ones(n_pin, dtype=int),
        2 * np.ones(n_roller, dtype=int)
    ])
    bc_mags = np.zeros(len(bc_nodes))
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Load at x = L/2, y = 0
    fe.nloads = 1
    load_region = np.where((np.abs(x - L/2.0) < tol) & (y < tol))[0]
    
    if len(load_region) > 0:
        n_force = len(load_region)
        fe.BC.n_pre_force_dofs = n_force
        fe.BC.force_node = load_region
        fe.BC.force_dof = np.full(n_force, 2, dtype=int)
        fe.BC.force_value = np.full(n_force, -1.0 / n_force)
        fe.BC.force_id = np.full(n_force, 1, dtype=int)
    else:
        fe.BC.n_pre_force_dofs = 0
        fe.BC.force_node = np.array([], dtype=int)
        fe.BC.force_dof = np.array([], dtype=int)
        fe.BC.force_value = np.array([])
        fe.BC.force_id = np.array([], dtype=int)
        print("Warning: No load nodes found for Michell.")
        
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    if fe.BC.n_pre_force_dofs > 0:
        np.add.at(F, 2 * fe.BC.force_node + (fe.BC.force_dof - 1), fe.BC.force_value)
    fe.F = F
    fe.n_loads = 1

def setup_bridge2d_bcs(problem):
    """
    Sets up BCs for 2D Bridge (Distributed MBB) matching TOuNN.
    """
    fe = problem.fe
    coords = fe.coords
    from .struct_prob import BC_Struct
    fe.BC = BC_Struct()
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    L = np.max(x)
    H = np.max(y)
    tol = 1e-6
    
    # Pin at x=0, y=0 (fixed x,y), Pin at x=L, y=0 (fixed x,y)
    pin1_region = np.where((x < tol) & (y < tol))[0]
    pin2_region = np.where((x > L - tol) & (y < tol))[0]
    
    n_pin1 = len(pin1_region)
    n_pin2 = len(pin2_region)
    
    bc_nodes = np.concatenate([pin1_region, pin1_region, pin2_region, pin2_region])
    bc_dirs = np.concatenate([
        np.ones(n_pin1, dtype=int), 2 * np.ones(n_pin1, dtype=int),
        np.ones(n_pin2, dtype=int), 2 * np.ones(n_pin2, dtype=int)
    ])
    bc_mags = np.zeros(len(bc_nodes))
    
    fe.BC.n_pre_disp_dofs = len(bc_nodes)
    fe.BC.disp_node = bc_nodes
    fe.BC.disp_dof = bc_dirs
    fe.BC.disp_value = bc_mags
    
    # Distributed Load along top edge y = H
    fe.nloads = 1
    load_region = np.where(y > H - tol)[0]
    
    if len(load_region) > 0:
        n_force = len(load_region)
        fe.BC.n_pre_force_dofs = n_force
        fe.BC.force_node = load_region
        fe.BC.force_dof = np.full(n_force, 2, dtype=int)
        fe.BC.force_value = np.full(n_force, -1.0 / n_force)
        fe.BC.force_id = np.full(n_force, 1, dtype=int)
    else:
        fe.BC.n_pre_force_dofs = 0
        fe.BC.force_node = np.array([], dtype=int)
        fe.BC.force_dof = np.array([], dtype=int)
        fe.BC.force_value = np.array([])
        fe.BC.force_id = np.array([], dtype=int)
        print("Warning: No load nodes found for Bridge.")
        
    n_dof = fe.n_node * fe.dim
    F = np.zeros(n_dof)
    if fe.BC.n_pre_force_dofs > 0:
        np.add.at(F, 2 * fe.BC.force_node + (fe.BC.force_dof - 1), fe.BC.force_value)
    fe.F = F
    fe.n_loads = 1

