import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
def compute_filter_matrix(centroids: np.ndarray, radius: float, max_elem_side: float = 0.0) -> sparse.csr_matrix:
    """
    Computes the filter matrix H matching MATLAB logic.
    centroids: (dim, n_elem)
    radius: filter radius
    max_elem_side: maximum element side length for tolerance calculation
    """
    # KDTree expects (n_points, dim)
    points = centroids.T
    n_elem = points.shape[0]
    tree = cKDTree(points)
    
    # MATLAB uses tol = FE.max_elem_side/1000 and search_dist = radius - tol
    if max_elem_side > 0:
        tol = max_elem_side / 1000.0
    else:
        tol = 1e-10
        
    search_dist = radius - tol
    
    neighbors_list = tree.query_ball_point(points, r=search_dist)
    
    # Construct sparse matrix
    rows = []
    cols = []
    data = []
    
    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) == 0:
            continue
            
        pts_neigh = points[neighbors]
        diff = pts_neigh - points[i]
        dists = np.linalg.norm(diff, axis=1)
        
        # Weight function: 1 - dist/radius
        weights = 1.0 - dists / radius
        
        # Normalize: den = sum(num)
        den = np.sum(weights)
        if den > 1e-16:
            weights = weights / den
        else:
            # Fallback if somehow den is zero (shouldn't happen with self-neighbor)
            weights = np.zeros_like(weights)
            # Find self-index in neighbors and set to 1
            for idx, n_idx in enumerate(neighbors):
                if n_idx == i:
                    weights[idx] = 1.0
                    break
            
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
        data.extend(weights)
        
    H = sparse.csr_matrix((data, (rows, cols)), shape=(n_elem, n_elem))
    return H

def plot_density(coords: np.ndarray, elem_node: np.ndarray, x: np.ndarray,
                 filename: str = 'density.png', show: bool = False,
                 title: str = 'Density Distribution',
                 H=None, beta=None, eta=None, penal=None, rho_min=1e-3):
    # ---- Density processing ----
    x_plot = x.copy()

    if H is not None:
        x_plot = H.dot(x_plot)

    if beta is not None and eta is not None:
        x_plot, _ = compute_heaviside_projection(x_plot, beta, eta)

    if penal is not None:
        x_plot, _ = compute_modified_simp(x_plot, penal, rho_min)

    if show and not plt.isinteractive():
        plt.ion()

    fig = plt.figure(1)
    fig.set_size_inches(8, 6)
    plt.clf()
    ax = fig.add_subplot(111)

    # ---- Density plot ----
    verts = coords[elem_node]
    pc = PolyCollection(
        verts,
        array=x_plot,
        cmap='gray_r',
        edgecolors='none'
    )
    pc.set_clim(0, 1)

    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')

    cbar = plt.colorbar(pc, ticks=[0, 1])
    cbar.ax.set_title(r'$\rho$', fontsize=14)

    # ---- Boundary edges ----
    boundary_edges = extract_boundary_edges(elem_node)
    boundary_lines = coords[boundary_edges]

    lc = LineCollection(
        boundary_lines,
        colors='k',
        linewidths=1.5
    )
    ax.add_collection(lc)

    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if show:
        plt.draw()
        plt.pause(0.1)
    else:
        if not filename.lower().endswith('.png'):
            filename += '.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_stress(coords: np.ndarray, elem_node: np.ndarray, stress: np.ndarray, filename: str = 'stress.png', show: bool = False, title: str = None):
    """
    Plots the element Von Mises stress.
    stress: (n_elem,)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    # Interactive mode logic for show=True
    if show:
        if not plt.isinteractive():
             plt.ion()
    
    fig_num = 102
    fig = plt.figure(fig_num)
    fig.set_size_inches(8, 6) # Optional: Ensure consistent size
    plt.clf()
    ax = fig.add_subplot(111)
    
    if elem_node.shape[1] == 4:
        verts = coords[elem_node]
        # Use a stress-friendly colormap, e.g., 'jet' or 'viridis'
        pc = PolyCollection(verts, array=stress, cmap='jet', edgecolors='none')
        
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')
        cbar = plt.colorbar(pc) # Removed ticks=[0,1] for stress as it is not 0-1
        cbar.ax.set_title(r'$\sigma_{VM}$', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        if title is None:
            title_str = fr'Von Mises Stress; $\max_e(\sigma_{{VM}}) = {np.max(stress):.2f}$'
        else:
             title_str = f"{title} (Max: {np.max(stress):.2f})"
             
        plt.title(title_str, fontsize=14)
        
        if show:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.savefig(filename)
            plt.close(fig)

def plot_history(history: dict, filename: str = 'history.png', show: bool = False):
    """
    Plots optimization history: Objective and Constraint on the same plot
    using two vertical axes.
    
    history: dict containing lists 'fval', 'fconsval', etc.
    """
    if show:
        if not plt.isinteractive():
            plt.ion()

    fig = plt.figure(103) # Use specific number for history too
    fig.set_size_inches(10, 6)
    plt.clf()

    # Main axis: Objective
    ax1 = fig.add_subplot(111)

    if 'fval' in history:
        ax1.plot(history['fval'], 'b-', label='Objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid(True)

    # Secondary axis: Constraint
    ax2 = ax1.twinx()

    if 'fconsval' in history:
        cons_hist = np.array(history['fconsval'])

        if cons_hist.ndim == 1:
            max_viol = cons_hist
        else:
            max_viol = np.max(cons_hist, axis=1)

        ax2.plot(max_viol, 'r--', label='Max Constraint Violation')
        ax2.set_ylabel('Constraint Violation', color='r')
    
    if show:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.savefig(filename)
        plt.close(fig)
def compute_heaviside_projection(x: np.ndarray, beta: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Heaviside projection based on hyperbolic tangent.
    Returns:
        x_proj: Projected density.
        d_proj: Derivative of projected density w.r.t input density (d_proj/d_x).
    """
    # tanh(beta*eta) + tanh(beta*(x-eta))
    # ------------------------------------
    # tanh(beta*eta) + tanh(beta*(1-eta))
    
    # Use numpy for vectorization
    tanh_beta_eta = np.tanh(beta * eta)
    tanh_beta_1_eta = np.tanh(beta * (1.0 - eta))
    
    denom = tanh_beta_eta + tanh_beta_1_eta
    
    # Avoid division by zero if beta is 0 (unlikely) or denom is 0
    if abs(denom) < 1e-16:
        denom = 1e-16
        
    num = tanh_beta_eta + np.tanh(beta * (x - eta))
    x_proj = num / denom
    
    # Derivative:
    # d(num)/dx = beta * sech^2(beta*(x-eta)) = beta * (1 - tanh^2)
    t = np.tanh(beta * (x - eta))
    d_num = beta * (1.0 - t**2)
    
    d_proj = d_num / denom
    
    return x_proj, d_proj

def compute_simp(x: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SIMP penalization (x^p).
    Returns:
        x_simp: Penalized density (x^p).
        d_simp: Derivative (p * x^(p-1)).
    """
    # Handle x=0 case for derivative if p < 1 (singularity) 
    # But usually p >= 1 (e.g. 3).
    # If p >= 1, 0^(p-1) is 0 (if p>1) or 1 (if p=1).
    
    x_simp = x ** p
    d_simp = p * (x ** (p - 1.0))
    
    return x_simp, d_simp

def compute_modified_simp(x: np.ndarray, p: float, rho_min: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Modified SIMP interpolation: E(x) = E_min + x^p (E0 - E_min).
    K(x) = (rho_min + (1 - rho_min) * x^p) * K0.
    
    Returns:
        x_mod: Interpolated density factor (rho_min + (1-rho_min)x^p).
        d_mod: Derivative w.r.t x.
    """
    
    term_p = x ** p
    x_mod = rho_min + (1.0 - rho_min) * term_p
    d_mod = (1.0 - rho_min) * p * (x ** (p - 1.0))
    
    return x_mod, d_mod

def extract_boundary_edges(elem_node):
    """
    Returns array of boundary edges (n_edges, 2)
    """
    import numpy as np
    from collections import Counter

    edges = []

    for e in elem_node:
        edges.extend([
            tuple(sorted((e[0], e[1]))),
            tuple(sorted((e[1], e[2]))),
            tuple(sorted((e[2], e[3]))),
            tuple(sorted((e[3], e[0])))
        ])

    edge_count = Counter(edges)
    boundary_edges = [edge for edge, c in edge_count.items() if c == 1]

    return np.array(boundary_edges)
