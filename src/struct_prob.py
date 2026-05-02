import dataclasses
from typing import List, Optional, Dict, Any, Union
import numpy as np

@dataclasses.dataclass
class MeshInput:
    type: str = 'generate'
    box_dimensions: Optional[List[float]] = None
    elements_per_side: Optional[List[int]] = None
    mesh_filename: Optional[str] = None
    gmsh_filename: Optional[str] = None
    # For L-bracket
    L_side: Optional[float] = None
    L_cutout: Optional[float] = None
    L_element_size: Optional[float] = None
    # BCs file path (in Python this might be a function or module path)
    bcs_file: Optional[str] = None

@dataclasses.dataclass
class Material:
    E: float = 1.0
    nu: float = 0.3
    rho_min: float = 1e-3
    nu_void: float = 0.3

@dataclasses.dataclass
class SolverParams:
    type: str = 'direct'
    tol: float = 1e-6
    maxit: int = int(1e4)
    use_gpu: bool = False

@dataclasses.dataclass
class FE_Struct:
    mesh_input: MeshInput = dataclasses.field(default_factory=MeshInput)
    material: Material = dataclasses.field(default_factory=Material)
    solver: SolverParams = dataclasses.field(default_factory=SolverParams)
    dim: int = 2
    n_node: int = 0
    n_elem: int = 0
    n_dof: int = 0
    nloads: int = 0
    max_elem_side: float = 0.0
    centroids: Optional[np.ndarray] = None
    elem_vol: Optional[np.ndarray] = None
    fixeddofs_ind: Optional[np.ndarray] = None
    freedofs_ind: Optional[np.ndarray] = None
    fixeddofs: Optional[np.ndarray] = None
    freedofs: Optional[np.ndarray] = None
    C_mat: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    svm: Optional[np.ndarray] = None    
    dJdu: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    Lambda: Optional[np.ndarray] = None
    Ke: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    rhs: Optional[np.ndarray] = None
    prhs: Optional[np.ndarray] = None
    B0e: Optional[np.ndarray] = None
    
    # Topology and Mesh
    coords: Optional[np.ndarray] = None
    elem_node: Optional[np.ndarray] = None
    edof_mat: Optional[np.ndarray] = None
    iK: Optional[np.ndarray] = None
    jK: Optional[np.ndarray] = None
    n_free_dof: int = 0
    
    # Verification
    make_fd_check: bool = False
    fd_step_size: float = 1e-8
    
    # ----------------------
    Kpp: Optional[np.ndarray] = None
    Kfp: Optional[np.ndarray] = None
    Kff: Optional[np.ndarray] = None
    RF: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    BC: Optional['BC_Struct'] = None

@dataclasses.dataclass
class BC_Struct:
    n_pre_force_dofs: int = 0
    force_node: Optional[np.ndarray] = None
    force_dof: Optional[np.ndarray] = None
    force_value: Optional[np.ndarray] = None
    force_id: Optional[np.ndarray] = None
    
    n_pre_disp_dofs: int = 0
    disp_node: Optional[np.ndarray] = None
    disp_dof: Optional[np.ndarray] = None
    disp_value: Optional[np.ndarray] = None

@dataclasses.dataclass
class ACSParams:
    use: bool = False
    alpha_osc: float = 0.8
    alpha_no_osc: float = 1.0
    c: List[float] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class ProjectionParams:
    use: bool = False
    type: str = 'heaviside' # 'heaviside', 'tanh'
    eta: float = 0.5
    beta_init: float = 1.0
    beta_final: float = 30.0
    current_beta: float = 1.0

@dataclasses.dataclass
class MMAParams:
    version: int = 1999
    a0: float = 1.0
    a: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None

@dataclasses.dataclass
class OptimParams:
    # Optimization Goal Configuration
    objective_type: str = 'volume' # 'volume' or 'compliance'
    constraint_types: List[str] = dataclasses.field(default_factory=lambda: ['stress']) # 'stress', 'volume'
    target_volume: float = 0.3 # Used if 'volume' is a constraint
    init_dens: float = 0.5
    
    penalization_scheme: str = 'modified_SIMP'
    penalization_param: float = 3.0
    relaxation_param: float = 0.5 # parameter for stress relaxation
    filter_type: str = 'density' # 'density', 'sensitivity'
    filter_radius_factor: float = 1.5
    
    # Aggregation
    aggregation_type: str = 'p-norm' # 'p-norm', 'mrf'
    aggregation_parameter: float = 10.0 # P or K
    
    # Continuation
    continuation: bool = False
    maxB: float = 100.0
    deltaB: float = 5.0
    maxK: float = 50.0
    deltaK: float = 2.0
    
    # MRF Specific
    rectifier_function: str = 'shiftedKS'
    rectifier_parameter: float = 10.0 # krf
    rectifier_eps: float = 1e-3
    
    # Interpolation
    interpolation_type: str = 'modified_simp' # 'simp' or 'modified_simp'
    rho_min: float = 1e-3
    
    # Adaptive Constraint Scaling
    ACS: ACSParams = dataclasses.field(default_factory=ACSParams)
    
    # Projection
    projection: ProjectionParams = dataclasses.field(default_factory=ProjectionParams)
    
    slimit: float = 1.0

    # Augmented-Lagrangian (AL) update control
    al_increase_factor: float = 2.0   # multiply mu by this factor when needed
    al_increase_tol: float = 0.95     # if violation does not reduce by this factor, increase mu
    al_max_mu: float = 150.0         # maximum allowed mu
    objective_normalize: bool = True # automatically normalize objective by initial value

    # MMA Parameters
    mma: MMAParams = dataclasses.field(default_factory=MMAParams)

@dataclasses.dataclass
class OptimOptions:
    plot: bool = True
    save_outputs: bool = True
    write_to_vtk: str = 'last'
    vtk_output_path: str = 'output_files/'
    mat_output_path: str = 'output_files/'
    outputs_path: str = 'output_files/'
    move_limit: float = 0.02
    max_GRF: float = 0.15
    max_iter: int = 500
    obj_tol: float = 1e-5

@dataclasses.dataclass
class OptimFunctions:
    objective: str = 'volume fraction'
    objective_scale: float = 1.0
    constraints: List[str] = dataclasses.field(default_factory=lambda: ['maximum stress violation'])
    constraint_limit: List[float] = dataclasses.field(default_factory=lambda: [0.0])
    constraint_scale: List[float] = dataclasses.field(default_factory=lambda: [1.0])

@dataclasses.dataclass
class NNParameters:
    hidden_dim: int = 64
    num_layers: int = 8
    activation: str = 'ReLU'
    learning_rate: float = 0.01
    vol_penal_min: float = 0.1
    vol_penal_max: float = 150.0
    use_fourier: bool = False
    fourier_scale: float = 10.0
    feature_type: str = 'fourier'
    optimizer_type: str = 'Adam'


@dataclasses.dataclass
class OPT_Struct:
    parameters: OptimParams = dataclasses.field(default_factory=OptimParams)
    options: OptimOptions = dataclasses.field(default_factory=OptimOptions)
    functions: OptimFunctions = dataclasses.field(default_factory=OptimFunctions)
    nn_params: NNParameters = dataclasses.field(default_factory=NNParameters)
    n_dv: int = 0
    dv: Optional[np.ndarray] = None # Design variables
    
    # Run-time variables
    filter_radius: float = 0.0
    H: Optional[Any] = None # Filter matrix (scipy sparse)
    
    history: Dict[str, List[Any]] = dataclasses.field(default_factory=dict)
    
    # Stress related
    stress_needed: bool = False
    write_stress_to_vtk: bool = False
    
    true_stress_max: float = 0.0
    true_h_max: float = 0.0
    approx_h_max: float = 0.0
    grf: float = 0.0
    grad_stress: Optional[np.ndarray] = None
    make_fd_check: bool = False
    fd_step_size: float = 1e-8 # Step size for FD check

class Problem:
    def __init__(self):
        self.fe = FE_Struct()
        self.opt = OPT_Struct()
        
    def run(self):
        pass
