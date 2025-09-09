import jax.numpy as jnp
from . import flux, aux_func
from ..grid import read_grid
from ..model import thermo_model,reaction_model,transport_model
from ..boundary import boundary

point_implicit = 'off'
def set_rhs(thermo_config,reaction_config,flux_config,transport_config,boundary_config,source_config=None):
    global point_implicit
    dim = '1D'
    thermo_model.set_thermo(thermo_config)
    reaction_model.set_reaction(reaction_config,dim)
    flux.set_flux_solver(flux_config,transport_config)
    boundary.set_boundary(boundary_config,dim)
    aux_func.set_source_terms(source_config)
    if reaction_config['is_detailed_chemistry']:
        point_implicit = 'on'

def rhs(U, aux, metrics, dt, theta):
    U_with_ghost,aux_with_ghost = boundary.boundary_conditions_1D(U,aux,metrics,theta)
    #rhs = dt*(flux.total_flux(U_with_ghost,aux_with_ghost,metrics)
    return dt*(flux.total_flux(U_with_ghost,aux_with_ghost,metrics))
    

















