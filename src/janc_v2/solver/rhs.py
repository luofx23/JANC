from jax import vmap
import jax.numpy as jnp
from . import flux, aux_func
from ..model import thermo_model,reaction_model,transport_model
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from functools import partial

point_implicit = 'off'
parallel = 'off'
def set_rhs(thermo_config,reaction_config,flux_config,transport_config,boundary_config,source_config=None,nondim_config=None,is_parallel=False):
    global point_implicit,parallel
    dim = '2D'
    thermo_model.set_thermo(thermo_config,nondim_config)
    reaction_model.set_reaction(reaction_config,nondim_config,dim)
    flux.set_flux_solver(flux_config,transport_config,nondim_config)
    boundary.set_boundary(boundary_config)
    aux_func.set_source_terms(source_config)
    if reaction_config['is_detailed_chemistry']:
        point_implicit = 'on'
    if is_parallel:
        parallel = 'on'

def rhs_source_explicit(U, aux, dt, theta):
    return aux_func.user_source(U,aux,theta)*dt + reaction_model.reaction_source_terms(U,aux,dt,theta)

def rhs_source_implicit(U, aux, dt, theta):
    return aux_func.user_source(U,aux,theta)*dt

boundary_conditions_dict = {'on':parallel_boundary.boundary_conditions,
                            'off':boundary.boundary_conditions}

def rhs_flux(U, aux, dx, dy, dt, theta):
    U_with_ghost,aux_with_ghost = boundary_conditions_dict[parallel](U,aux,theta)
    rhs = dt*(flux.total_flux(U_with_ghost,aux_with_ghost,dx,dy))
    return rhs

@partial(vmap,in_axes=(0,0,None,None,None,None))
def rhs_flux_amr(U,aux,dx,dy,dt,theta):
    physical_rhs = dt*(flux.total_flux(U,aux,dx,dy))
    return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))

rhs_source_dict = {'off':rhs_source_explicit,
                   'on':rhs_source_implicit}

def rhs_source(U, aux, dt, theta):
    return rhs_source_dict[point_implicit](U, aux, dt, theta)

@partial(vmap,in_axes=(0,0,None,None))
def rhs_source_amr(U, aux, dt, theta):
    return rhs_source_dict[point_implicit](U[:,3:-3,3:-3], aux[:,3:-3,3:-3], dt, theta)
    








