from jax import jit
import jax.numpy as jnp
from ..solver_1D import aux_func as aux_func_1D
from ..solver_2D import rhs as rhs_2D
from ..solver_2D import time_step as time_step_2D
from ..solver_2D import aux_func as aux_func_2D
from ..model import reaction_model

def set_simulation(simulation_config):
    dim = simulation_config['dimension']
    thermo_config = simulation_config['thermo_config']
    reaction_config = simulation_config['reaction_config']
    if 'transport_config' in simulation_config:
        transport_config = simulation_config['transport_config']
    else:
        transport_config = None
    flux_config = simulation_config['flux_config']
    boundary_config = simulation_config['boundary_config']
    if 'source_config' in simulation_config:
        source_config = simulation_config['source_config']
    else:
        source_config = None
    if dim == '1D':
        rhs = rhs_1D
        time_step = time_step_1D
        aux_func = aux_func_1D
    if dim == '2D':
        rhs = rhs_2D
        time_step = time_step_2D
        aux_func = aux_func_2D
    rhs.set_rhs(thermo_config, reaction_config, flux_config, transport_config, boundary_config, source_config)
    time_scheme = simulation_config['temporal_evolution_scheme']
    if reaction_config['is_detailed_chemistry']:
        @jit
        def advance_one_step(U,aux,dx,dy,dt,theta=None):
            U, aux = time_step.time_step_dict[time_scheme](U,aux,dx,dy,dt,theta)
            dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
            U = U + dU
            aux = aux_func.update_aux(U, aux)
            return U, aux
    else:
        #advance_one_step = jit(time_step.time_step_dict[time_scheme])
        @jit
        def advance_one_step(U,aux,dx,dy,dt,theta=None):
            U, aux = time_step.time_step_dict[time_scheme](U,aux,dx,dy,dt,theta)
            #U1 = U + reaction_model.reaction_source_terms(U,aux,dt,theta)
            #aux1 = aux_func.update_aux(U1, aux)
            #U2 = U + 1/2*(reaction_model.reaction_source_terms(U,aux,dt,theta)+reaction_model.reaction_source_terms(U1,aux1,dt,theta))
            #aux2 = aux_func.update_aux(U2, aux1)
            return U,aux
    return advance_one_step
            

    








