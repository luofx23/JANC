from jax import jit
import jax.numpy as jnp
from ..solver_2D import rhs
from ..solver_2D import time_step
from ..solver_2D import aux_func

from ..model import reaction_model

def set_simulation(simulation_config):
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
    if 'nondim_config' in simulation_config:
        nondim_config = simulation_config['nondim_config']
    else:
        nondim_config = None
    if 'computation_config' in simulation_config:
        computation_config = simulation_config['computation_config']
        is_parallel = computation_config['is_parallel']
        is_amr = computation_config['is_amr']
        if is_parallel and is_amr:
            raise RuntimeError('The parallel version of AMR is currently unavaliable.')
    else:
        is_parallel = False
        is_amr = False
    rhs.set_rhs(thermo_config, reaction_config, flux_config, transport_config, boundary_config, source_config, nondim_config,is_parallel=is_parallel)
    time_scheme = simulation_config['temporal_evolution_scheme'] + '_' + flux_config['solver_type']
    if is_amr:
        time_scheme += '_amr'
    if reaction_config['is_detailed_chemistry']:
        @jit
        def advance_one_step(U,aux,dx,dy,dt,theta=None):
            U, aux = time_step.time_step_dict[time_scheme](U,aux,dx,dy,dt,theta)
            dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
            U = U + dU
            aux = aux_func.update_aux(U, aux)
            return U, aux
    else:
        advance_one_step = jit(time_step.time_step_dict[time_scheme])
    return advance_one_step
            

    







