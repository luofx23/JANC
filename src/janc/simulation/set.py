from jax import jit
import jax.numpy as jnp
from ..solver import rhs,time_step,aux_func
from ..model import reaction_model

def set_simulation(simulation_config):
    grid_config = simulation_config['grid_config']
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
    rhs.set_rhs(grid_config, thermo_config, reaction_config, flux_config, transport_config, boundary_config, source_config)
    time_scheme = simulation_config['temporal_evolution_scheme']
    if reaction_config['is_detailed_chemistry']:
        @jit
        def advance_one_step(U,aux,metrics,dt,theta=None):
            U, aux = time_step.time_step_dict[time_scheme](U,aux,metrics,dt,theta)
            dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
            U = U + dU
            aux = aux_func.update_aux(U, aux)
            return U, aux
    else:
        advance_one_step = jit(time_step.time_step_dict[time_scheme])
    return advance_one_step
            
    