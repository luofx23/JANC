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
    rhs.set_rhs(thermo_config, reaction_config, flux_config, transport_config, boundary_config, source_config)
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
        @jit
        def advance_one_step(U,aux,metrics,dt,theta=None):
            U, aux = time_step.time_step_dict[time_scheme](U,aux,metrics,dt,theta)
            U1 = U + reaction_model.reaction_source_terms(U,aux,dt,theta)
            aux1 = aux_func.update_aux(U1, aux)
            U2 = U + 1/2*(reaction_model.reaction_source_terms(U,aux,dt,theta)+reaction_model.reaction_source_terms(U1,aux1,dt,theta))
            aux2 = aux_func.update_aux(U2, aux1)
            return U2,aux2
    return advance_one_step
            

    





