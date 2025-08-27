from jax import jit
import jax.numpy as jnp
from ..solver import rhs,time_step,aux_func
from ..model import reaction_model

def set_simulation(simulation_config):
    thermo_config = simulation_config['thermo_config']
    reaction_config = simulation_config['reaction_config']
    reaction_source_terms = reaction_model.set_reaction(reaction_config)
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
    rhs_func = rhs.set_rhs(thermo_config, reaction_config, flux_config, transport_config, boundary_config, source_config)
    time_scheme = simulation_config['temporal_evolution_scheme']
    time_step_func = time_step.time_step_dict[time_scheme]
    if reaction_config['is_detailed_chemistry']:
        @jit
        def advance_one_step(U,aux,metrics,dt,theta=None):
            U1 = U + rhs_func(U,aux,metrics,dt,theta)
            aux1 = aux_func.update_aux(U1, aux)
            U2 = 3/4*U + 1/4 * (U1 + rhs_func(U1,aux1,metrics,dt,theta))
            aux2 = aux_func.update_aux(U2, aux1)
            U3 = 1/3*U + 2/3 * (U2 + rhs_func(U2,aux2,metrics,dt,theta))
            aux3 = aux_func.update_aux(U3, aux2)
            dU = reaction_source_terms(U3,aux3,dt,theta)
            U = U3 + dU
            aux = aux_func.update_aux(U, aux3)
            return U, aux
    else:
        @jit
        def advance_one_step(U,aux,metrics,dt,theta=None):
            U, aux = time_step_func(U,aux,metrics,dt,theta,rhs_func)
            return U, aux
    return advance_one_step
            

    


