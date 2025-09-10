from jax import vmap
import jax.numpy as jnp
from ..solver_1D import flux as flux_1D
from ..solver_1D import aux_func as aux_func_1D
from ..solver_2D import flux as flux_2D
from ..solver_2D import aux_func as aux_func_2D
from ..model import thermo_model,reaction_model,transport_model
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from functools import partial


def set_rhs(dim,thermo_config,reaction_config,flux_config,transport_config,boundary_config,source_config=None,nondim_config=None,is_parallel=False,is_amr=False):
    thermo_model.set_thermo(thermo_config,nondim_config)
    reaction_model.set_reaction(reaction_config,nondim_config,dim)
    flux.set_flux_solver(flux_config,transport_config,nondim_config)
    boundary.set_boundary(boundary_config)
    aux_func.set_source_terms(source_config)
    if dim == '1D':
        if is_parallel:
            boundary_conditions = parallel_boundary.boundary_conditions_1D
        else:
            boundary_conditions = boundary.boundary_conditions_1D
        def flux_func(U, aux, dx, dt, theta):
            U_with_ghost,aux_with_ghost = boundary_conditions(U,aux,theta)
            rhs = dt*(flux_1D.total_flux(U_with_ghost,aux_with_ghost,dx))
            return rhs
        update_func = aux_func_1D.update_aux
         
    if dim == '2D':
        if is_parallel:
            boundary_conditions = parallel_boundary.boundary_conditions_2D
        else:
            boundary_conditions = boundary.boundary_conditions_2D
        if is_amr:
            @partial(vmap,in_axes=(0,0,None,None,None,None))
            def flux_func(U, aux, dx, dy, dt, theta):
                physical_rhs = dt*(flux_2D.total_flux(U,aux,dx,dy))
                return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))
            update_func = vmap(aux_func_2D.update_aux,in_axes=(0,0))
        else:
            def flux_func(U, aux, dx, dy, dt, theta):
                U_with_ghost,aux_with_ghost = boundary_conditions(U,aux,theta)
                rhs = dt*(flux_2D.total_flux(U_with_ghost,aux_with_ghost,dx,dy))
                return rhs
            update_func = aux_func_2D.update_aux
            
    if reaction_config['is_detailed_chemistry']:
        if source_config['self_defined_source_terms'] is not None:
            source_func  = source_config['self_defined_source_terms']
        else:
            source_func = None
    else:
        if source_config['self_defined_source_terms'] is not None:
            user_source_func  = source_config['self_defined_source_terms']
            def source_func(U, aux, dt, theta):
                return user_source_func(U,aux,theta)*dt + reaction_model.reaction_source_terms(U,aux,dt,theta)
        else:
            source_func = reaction_model.reaction_source_terms

    if is_amr and (source_func is not None):
        temp_source_func = source_func
        @partial(vmap,in_axes=(0,0,None,None))
        def source_func(U, aux, dt, theta):
            return temp_source_func(U[:,3:-3,3:-3],aux[:,3:-3,3:-3],dt,theta)
    return flux_func, update_func, source_func

def set_advance_func(flux_config,reaction_config,flux_func,update_func,source_func):
    is_detailed_chemistry = reaction_config['is_detailed_chemistry']
    solver_type = flux_config['solver_type']
    if source_func is None:
        
        
            
    


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
        if is_amr:
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info,theta=None):
                blk_data_adv = time_step.time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info,theta)
                U,aux = blk_data_adv[:,0:-2],blk_data_adv[:,-2:]
                dU = vmap(reaction_model.reaction_source_terms,in_axes=(0, 0, None, None))(U,aux,dt,theta)
                U = U + dU
                aux = vmap(aux_func.update_aux,in_axes=(0,0))(U, aux)
                return jnp.concatenate([U,aux],axis=1)
        else:
            def advance_one_step(U,aux,dx,dy,dt,theta=None):
                U, aux = time_step.time_step_dict[time_scheme](U,aux,dx,dy,dt,theta)
                dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
                U = U + dU
                aux = aux_func.update_aux(U, aux)
                return U, aux
    else:
        advance_one_step = time_step.time_step_dict[time_scheme]
    return advance_one_step
            

    











