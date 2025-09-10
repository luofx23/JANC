from jax import vmap,jit
import jax.numpy as jnp
from .time_step import time_step_dict
from ..solver_1D import flux as flux_1D
from ..solver_1D import aux_func as aux_func_1D
from ..solver_2D import flux #as flux
from ..solver_2D import aux_func #as aux_func
from ..model import thermo_model,reaction_model,transport_model
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from functools import partial
from tqdm import tqdm


def set_rhs(dim,reaction_config,source_config=None,is_parallel=False,is_amr=False):
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
                physical_rhs = dt*(flux.total_flux(U,aux,dx,dy))
                return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))
            update_func = vmap(aux_func.update_aux,in_axes=(0,0))
        else:
            def flux_func(U, aux, dx, dy, dt, theta):
                U_with_ghost,aux_with_ghost = boundary_conditions(U,aux,theta)
                rhs = dt*(flux.total_flux(U_with_ghost,aux_with_ghost,dx,dy))
                return rhs
            update_func = aux_func.update_aux
            
    if reaction_config['is_detailed_chemistry']:
        if source_config is not None:
            source_func  = source_config['self_defined_source_terms']
        else:
            source_func = None
    else:
        if source_config is not None:
            user_source_func  = source_config['self_defined_source_terms']
            def source_func(U, aux, dt, theta):
                return user_source_func(U,aux,theta)*dt + reaction_model.reaction_source_terms(U,aux,dt,theta)
        else:
            if ('self_defined_reaction_source_terms' not in reaction_config) or (reaction_config['self_defined_reaction_source_terms'] is None):
                source_func = None
            else:
                source_func = reaction_model.reaction_source_terms

    if is_amr and (source_func is not None):
        temp_source_func = source_func
        @partial(vmap,in_axes=(0,0,None,None))
        def source_func(U, aux, dt, theta):
            return temp_source_func(U[:,3:-3,3:-3],aux[:,3:-3,3:-3],dt,theta)
    return flux_func, update_func, source_func

def set_advance_func(dim,flux_config,reaction_config,time_control,is_amr,flux_func,update_func,source_func):
    is_detailed_chemistry = reaction_config['is_detailed_chemistry']
    solver_type = flux_config['solver_type']
    time_scheme = time_control['temporal_evolution_scheme'] + (is_amr)*('_amr')

    if source_func is None:
        if is_amr:
            def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, flux_func, update_func)
        else:
            if dim == '1D':
                def advance_flux(U,aux,dx,dt,theta=None):
                    return time_step_dict[time_scheme](U,aux,dx,dt,theta,flux_func,update_func)
            if dim == '2D':
                def advance_flux(U,aux,dx,dy,dt,theta=None):
                    return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
    else:
        if solver_type == 'flux_splitting':
            if dim == '1D':
                def rhs_func(U,aux,dx,dt,theta=None):
                    return flux_func(U,aux,dx,dt,theta) + source_func(U,aux,dt,theta)
                def advance_flux(U,aux,dx,dt,theta=None):
                    return time_step_dict[time_scheme](U,aux,dx,dt,theta,rhs_func,update_func)
            if dim == '2D':
                def rhs_func(U,aux,dx,dy,dt,theta=None):
                    return flux_func(U,aux,dx,dy,dt,theta) + source_func(U,aux,dt,theta)
                if is_amr:
                    def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                        return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, rhs_func, update_func)
                else:
                    def advance_flux(U,aux,dx,dy,dt,theta=None):
                        return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,rhs_func,update_func)
        if solver_type == 'godunov':
            if dim == '1D':
                def wrapped_source_func(U,aux,dx,dt,theta=None):
                    return source_func(U,aux,dt,theta)
                def advance_flux(U,aux,dx,dt,theta=None):
                    U_adv,aux_adv = time_step_dict[time_scheme](U,aux,dx,dt,theta,flux_func,update_func)
                    return time_step_dict[time_scheme](U_adv,aux_adv,dx,dt,theta,wrapped_source_func,update_func)
            if dim == '2D':
                def wrapped_source_func(U,aux,dx,dy,dt,theta=None):
                    return source_func(U,aux,dt,theta)
                if is_amr:
                    def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                        field_adv = time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, flux_func, update_func)
                        return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, field_adv, ref_blk_info, theta, wrapped_source_func, update_func)
                else:
                    def advance_flux(U,aux,dx,dy,dt,theta=None):
                        U_adv,aux_adv = time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
                        return time_step_dict[time_scheme](U_adv,aux_adv,dx,dy,dt,theta,wrapped_source_func,update_func)

    if is_detailed_chemistry:
        if dim == '1D':
            def advance_one_step(U,aux,dx,dt,theta=None):
                U, aux = advance_flux(U,aux,dx,dt,theta)
                dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
                U = U + dU
                aux = update_func(U, aux)
                return U, aux
        if dim == '2D':
            if is_amr:
                def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                    blk_data = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None)
                    U, aux = blk_data[:,0:-2],blk_data[:,-2:]
                    dU = vmap(reaction_model.reaction_source_terms,in_axes=(0,0,None,None))(U,aux,dt,theta)
                    U = U + dU
                    aux = update_func(U, aux)
                    return jnp.concatenate([U,aux],axis=1)
            else:
                def advance_one_step(U,aux,dx,dy,dt,theta=None):
                    U, aux = advance_flux(U,aux,dx,dy,dt,theta)
                    dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
                    U = U + dU
                    aux = update_func(U, aux)
                    return U, aux
    else:
        advance_one_step = advance_flux
    return advance_one_step

class Simulator:
    def __init__(self,simulation_config):
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
        if 'nondim_config' in simulation_config:
            nondim_config = simulation_config['nondim_config']
        else:
            nondim_config = None
        time_control = simulation_config['time_control']
        if 'computation_config' in simulation_config:
            computation_config = simulation_config['computation_config']
            is_parallel = computation_config['is_parallel']
            is_amr = computation_config['is_amr']
            if is_parallel and is_amr:
                raise RuntimeError('The parallel version of AMR is currently unavaliable.')
        else:
            is_parallel = False
            is_amr = False
        thermo_model.set_thermo(thermo_config,nondim_config)
        reaction_model.set_reaction(reaction_config,nondim_config,dim)
        if dim == '1D':
            flux_1D.set_flux_solver(flux_config,transport_config,nondim_config)
        if dim == '2D':
            flux.set_flux_solver(flux_config,transport_config,nondim_config)
            print(flux_config['viscosity'],flux.viscosity)
            print(flux.viscosity)
        boundary.set_boundary(boundary_config,dim)
        flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,is_parallel,is_amr)
        advance_func = set_advance_func(dim,flux_config,reaction_config,time_control,is_amr,flux_func,update_func,source_func)
        if is_amr:
            self.advance_func = jit(advance_func,static_argnames='level')
        else:
            self.advance_func = jit(advance_func)

    def run(self,nt,U_init,aux_init,dx,dy,dt,theta):
        advance_func = self.advance_func
        U,aux = U_init,aux_init
        for step in tqdm(range(nt),desc="progress", unit="step"):
              U, aux = advance_func(U,aux,dx,dy,dt,theta)
        return U, aux

    def get_step_func(self):
        return self.advance_func

    

























