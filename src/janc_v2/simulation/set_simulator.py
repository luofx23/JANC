import jax
from jax.sharding import Mesh, PartitionSpec as P
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
import h5py
import numpy as np
from pathlib import Path

def CFL_1D(U,aux,dx,cfl=0.30):
    _,u,_,_,a = aux_func_1D.U_to_prim(U,aux)
    sx = jnp.abs(u) + a
    return cfl*jnp.min(dx/sx)

def CFL_2D(U,aux,dx,dy,cfl=0.30):
    _,u,v,_,_,a = aux_func.U_to_prim(U,aux)
    sx = jnp.abs(u) + a
    sy = jnp.abs(v) + a
    dt = cfl*jnp.min(1/(sx/dx + sy/dy))
    return dt
    
def set_rhs(dim,reaction_config,source_config=None,is_parallel=False,is_amr=False):
    if dim == '1D':
        if is_parallel:
            boundary_conditions = parallel_boundary.boundary_conditions_1D
        else:
            boundary_conditions = boundary.boundary_conditions_1D
        def flux_func(U, aux, dx, dy, dt, theta):
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
            def advance_flux(U,aux,dx,dy,dt,theta=None):
                return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
            #if dim == '1D':
                #def advance_flux(U,aux,dx,dy,dt,theta=None):
                    #return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
            #if dim == '2D':
                #def advance_flux(U,aux,dx,dy,dt,theta=None):
                    #return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,flux_func,update_func)
    else:
        if solver_type == 'flux_splitting':
            #if dim == '1D':
                #def rhs_func(U,aux,dx,dt,theta=None):
                    #return flux_func(U,aux,dx,dt,theta) + source_func(U,aux,dt,theta)
                #def advance_flux(U,aux,dx,dt,theta=None):
                    #return time_step_dict[time_scheme](U,aux,dx,dt,theta,rhs_func,update_func)
            #if dim == '2D':
            def rhs_func(U,aux,dx,dy,dt,theta=None):
                return flux_func(U,aux,dx,dy,dt,theta) + source_func(U,aux,dt,theta)
            if is_amr:
                def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
                    return time_step_dict[time_scheme](level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta, rhs_func, update_func)
            else:
                def advance_flux(U,aux,dx,dy,dt,theta=None):
                    return time_step_dict[time_scheme](U,aux,dx,dy,dt,theta,rhs_func,update_func)
        if solver_type == 'godunov':
            #if dim == '1D':
                #def wrapped_source_func(U,aux,dx,dt,theta=None):
                    #return source_func(U,aux,dt,theta)
                #def advance_flux(U,aux,dx,dt,theta=None):
                    #U_adv,aux_adv = time_step_dict[time_scheme](U,aux,dx,dt,theta,flux_func,update_func)
                    #return time_step_dict[time_scheme](U_adv,aux_adv,dx,dt,theta,wrapped_source_func,update_func)
            #if dim == '2D':
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
        #if dim == '1D':
            #def advance_one_step(U,aux,dx,dt,theta=None):
                #U, aux = advance_flux(U,aux,dx,dt,theta)
                #dU = reaction_model.reaction_source_terms(U,aux,dt,theta)
                #U = U + dU
                #aux = update_func(U, aux)
                #return U, aux
        #if dim == '2D':
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


class H5Saver:
    def __init__(self, filepath: str | Path, data_dim: str):
        self.filepath = Path(filepath)
        # 新建文件（如果已存在会覆盖）
        self.file = h5py.File(self.filepath, "a")
        if data_dim == '1D':
            def get_prim(U,aux):
                aux = aux_func_1D.update_aux(U,aux)
                rho,u,Y,p,a = aux_func_1D.U_to_prim(U,aux)
                q = {'density':rho[0],
                     'x-velocity':u[0],
                     'pressure':p[0],
                     'tempreature':aux[1],
                     'species':Y}
                return q
            self.get_prim = get_prim
        if data_dim == '2D':
            def get_prim(U,aux):
                aux = aux_func_2D.update_aux(U,aux)
                rho,u,v,Y,p,a = aux_func_2D.U_to_prim(U,aux)
                q = {'density':rho[0],
                     'x-velocity':u[0],
                     'y-velocity':v[0],
                     'pressure':p[0],
                     'tempreature':aux[1],
                     'species':Y}
                return q
            self.get_prim = get_prim

    def save(self, save_step: int, t: float, step: int, **arrays: jnp.ndarray):
        """
        保存一个时间步的数据
        参数:
            t: 当前时间
            step: 当前步数
            arrays: 需要保存的 JAX arrays，形式为 name=array
        """
        key = f"save_step{save_step}"
        # 避免 name already exists
        if key in self.file:
            del self.file[key]
        grp = self.file.create_group(key)
        #grp = self.file.create_group(f"save_step{save_step}")
        grp.attrs["time"] = t
        grp.attrs["step"] = step
        for name, arr in arrays.items():
            arr_cpu = np.array(arr.block_until_ready())  # 拉回CPU
            grp.create_dataset(name, data=arr_cpu, compression="gzip")
        self.file.flush()  # 立即写盘，防止崩溃丢数据

    def list_snapshots(self):
        "返回所有存储的时间步名称"
        return list(self.file.keys())

    def load(self, key: str):
        "加载指定时间步的数据，返回 dict"
        if not self.file or not self.file.id.valid:
            self.file = h5py.File(self.filepath, 'r')
        grp = self.file[key]
        out = {}
        for name in grp.keys():
            out[name] = jnp.array(grp[name][:])
        U_init = jnp.array(out['u'])
        gamma_init = jnp.full_like(U_init[0:1],1.4)
        T_init = jnp.full_like(gamma_init,500.0)
        aux_init = jnp.concatenate([gamma_init,T_init],axis=0)
        prim = self.get_prim(U_init,aux_init)
        meta = dict(grp.attrs)
        time = meta['time']
        self.file.close()
        return time, prim

    def close(self):
        self.file.close()

class Simulator:
    def __init__(self,simulation_config):
        dim = simulation_config['dimension']
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
        if 'nondim_config' in simulation_config:
            nondim_config = simulation_config['nondim_config']
        else:
            nondim_config = None
        time_control = simulation_config['time_config']
        self.t_end = time_control['t_end']

        if 'solver_parameters' in simulation_config:
            theta = simulation_config['solver_parameters']
        else:
            theta = None

        if 'computation_config' in simulation_config:
            computation_config = simulation_config['computation_config']
            if 'is_parallel' in computation_config:
                is_parallel = computation_config['is_parallel']
                if is_parallel and (theta is not None):
                    assert 'PartitionDict' in computation_config, "A python dict specifying the partition axes of theta should be provided!"
                    raw_dict = computation_config['PartitionDict']
                    PartitionDict = {}
                    for key, axis in raw_dict.items():
                        n = len(jnp.shape(theta[key]))
                        if n == 0:
                            PartitionDict[key] = P()
                        else:
                            if axis == None:
                                tup = tuple(None for i in range(n))
                            else:
                                tup = tuple('x' if i == axis else None for i in range(n))
                            PartitionDict[key] = P(*tup)
                    num_devices = len(jax.devices())
                    mesh = jax.make_mesh((num_devices,))
                else:
                    PartitionDict = P()
            else:
                is_parallel = False

            if 'output_settings' in computation_config:
                output_settings = computation_config['output_settings']
                self.save_dt = output_settings['save_dt']
                self.results_path = output_settings['results_path']
            else:
                self.save_dt = self.t_end
                self.results_path = 'results.h5'
        else:
            is_parallel = False
            self.save_dt = self.t_end
            self.results_path = 'results.h5'
        self.saver = H5Saver(self.results_path,dim)
            
        thermo_model.set_thermo(thermo_config,nondim_config,dim)
        reaction_model.set_reaction(reaction_config,nondim_config,dim)
        if dim == '1D':
            flux_1D.set_flux_solver(flux_config,transport_config,nondim_config)
        if dim == '2D':
            flux.set_flux_solver(flux_config,transport_config,nondim_config)
        boundary.set_boundary(boundary_config,dim)
        is_amr = False
        flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,is_parallel,is_amr)
        advance_func_body = set_advance_func(dim,flux_config,reaction_config,time_control,is_amr,flux_func,update_func,source_func)
        if is_parallel:
            advance_func_body = jax.shard_map(advance_func_body,
                                              mesh=mesh,
                                              in_specs=(P(None,'x',None,None),P(None,'x',None,None),P(),P(),P(),PartitionDict),
                                              out_specs = (P(None,'x',None,None),P(None,'x',None,None)),
                                              check_vma=False)
            CFL_1D_body = jax.shard_map(CFL_1D,mesh=mesh,in_specs=(P(None,'x'),P(None,'x'),P(),P(None,)),out_specs=P('x',),check_vma=False)
            CFL_2D_body = jax.shard_map(CFL_2D,mesh=mesh,in_specs=(P(None,'x'),P(None,'x'),P(),P(),P(None,)),out_specs=P('x',),check_vma=False)
            def CFL_1D(U,aux,dx,cfl):
                cfl = jax.array([cfl])
                dt = CFL_1D_body(U,aux,dx,cfl)
                return jnp.min(dt)
            def CFL_2D(U,aux,dx,dy,cfl):
                cfl = jax.array([cfl])
                dt = CFL_2D_body(U,aux,dx,dy,cfl)
                return jnp.min(dt)
                
        if 'dt' in time_control:
            dt = time_control['dt']
            if dim == '1D':
                dx = grid_config['dx']
                dy = None
            if dim == '2D':
                dx, dy = grid_config['dx'], grid_config['dy']
            def advance_func(U,aux,t):
                U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
                t = t + dt
                return U,aux,t,dt

        if 'CFL' in time_control:
            CFL = time_control['CFL']
            if dim == '1D':
                dx = grid_config['dx']
                def advance_func(U,aux,t):
                    dt = CFL_1D(U,aux,dx,CFL)
                    U, aux = advance_func_body(U,aux,dx,None,dt,theta)
                    return U, aux, t+dt, dt
            if dim == '2D':
                dx, dy = grid_config['dx'],grid_config['dy']
                def advance_func(U,aux,t):
                    dt = CFL_2D(U,aux,dx,dy,CFL)
                    U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
                    return U, aux, t+dt, dt

        self.advance_func = jit(advance_func)

    def run(self,U_init,aux_init):
        advance_func = self.advance_func
        U, aux = U_init,aux_init
        t = 0.0
        step = 0
        save_step = 1
        self.saver.save(save_step=0, t=t, step=step, u=U)
        save_dt = self.save_dt
        next_save_time = save_dt
        t_end = self.t_end
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        with tqdm(total=t_end, desc="Simulation", bar_format=bar_format) as pbar:
            while t < t_end:
                U, aux, t, dt = advance_func(U,aux,t)
                step += 1

                if t >= next_save_time or t >= t_end:
                    self.saver.save(save_step, t, step, u=U)  # 保存
                    #tqdm.write(f"[SAVE] t = {t:.3e}, step = {step}")
                    next_save_time += save_dt
                    save_step += 1

                if t > t_end:
                    pbar.n = pbar.total
                    pbar.refresh()
                else:
                    pbar.update(float(dt))
                    
        self.saver.close()
        return U, aux, t

def AMR_Simulator(simulation_config):
    dim = '2D'
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
    time_control = simulation_config['time_config']
    if 'solver_parameters' in simulation_config:
            theta = simulation_config['solver_parameters']
    else:
        theta = None
    thermo_model.set_thermo(thermo_config,nondim_config,dim)
    reaction_model.set_reaction(reaction_config,nondim_config,dim)
    flux.set_flux_solver(flux_config,transport_config,nondim_config)
    boundary.set_boundary(boundary_config,dim)
    flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,False,True)
    advance_func_amr = set_advance_func(dim,flux_config,reaction_config,time_control,True,flux_func,update_func,source_func)
    flux_func, update_func, source_func = set_rhs(dim,reaction_config,source_config,False,False)
    advance_func_body = set_advance_func(dim,flux_config,reaction_config,time_control,False,flux_func,update_func,source_func)
    def advance_func_base(blk_data,dx,dy,dt,theta=None):
        U, aux = blk_data[0,:-2],blk_data[0,-2:]
        U, aux = advance_func_body(U,aux,dx,dy,dt,theta)
        blk_data = jnp.array([jnp.concatenate([U,aux],axis=0)])
        return blk_data
    return jit(advance_func_amr,static_argnames='level'),jit(advance_func_base)


























