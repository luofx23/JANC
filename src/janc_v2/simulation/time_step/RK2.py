from jax import vmap
import jax.numpy as jnp
from .. import rhs,aux_func
from jaxamr import amr

def advance_one_step_flux_splitting(U,aux,dx,dy,dt,theta):
    dev = rhs.rhs_flux(U,aux,dx,dy,dt,theta)+rhs.rhs_source(U,aux,dt,theta)
    U1 = U + dev
    aux1 = aux_func.update_aux(U1, aux)
    dev1 = rhs.rhs_flux(U1,aux1,dx,dy,dt,theta)+rhs.rhs_source(U1,aux1,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2

def advance_one_step_godunov(U,aux,dx,dy,dt,theta):
    dev = rhs.rhs_flux(U,aux,dx,dy,dt,theta)
    U1 = U + dev
    aux1 = aux_func.update_aux(U1, aux)
    dev1 = rhs.rhs_flux(U1,aux1,dx,dy,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = aux_func.update_aux(U2, aux1)
    
    U, aux = U2,aux2
    
    dev = rhs.rhs_source(U,aux,dt,theta)
    U1 = U + dev
    aux1 = aux_func.update_aux(U1, aux)
    dev1 = rhs.rhs_source(U1,aux1,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2

def advance_one_step_flux_splitting_amr(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
    ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
    U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
    dev = rhs.rhs_flux_amr(U,aux,dx,dy,dt,theta)+rhs.rhs_source_amr(U,aux,dt,theta)
    U1 = U + dev
    aux1 = vmap(aux_func.update_aux,in_axes=(0,0))(U1, aux)
    blk_data1 = jnp.concatenate([U1,aux1],axis=1)
    blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., 3:-3, 3:-3], ref_blk_info)
    ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
    U1, aux1 = ghost_blk_data1[:,0:-2],ghost_blk_data1[:,-2:]
    dev1 = rhs.rhs_flux_amr(U1,aux1,dx,dy,dt,theta)+rhs.rhs_source_amr(U1,aux1,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = vmap(aux_func.update_aux,in_axes=(0,0))(U2, aux1)
    blk_data2 = jnp.concatenate([U2,aux2],axis=1)
    blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., 3:-3, 3:-3], ref_blk_info)
    return blk_data2

def advance_one_step_godunov_amr(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):
    ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
    U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
    dev = rhs.rhs_flux_amr(U,aux,dx,dy,dt,theta)
    U1 = U + dev
    aux1 = vmap(aux_func.update_aux,in_axes=(0,0))(U1, aux)
    blk_data1 = jnp.concatenate([U1,aux1],axis=1)
    blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., 3:-3, 3:-3], ref_blk_info)
    ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
    U1, aux1 = ghost_blk_data1[:,0:-2],ghost_blk_data1[:,-2:]
    dev1 = rhs.rhs_flux_amr(U1,aux1,dx,dy,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = vmap(aux_func.update_aux,in_axes=(0,0))(U2, aux1)
    blk_data2 = jnp.concatenate([U2,aux2],axis=1)
    blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., 3:-3, 3:-3], ref_blk_info)
    
    ref_blk_data = blk_data2
    
    ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
    U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
    dev = rhs.rhs_source_amr(U,aux,dt,theta)
    U1 = U + dev
    aux1 = vmap(aux_func.update_aux,in_axes=(0,0))(U1, aux)
    blk_data1 = jnp.concatenate([U1,aux1],axis=1)
    blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., 3:-3, 3:-3], ref_blk_info)
    ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
    U1, aux1 = ghost_blk_data1[:,0:-2],ghost_blk_data1[:,-2:]
    dev1 = rhs.rhs_source_amr(U1,aux1,dt,theta)
    U2 = U + 1/2*(dev + dev1)
    aux2 = vmap(aux_func.update_aux,in_axes=(0,0))(U2, aux1)
    blk_data2 = jnp.concatenate([U2,aux2],axis=1)
    blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., 3:-3, 3:-3], ref_blk_info)
    return blk_data2
