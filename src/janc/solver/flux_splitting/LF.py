import jax.numpy as jnp
from .. import aux_func

def split_flux(ixy, U, aux, metrics):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]
    
    dξ_dx = metrics['dξ_dx']
    dη_dx = metrics['dη_dx']
    dξ_dy = metrics['dξ_dy']
    dη_dy = metrics['dη_dy']
    J = metrics['Jc']
    zx = (ixy == 1) * dξ_dx + (ixy == 2) * dη_dx
    zy = (ixy == 1) * dξ_dy + (ixy == 2) * dη_dy
    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    theta = zx*um + zy*vm
    Hplus = 0.5 * J * (F + theta * U)
    Hminus = 0.5 * J * (F - theta * U)
    
    return Hplus, Hminus

