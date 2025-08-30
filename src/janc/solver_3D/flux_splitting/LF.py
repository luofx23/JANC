import jax.numpy as jnp
from .. import aux_func

def split_flux(ixy, U, aux, metrics):
    rho,u,v,w,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[4:5,:,:]
    
    dξ_dx = metrics['dξ_dx']
    dη_dx = metrics['dη_dx']
    dζ_dx = metrics['dζ_dx']
    dξ_dy = metrics['dξ_dy']
    dη_dy = metrics['dη_dy']
    dζ_dy = metrics['dζ_dy']
    dξ_dz = metrics['dξ_dz']
    dη_dz = metrics['dη_dz']
    dζ_dz = metrics['dζ_dz']
    J = metrics['Jc']
    zx = (ixy == 1) * dξ_dx + (ixy == 2) * dη_dx + (ixy == 3) * dζ_dx
    zy = (ixy == 1) * dξ_dy + (ixy == 2) * dη_dy + (ixy == 3) * dζ_dy
    zz = (ixy == 1) * dξ_dz + (ixy == 2) * dη_dz + (ixy == 3) * dζ_dz
    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, rho * u * w, u * (rhoE + p), rho * u * Y], axis=0) \
        + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, rho * v * w, v * (rhoE + p), rho * v * Y], axis=0) \
        + zz*jnp.concatenate([rho * w, rho * u * w, rho * v * w, rho * w ** 2 + p, w * (rhoE + p), rho * w * Y], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    zm = jnp.nanmax(abs(w) + a)
    theta = zx*um + zy*vm + zz*zm
    Hplus = 0.5 * J * (F + theta * U)
    Hminus = 0.5 * J * (F - theta * U)
    
    return Hplus, Hminus

