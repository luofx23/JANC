import jax.numpy as jnp
from .. import aux_func

def split_flux(U, aux, metrics):
    rho,u,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[2:3]
    
    dξ_dx = metrics['dξ_dx']
    J = metrics['Jc']
    zx = dξ_dx
    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, u * (rhoE + p), rho * u * Y], axis=0)
    um = zx*jnp.nanmax(abs(u) + a)
    Hplus = 0.5 * J * (F + um * U)
    Hminus = 0.5 * J * (F - um * U)
    
    return Hplus, Hminus

