import jax.numpy as jnp
from .. import rhs,aux_func

def advance_one_step_flux_splitting(U,aux,metrics,dt,theta):
    U1 = U + (rhs.rhs_flux(U,aux,metrics,dt,theta)+rhs.rhs_source(U,aux,metrics,dt,theta))
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*(rhs.rhs_flux(U,aux,metrics,dt,theta)+rhs.rhs_source(U,aux,metrics,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2

def advance_one_step_godunov(U,aux,metrics,dt,theta):
    U1 = U + rhs.rhs_flux(U,aux,metrics,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*rhs.rhs_flux(U,aux,metrics,dt,theta)
    aux2 = aux_func.update_aux(U2, aux1)
    
    U, aux = U2,aux2
    
    U1 = U + rhs.rhs_source(U,aux,metrics,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*rhs.rhs_source(U,aux,metrics,dt,theta)
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2
