import jax.numpy as jnp
from .. import rhs,aux_func

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
