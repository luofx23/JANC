import jax.numpy as jnp
from .. import rhs,aux_func

def advance_one_step_flux_splitting(U,aux,dx,dy,dt,theta):
    U1 = U + rhs.flux(U,aux,dx,dy,dt,theta)+rhs.source(U,aux,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = 3/4*U + 1/4 * (U1 + rhs.flux(U1,aux1,dx,dy,dt,theta)+rhs.source(U1,aux1,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    U3 = 1/3*U + 2/3 * (U2 + rhs.flux(U2,aux2,dx,dy,dt,theta)+rhs.source(U2,aux2,dt,theta))
    aux3 = aux_func.update_aux(U3, aux2)
    return U3,aux3

def advance_one_step_godunov(U,aux,dx,dy,dt,theta):
    U1 = U + rhs.flux(U,aux,dx,dy,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = 3/4*U + 1/4 * (U1 + rhs.flux(U1,aux1,dx,dy,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    U3 = 1/3*U + 2/3 * (U2 + rhs.flux(U2,aux2,dx,dy,dt,theta))
    aux3 = aux_func.update_aux(U3, aux2)

    U,aux = U3,aux3

    U1 = U + rhs.source(U,aux,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = 3/4*U + 1/4 * (U1 + rhs.source(U1,aux1,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    U3 = 1/3*U + 2/3 * (U2 + rhs.source(U2,aux2,dt,theta))
    aux3 = aux_func.update_aux(U3, aux2)
    return U3,aux3

