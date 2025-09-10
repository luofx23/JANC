import jax.numpy as jnp
from .. import rhs,aux_func

def advance_one_step(U,aux,dx,dy,dt,theta):
    U1 = U + rhs.rhs(U,aux,dx,dy,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*(rhs.rhs(U,aux,dx,dy,dt,theta)+rhs.rhs(U1,aux1,dx,dy,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2
