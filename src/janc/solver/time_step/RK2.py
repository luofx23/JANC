import jax.numpy as jnp
from .. import aux_func

def advance_one_step(U,aux,metrics,dt,theta,rhs):
    U1 = U + rhs(U,aux,metrics,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*(rhs(U,aux,metrics,dt,theta)+rhs(U1,aux1,metrics,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2