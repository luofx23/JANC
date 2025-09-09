import jax.numpy as jnp
from .. import rhs,aux_func
#from ...model import reaction_model

def advance_one_step(U,aux,metrics,dt,theta):
    U1 = U + rhs.rhs(U,aux,metrics,dt,theta)
    aux1 = aux_func.update_aux(U1, aux)
    U2 = U + 1/2*(rhs.rhs(U,aux,metrics,dt,theta)+rhs.rhs(U1,aux1,metrics,dt,theta))
    aux2 = aux_func.update_aux(U2, aux1)
    return U2,aux2
