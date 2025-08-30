import jax.numpy as jnp
from .. import aux_func

def split_flux(ixy, U, aux, metrics):
    rho,u,v,w,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[4:5,:,:]
    gamma = aux[0:1]
    
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
    theta = zx * u + zy * v + zz * w

    H1 = J*(1 / (2 * gamma)) * jnp.concatenate([rho, rho * u - rho * a * zx, rho * v - rho * a * zy,
                          rho * w  - rho * a * zz, rhoE + p - rho * a * theta, rho * Y], axis=0)
    H2 = J*((gamma - 1) / gamma) * jnp.concatenate(
        [rho, rho * u, rho * v, rho * w, 0.5 * rho * (u ** 2 + v ** 2 + w ** 2), rho * Y], axis=0)
    H4 = J*(1 / (2 * gamma)) * jnp.concatenate([rho, rho * u + rho * a * zx, rho * v + rho * a * zy,
                          rho * w  + rho * a * zz,rhoE + p + rho * a * theta, rho * Y], axis=0)

    lambda1 = theta - a
    lambda2 = theta
    lambda4 = theta + a
    eps = 1e-6

    lap1 = 0.5 * (lambda1 + jnp.sqrt(jnp.power(lambda1, 2) + eps**2))
    lam1 = 0.5 * (lambda1 - jnp.sqrt(jnp.power(lambda1, 2) + eps**2))

    lap2 = 0.5 * (lambda2 + jnp.sqrt(jnp.power(lambda2, 2) + eps**2))
    lam2 = 0.5 * (lambda2 - jnp.sqrt(jnp.power(lambda2, 2) + eps**2))

    lap4 = 0.5 * (lambda4 + jnp.sqrt(jnp.power(lambda4, 2) + eps**2))
    lam4 = 0.5 * (lambda4 - jnp.sqrt(jnp.power(lambda4, 2) + eps**2))

    Hplus = lap1 * H1 + lap2 * H2 + lap4 * H4
    Hminus = lam1 * H1 + lam2 * H2 + lam4 * H4

    return Hplus, Hminus
