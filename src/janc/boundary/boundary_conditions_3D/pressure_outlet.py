import jax
import jax.numpy as jnp
from ...model import thermo_model as thermo

def pressure_outlet(state_out,gamma_out,T_out,normal_vel,Pb):
    rho_out = state_out[0:1]
    u_out = state_out[1:2]/rho_out
    v_out = state_out[2:3]/rho_out
    w_out = state_out[3:4]/rho_out
    Y_out = state_out[5:]/rho_out
    R_out = thermo.get_R(Y_out)
    p_out = rho_out*(R_out*T_out)
    a_out = jnp.sqrt(gamma_out*p_out/rho_out)
    mask = (normal_vel/a_out < 1)
    rho_cor_out = jax.lax.select(mask, Pb / (p_out / rho_out),rho_out)
    p_cor_out = jax.lax.select(mask, Pb*jnp.ones_like(p_out),p_out)
    T_cor_out = jax.lax.select(mask, p_cor_out/(rho_cor_out*R_out),T_out)
    _, gamma_out, h_out, _, _ = thermo.get_thermo(T_cor_out,Y_out)
    U_bd = jnp.concatenate([rho_cor_out, rho_cor_out * u_out, rho_cor_out * v_out, rho_cor_out * w_out,
                      rho_cor_out*h_out - p_cor_out + 0.5 * rho_cor_out * (u_out ** 2 + v_out ** 2 + w_out ** 2),
                      rho_cor_out * Y_out], axis=0)
    aux_bd = jnp.concatenate([gamma_out,T_cor_out], axis=0)
    return U_bd, aux_bd

def left(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny,wall_nz = metrics['left_n_x'],metrics['left_n_y'],metrics['left_n_z']
    state_out = U_bd[:,0:1,:]
    gamma_out = aux_bd[0:1,0:1,:]
    T_out = aux_bd[1:2,0:1,:]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny, wall_nz = metrics['right_n_x'],metrics['right_n_y'],metrics['right_n_z']
    state_out = U_bd[:,-1:,:]
    gamma_out = aux_bd[0:1,-1:,:]
    T_out = aux_bd[1:2,-1:,:]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    return U_bd_ghost, aux_bd_ghost

def bottom(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny,wall_nz = metrics['bottom_n_x'],metrics['bottom_n_y'],metrics['bottom_n_z']
    state_out = U_bd[:,:,0:1]
    gamma_out = aux_bd[0:1,:,0:1]
    T_out = aux_bd[1:2,:,0:1]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    return U_bd_ghost, aux_bd_ghost

def top(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny,wall_nz = metrics['top_n_x'],metrics['top_n_y'],metrics['top_n_z']
    state_out = U_bd[:,:,-1:]
    gamma_out = aux_bd[0:1,:,-1:]
    T_out = aux_bd[1:2,:,-1:]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    return U_bd_ghost, aux_bd_ghost

def front(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny,wall_nz = metrics['front_n_x'],metrics['front_n_y'],metrics['front_n_z']
    state_out = U_bd[:,:,:,0:1]
    gamma_out = aux_bd[0:1,:,:,0:1]
    T_out = aux_bd[1:2,:,:,0:1]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,:,0:1],U_bd[:,:,:,0:1],U_bd[:,:,:,0:1]],axis=3)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1]],axis=3)
    return U_bd_ghost, aux_bd_ghost

def back(U_bd, aux_bd, metrics, theta):
    Pb = theta['Pb']
    wall_nx,wall_ny,wall_nz = metrics['back_n_x'],metrics['back_n_y'],metrics['back_n_z']
    state_out = U_bd[:,:,:,-1:]
    gamma_out = aux_bd[0:1,:,:,-1:]
    T_out = aux_bd[1:2,:,:,-1:]
    normal_vel = -(state_out[1:2]/state_out[0:1]*wall_nx + state_out[2:3]/state_out[0:1]*wall_ny + state_out[3:4]/state_out[0:1]*wall_nz)
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,:,0:1],U_bd[:,:,:,0:1],U_bd[:,:,:,0:1]],axis=3)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1]],axis=3)
    return U_bd_ghost, aux_bd_ghost

