import jax.numpy as jnp

def left(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['left_n_x'],metrics['left_n_y'],metrics['left_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['right_n_x'],metrics['right_n_y'],metrics['right_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-1:,:],U_bd[:,-1:,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,-1:,:],aux_bd[:,-1:,:],aux_bd[:,-1:,:]],axis=1)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def bottom(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['bottom_n_x'],metrics['bottom_n_y'],metrics['bottom_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def top(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['top_n_x'],metrics['top_n_y'],metrics['top_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-1:],U_bd[:,:,-1:]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,-1:],aux_bd[:,:,-1:],aux_bd[:,:,-1:]],axis=2)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def front(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['front_n_x'],metrics['front_n_y'],metrics['front_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,:,0:1],U_bd[:,:,:,0:1],U_bd[:,:,:,0:1]],axis=3)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1],aux_bd[:,:,:,0:1]],axis=3)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def back(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny,wall_nz = metrics['back_n_x'],metrics['back_n_y'],metrics['back_n_z']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,:,-1:],U_bd[:,:,:,-1:],U_bd[:,:,:,-1:]],axis=3)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,:,-1:],aux_bd[:,:,:,-1:],aux_bd[:,:,:,-1:]],axis=3)
    rhou,rhov,rhow = U_bd_ghost[1:2],U_bd_ghost[2:3],U_bd_ghost[3:4]
    normal_vel = rhou*wall_nx + rhov*wall_ny + rhow*wall_nz
    rhoun = normal_vel*wall_nx
    rhovn = normal_vel*wall_ny
    rhown = normal_vel*wall_nz
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    rhow_new = rhow - 2*rhown
    U_bd_ghost = U_bd_ghost.at[1:4].set(jnp.concatenate([rhou_new,rhov_new,rhow_new],axis=0))
    return U_bd_ghost, aux_bd_ghost