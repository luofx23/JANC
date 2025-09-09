import jax.numpy as jnp

def left(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny = metrics['left_n_x'],metrics['left_n_y']
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    rhou,rhov = U_bd_ghost[1:2],U_bd_ghost[2:3]
    rhoun = (rhou*wall_nx + rhov*wall_ny)*wall_nx
    rhovn = (rhou*wall_nx + rhov*wall_ny)*wall_ny
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    U_bd_ghost = U_bd_ghost.at[1:3].set(jnp.concatenate([rhou_new,rhov_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny = metrics['right_n_x'],metrics['right_n_y']
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-1:,:],U_bd[:,-1:,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,-1:,:],aux_bd[:,-1:,:],aux_bd[:,-1:,:]],axis=1)
    rhou,rhov = U_bd_ghost[1:2],U_bd_ghost[2:3]
    rhoun = (rhou*wall_nx + rhov*wall_ny)*wall_nx
    rhovn = (rhou*wall_nx + rhov*wall_ny)*wall_ny
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    U_bd_ghost = U_bd_ghost.at[1:3].set(jnp.concatenate([rhou_new,rhov_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def bottom(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny = metrics['bottom_n_x'],metrics['bottom_n_y']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    rhou,rhov = U_bd_ghost[1:2],U_bd_ghost[2:3]
    rhoun = (rhou*wall_nx + rhov*wall_ny)*wall_nx
    rhovn = (rhou*wall_nx + rhov*wall_ny)*wall_ny
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    U_bd_ghost = U_bd_ghost.at[1:3].set(jnp.concatenate([rhou_new,rhov_new],axis=0))
    return U_bd_ghost, aux_bd_ghost

def top(U_bd, aux_bd, metrics, theta=None):
    wall_nx,wall_ny = metrics['top_n_x'],metrics['top_n_y']
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-1:],U_bd[:,:,-1:]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,-1:],aux_bd[:,:,-1:],aux_bd[:,:,-1:]],axis=2)
    rhou,rhov = U_bd_ghost[1:2],U_bd_ghost[2:3]
    rhoun = (rhou*wall_nx + rhov*wall_ny)*wall_nx
    rhovn = (rhou*wall_nx + rhov*wall_ny)*wall_ny
    rhou_new = rhou - 2*rhoun
    rhov_new = rhov - 2*rhovn
    U_bd_ghost = U_bd_ghost.at[1:3].set(jnp.concatenate([rhou_new,rhov_new],axis=0))
    return U_bd_ghost, aux_bd_ghost