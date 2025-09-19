import jax
import jax.numpy as jnp

##parallel settings##
devices = jax.devices()
num_devices = len(devices)

def split_and_distribute_grid(grid,split_axis=1):
    nx = jnp.size(grid,axis=split_axis)
    assert nx % num_devices == 0, "nx should be divisible by number of avaliable devices"
    shards = jnp.split(grid,num_devices,axis=split_axis)
    sharded_grid = jax.device_put_sharded(shards, devices)
    return sharded_grid

def split_face(face, split_axis=1):
    n_cells = face.shape[split_axis]-1
    cells_per_device = n_cells // num_devices
    face_list = []
    for i in range(num_devices):
        start = i*cells_per_device
        end = (i+1)*cells_per_device + 1
        face_list.append(face[:,start:end])
    sharded_face = jax.device_put_sharded(face_list, devices)
    return sharded_face

def split_ghost_grid(ghost_grid, split_axis=1):
    n_cells = ghost_grid.shape[split_axis] - 2*3
    cells_per_device = n_cells // num_devices
    cell_list = []
    for i in range(num_devices):
        start = i*cells_per_device
        end = (i+1)*cells_per_device + 2*3
        cell_list.append(ghost_grid[:,start:end])
    sharded_grid = jax.device_put_sharded(cell_list, devices)
    return sharded_grid


    



