import jax.numpy as jnp
import numpy as np
import os

metrics={'ξ-n_x':1.0,'ξ-n_y':0.0,
         'η-n_x':0.0,'η-n_y':1.0,
         'ξ-dl':1.0,'η-dl':1.0,
         'J':1.0,'Jc':1.0,
         'dξ_dx':1.0,'dη_dx':0.0,
         'dξ_dy':0.0,'dη_dy':1.0,
         'left_n_x':1.0,'left_n_y':0.0,
         'right_n_x':-1.0,'right_n_y':0.0,
         'bottom_n_x':0.0,'bottom_n_y':1.0,
         'top_n_x':0.0,'top_n_y':-1.0}
dx = 1.0
dy = 1.0
nx = 5
ny = 5

def read_dat(file_path):
    raw_grid = np.loadtxt(file_path, usecols=(0,1))
    nx, ny = raw_grid[0,0],raw_grid[0,1]
    X = jnp.array(raw_grid[1:,0].reshape((nx,ny)))
    Y = jnp.array(raw_grid[1:,1].reshape((nx,ny)))
    return X, Y

def set_grid(grid_config):
    global metrics,dx,dy,nx,ny
    if 'grid_file_path' in grid_config:
        grid_path = read_dat(grid_config['grid_file_path'])
        _, ext = os.path.splitext(grid_path)
        assert ext.lower() == '.dat', "janc only read grid file with Pointwise_R18+【.dat】 format."
        if not os.path.isfile(grid_path):
            raise FileNotFoundError('No grid file detected in the specified directory.')
        X,Y = read_dat(grid_path)
        nx,ny = X.shape[0]-1,X.shape[1]-1
        x_f,y_f = X[:,1:],Y[:,1:]
        x_b,y_b = X[:,:-1],Y[:,:-1]
        ξ_dx = x_f - x_b
        dx_dη = 0.5*(ξ_dx[1:,:]+ξ_dx[:-1,:])
        ξ_dy = y_f - y_b
        dy_dη = 0.5*(ξ_dy[1:,:]+ξ_dy[:-1,:])
        ξ_dl = jnp.sqrt(ξ_dx**2+ξ_dy**2)
        theta = jnp.atan(ξ_dy/ξ_dx)
        theta = theta + (theta<0)*jnp.pi
        normal_theta = theta - jnp.pi/2
        ξ_n_x = jnp.cos(normal_theta)
        ξ_n_y = jnp.sin(normal_theta)
        left_n_x,left_n_y = ξ_n_x[:,0:1],ξ_n_y[:,0:1]
        right_n_x,right_n_y = -ξ_n_x[:,-1:],ξ_n_y[:,-1:]
        
        x_f,y_f = X[1:,:],Y[1:,:]
        x_b,y_b = X[:-1,:],Y[:-1,:]
        η_dx = x_f - x_b
        dx_dξ = 0.5*(η_dx[:,1:]+η_dx[:,:-1])
        η_dy = y_f - y_b
        dy_dξ = 0.5*(η_dy[:,1:]+η_dy[:,:-1])
        η_dl = jnp.sqrt(η_dx**2+η_dy**2)
        theta = jnp.atan(η_dy/η_dx)
        normal_theta = theta + jnp.pi/2#(theta<0)*jnp.pi/2 - (theta>=0)*jnp.pi/2
        η_n_x = jnp.cos(normal_theta)
        η_n_y = jnp.sin(normal_theta)
        bottom_n_x,bottom_n_y = η_n_x[0:1,:],η_n_y[0:1,:]
        top_n_x,top_n_y = η_n_x[-1:,:],-η_n_y[-1:,:]
                
        x1,y1 = X[:-1,1:],Y[:-1,1:]
        x2,y2 = X[:-1,:-1],Y[:-1,:-1]
        x3,y3 = X[1:,:-1],Y[1:,:-1]
        x4,y4 = X[1:,1:],Y[1:,1:]
        
        J = 0.5*jnp.abs(x1*y2+x2*y3+x3*y4+x4*y1-(x2*y1+x3*y2+x4*y3+x1*y4))
        
        Jc = dx_dξ * dy_dη - dx_dη * dy_dξ
        dξ_dx = dy_dη/Jc
        dη_dx = -dy_dξ/Jc
        dξ_dy = -dx_dη/Jc
        dη_dy = dx_dξ/Jc
        
        metrics={'ξ-n_x':ξ_n_x[None,:,:],'ξ-n_y':ξ_n_y[None,:,:],
                 'η-n_x':η_n_x[None,:,:],'η-n_y':η_n_y[None,:,:],
                 'ξ-dl':ξ_dl[None,:,:],'η-dl':η_dl[None,:,:],
                 'J':J[None,:,:],'Jc':jnp.pad(Jc[None,:,:],((0,0),(3,3),(3,3)),mode='edge'),
                 'dξ_dx':jnp.pad(dξ_dx[None,:,:],((0,0),(3,3),(3,3)),mode='edge'),
                 'dη_dx':jnp.pad(dη_dx[None,:,:],((0,0),(3,3),(3,3)),mode='edge'),
                 'dξ_dy':jnp.pad(dξ_dy[None,:,:],((0,0),(3,3),(3,3)),mode='edge'),
                 'dη_dy':jnp.pad(dη_dy[None,:,:],((0,0),(3,3),(3,3)),mode='edge'),
                 'left_n_x':left_n_x[None,:,:],'left_n_y':left_n_y[None,:,:],
                 'right_n_x':right_n_x[None,:,:],'right_n_y':right_n_y[None,:,:],
                 'bottom_n_x':bottom_n_x[None,:,:],'bottom_n_y':bottom_n_y[None,:,:],
                 'top_n_x':top_n_x[None,:,:],'top_n_y':top_n_y[None,:,:]}
    else:
        Lx = grid_config['Lx']
        Ly = grid_config['Ly']
        nx = grid_config['Nx']
        ny = grid_config['Ny']
        dx = Lx/nx
        dy = Ly/ny
        metrics['ξ-dl'] = dy
        metrics['η-dl'] = dx
        metrics['J'] = dx*dy
        metrics['Jc'] = dx*dy
        metrics['dξ_dx'] = 1/dx
        metrics['dη_dy'] = 1/dy