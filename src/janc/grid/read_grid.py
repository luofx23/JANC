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

def set_grid_1D(grid_config):
    global metrics,dx,dy,nx,ny
    if 'grid_file_path' in grid_config:
        grid_path = read_dat(grid_config['grid_file_path'])
        _, ext = os.path.splitext(grid_path)
        assert ext.lower() == '.dat', "janc only read grid file with Pointwise_R18+【.dat】 format."
        if not os.path.isfile(grid_path):
            raise FileNotFoundError('No grid file detected in the specified directory.')
        X = read_dat(grid_path)
        nx = X.shape[0]-1
        x_f = X[1:]
        x_b = X[:-1]
        η_dx = x_f - x_b
        dx_dξ = η_dx
        J = η_dx
        Jc = J
        dξ_dx = 1/dx_dξ
        
        metrics={'J':J[None,:],'Jc':jnp.pad(Jc[None,:],((0,0),(3,3)),mode='edge'),
                 'dξ_dx':jnp.pad(dξ_dx[None,:],((0,0),(3,3)),mode='edge')}
    else:
        metrics={
         'J':1.0,'Jc':1.0,
         'dξ_dx':1.0}     
        Lx = grid_config['Lx']
        nx = grid_config['Nx']
        dx = Lx/nx
        metrics['J'] = dx
        metrics['Jc'] = dx
        metrics['dξ_dx'] = 1.0/dx
    return metrics




def set_grid_2D(grid_config):
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
    return metrics

def get_inv(a11,a12,a13,a21,a22,a23,a31,a32,a33):
	J = a11*a22*a33-a11*a23*a32-a12*a21*a33+a12*a23*a31+a13*a21*a32-a13*a22*a31
	b11 = a22*a33-a23*a32
	b12 = a13*a32-a12*a33
	b13 = a12*a23-a13*a22
	b21 = a23*a31-a21*a33
	b22 = a11*a33-a13*a31
	b23 = a13*a21-a11*a23
	b31 = a21*a32-a22*a31
	b32 = a12*a31-a11*a32
	b33 = a11*a22-a12*a21
	return b11/J,b12/J,b13/J,b21/J,b22/J,b23/J,b31/J,b32/J,b33/J,J
	
def set_grid_3D(grid_config):
	X,Y,Z = read_CGNS(grid_config['grid_file_path'])
	nx,ny,nz = X.shape[0]-1,X.shape[1]-1,X.shape[2]-1
	x1,y1,z1 = X[:-1,:-1,:-1],Y[:-1,:-1,:-1],Z[:-1,:-1,:-1]
	x2,y2,z2 = X[1:,:-1,:-1],Y[1:,:-1,:-1],Z[1:,:-1,:-1]
	x3,y3,z3 = X[:-1,1:,:-1],Y[:-1,1:,:-1],Z[:-1,1:,:-1]
	x4,y4,z4 = X[:-1,:-1,1:],Y[:-1,:-1,1:],Z[:-1,:-1,1:]
	V1 = jnp.stack([x2-x1,y2-y1,z2-z1],axis=0)
	V2 = jnp.stack([x3-x1,y3-y1,z3-z1],axis=0)
	V3 = jnp.stack([x4-x1,y4-y1,z4-z1],axis=0)
	n = jnp.cross(V1,V2,axis=0)
	n_norm = jnp.linalg.norm(n,axis=0)
	ζ_n_x,ζ_n_y,ζ_n_z = n[0]/n_norm,n[1]/n_norm,n[2]/n_norm
	n = jnp.cross(V3,V1,axis=0)
	n_norm = jnp.linalg.norm(n,axis=0)
	η_n_x,η_n_y,η_n_z = n[0]/n_norm,n[1]/n_norm,n[2]/n_norm
	n = jnp.cross(V2,V3,axis=0)
	n_norm = jnp.linalg.norm(n,axis=0)
	ξ_n_x,ξ_n_y,ξ_n_z = n[0]/n_norm,n[1]/n_norm,n[2]/n_norm
	ζ_τ1_x,ζ_τ1_y,ζ_τ1_z = V1[0],V1[1],V1[2]
	ζ_τ2_x,ζ_τ2_y,ζ_τ2_z = V2[0],V2[1],V2[2]
	η_τ1_x,η_τ1_y,η_τ1_z = V3[0],V3[1],V3[2]
	η_τ2_x,η_τ2_y,η_τ2_z = V1[0],V1[1],V1[2]
	ξ_τ1_x,ξ_τ1_y,ξ_τ1_z = V2[0],V2[1],V2[2]
	ξ_τ2_x,ξ_τ2_y,ξ_τ2_z = V3[0],V3[1],V3[2]
	ξ11,ξ12,ξ13,ξ21,ξ22,ξ23,ξ31,ξ32,ξ33,_ = get_inv(ξ_n_x,ξ_n_y,ξ_n_z,ξ_τ1_x,ξ_τ1_y,ξ_τ1_z,ξ_τ2_x,ξ_τ2_y,ξ_τ2_z)
	η11,η12,η13,η21,η22,η23,η31,η32,η33,_ = get_inv(η_τ1_x,η_τ1_y,η_τ1_z,η_n_x,η_n_y,η_n_z,η_τ2_x,η_τ2_y,η_τ2_z)
	ζ11,ζ12,ζ13,ζ21,ζ22,ζ23,ζ31,ζ32,ζ33,_ = get_inv(ζ_τ1_x,ζ_τ1_y,ζ_τ1_z,ζ_τ2_x,ζ_τ2_y,ζ_τ2_z,ζ_n_x,ζ_n_y,ζ_n_z)
	xf, yf, zf = X[1:],Y[1:],Z[1:]
	xb, yb, zb = X[:-1],Y[:-1],Z[:-1]
	d_x,d_y,d_z = xf-xb,yf-yb,zf-zb
	d_x,d_y,d_z = 0.5*(d_x[:,1:]+d_x[:,:-1]),0.5*(d_y[:,1:]+d_y[:,:-1]),0.5*(d_z[:,1:]+d_z[:,:-1])
	dx_dξ,dy_dξ,dz_dξ = 0.5*(d_x[:,:,1:]+d_x[:,:,:-1]),0.5*(d_y[:,:,1:]+d_y[:,:,:-1]),0.5*(d_z[:,:,1:]+d_z[:,:,:-1])
	xf, yf, zf = X[:,1:],Y[:,1:],Z[:,1:]
	xb, yb, zb = X[:,:-1],Y[:,:-1],Z[:,:-1]
	d_x,d_y,d_z = xf-xb,yf-yb,zf-zb
	d_x,d_y,d_z = 0.5*(d_x[1:]+d_x[:-1]),0.5*(d_y[1:]+d_y[:-1]),0.5*(d_z[1:]+d_z[:-1])
	dx_dη,dy_dη,dz_dη = 0.5*(d_x[:,:,1:]+d_x[:,:,:-1]),0.5*(d_y[:,:,1:]+d_y[:,:,:-1]),0.5*(d_z[:,:,1:]+d_z[:,:,:-1])
	xf, yf, zf = X[:,:,1:],Y[:,:,1:],Z[:,:,1:]
	xb, yb, zb = X[:,:,:-1],Y[:,:,:-1],Z[:,:,:-1]
	d_x,d_y,d_z = xf-xb,yf-yb,zf-zb
	d_x,d_y,d_z = 0.5*(d_x[1:]+d_x[:-1]),0.5*(d_y[1:]+d_y[:-1]),0.5*(d_z[1:]+d_z[:-1])
	dx_dζ,dy_dζ,dz_dζ = 0.5*(d_x[:,1:]+d_x[:,:-1]),0.5*(d_y[:,1:]+d_y[:,:-1]),0.5*(d_z[:,1:]+d_z[:,:-1])
	dξ_dx,dη_dx,dζ_dx,dξ_dy,dη_dy,dζ_dy,dξ_dz,dη_dz,dζ_dz,J = get_inv(dx_dξ,dy_dξ,dz_dξ,dx_dη,dy_dη,dz_dη,dx_dζ,dy_dζ,dz_dζ)
	left_n_x,left_n_y,left_n_z = ξ_n_x[0:1],ξ_n_y[0:1],ξ_n_z[0:1]
	right_n_x,right_n_y,right_n_z = -ξ_n_x[-1:],-ξ_n_y[-1:],-ξ_n_z[-1:]
	bottom_n_x,bottom_n_y,bottom_n_z = η_n_x[0:1],η_n_y[0:1],η_n_z[0:1]
	top_n_x,top_n_y,top_n_z = -η_n_x[-1:],-η_n_y[-1:],-η_n_z[-1:]
	front_n_x,front_n_y,front_n_z = ζ_n_x[0:1],ζ_n_y[0:1],ζ_n_z[0:1]
	back_n_x,back_n_y,back_n_z = -ζ_n_x[-1:],-ζ_n_y[-1:],-ζ_n_z[-1:]
	
	
	
	
	
	





