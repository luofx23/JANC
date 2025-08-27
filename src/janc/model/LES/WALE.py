import jax.numpy as jnp

von_karman_constant = 0.40
C_w = 0.325

def construct_distance_matrix(rho,dx,dy,dz):
    _,nx,ny,nz = rho.shape
    x = jnp.linspace(0+0.5*dx, dx*nx-0.5*dx, nx)
    y = jnp.linspace(0+0.5*dy, dy*ny-0.5*dy, ny)
    z = jnp.linspace(0+0.5*dz, dz*nz-0.5*dz, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    left_d = X
    right_d = nx*dx-X
    bottom_d = Y
    up_d = ny*dy-Y
    front_d = Z
    back_d = nz*dz-Z
    x_d = jnp.minimum(left_d,right_d)
    y_d = jnp.minimum(bottom_d,up_d)
    z_d = jnp.minimum(front_d,back_d)
    d1 = jnp.minimum(x_d,y_d)
    d = jnp.minimum(d1,z_d)
    return d[None,:,:,:]

def get_Ls(rho,V):
    #V = dx*dy*dz
    #d = construct_distance_matrix(rho, dx, dy, dz)
    #cof1 = von_karman_constant*d
    #cof2 = jnp.full_like(cof1,C_w*(V)**(1/3))
    cof2 = C_w*(V**(1/3))
    Ls = cof2#jnp.minimum(cof1,cof2)
    return Ls

def mu_t(rho,V,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz):
    g_sum = dudx**2 + dvdy**2 + dwdz**2
    S11_d = dudx**2 - 1/3*g_sum
    S22_d = dvdy**2 - 1/3*g_sum
    S33_d = dwdz**2 - 1/3*g_sum
    S12_d = 0.5*(dudy**2 + dvdx**2)
    S13_d = 0.5*(dudz**2 + dwdx**2)
    S23_d = 0.5*(dvdz**2 + dwdy**2)
    S11 = dudx; S22 = dvdy; S33 = dwdz;
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)
    S_d_sum = S11_d**2 + 2*S12_d**2 + 2*S13_d**2 + S22_d**2 + 2*S23_d**2 + S33_d**2
    S_sum = S11**2 + 2*S12**2 + 2*S13**2 + S22**2 + 2*S23**2 + S33**2
    Ls = get_Ls(rho,V)
    mu_t = rho*(Ls**2)*(S_d_sum**(3/2))/(S_sum**(5/2)+S_d_sum**(5/4))
    return mu_t


    
    