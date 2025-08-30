import jax.numpy as jnp
from ...model import thermo_model as thermo


def riemann_flux(q_L_x,q_R_x,q_L_y,q_R_y,q_L_z,q_R_z):
    rho_L_x,rho_R_x,rho_L_y,rho_R_y,rho_L_z,rho_R_z = q_L_x[0:1],q_R_x[0:1],q_L_y[0:1],q_R_y[0:1],q_L_z[0:1],q_R_z[0:1]
    p_L_x,p_R_x,p_L_y,p_R_y,p_L_z,p_R_z = q_L_x[4:5],q_R_x[4:5],q_L_y[4:5],q_R_y[4:5],q_L_z[4:5],q_R_z[4:5]
    Y_L_x,Y_R_x,Y_L_y,Y_R_y,Y_L_z,Y_R_z = q_L_x[5:],q_R_x[5:],q_L_y[5:],q_R_y[5:],q_L_z[5:],q_R_z[5:]
    R_L_x,R_R_x,R_L_y,R_R_y,R_L_z,R_R_z = thermo.get_R(Y_L_x),thermo.get_R(Y_R_x),thermo.get_R(Y_L_y),thermo.get_R(Y_R_y),thermo.get_R(Y_L_z),thermo.get_R(Y_R_z)
    T_L_x,T_R_x,T_L_y,T_R_y,T_L_z,T_R_z = p_L_x/(rho_L_x*R_L_x),p_R_x/(rho_R_x*R_R_x),p_L_y/(rho_L_y*R_L_y),p_R_y/(rho_R_y*R_R_y),p_L_z/(rho_L_z*R_L_z),p_R_z/(rho_R_z*R_R_z)
    _,gamma_L_x,h_L_x,_,_ = thermo.get_thermo(T_L_x,Y_L_x)
    _,gamma_R_x,h_R_x,_,_ = thermo.get_thermo(T_R_x,Y_R_x)
    _,gamma_L_y,h_L_y,_,_ = thermo.get_thermo(T_L_y,Y_L_y)
    _,gamma_R_y,h_R_y,_,_ = thermo.get_thermo(T_R_y,Y_R_y)
    _,gamma_L_z,h_L_z,_,_ = thermo.get_thermo(T_L_z,Y_L_z)
    _,gamma_R_z,h_R_z,_,_ = thermo.get_thermo(T_R_z,Y_R_z)
    
    #x-flux
    a_L = jnp.sqrt(gamma_L_x*p_L_x/rho_L_x)
    a_R = jnp.sqrt(gamma_R_x*p_R_x/rho_R_x)
    u_L,u_R = q_L_x[1:2],q_R_x[1:2]
    v_L,v_R = q_L_x[2:3],q_R_x[2:3]
    w_L,w_R = q_L_x[3:4],q_R_x[3:4]
    E_L = rho_L_x*h_L_x-p_L_x+0.5*rho_L_x*(u_L**2+v_L**2+w_L**2)
    E_R = rho_R_x*h_R_x-p_R_x+0.5*rho_R_x*(u_R**2+v_R**2+w_R**2)
    s_L = jnp.minimum(u_L-a_L,u_R-a_R)
    s_R = jnp.maximum(u_L+a_L,u_R+a_R)
    s_M = (p_R_x-p_L_x+rho_L_x*u_L*(s_L-u_L)-rho_R_x*u_R*(s_R-u_R))/(rho_L_x*(s_L-u_L)-rho_R_x*(s_R-u_R))
    U_L = jnp.concatenate([rho_L_x,rho_L_x*u_L,rho_L_x*v_L,rho_L_x*w_L,E_L,rho_L_x*Y_L_x],axis=0)
    U_L_M = rho_L_x*(s_L-u_L)/(s_L-s_M)*jnp.concatenate([jnp.ones_like(s_M),s_M,v_L,w_L,E_L/rho_L_x+(s_M-u_L)*(s_M+p_L_x/(rho_L_x*(s_L-u_L))),Y_L_x],axis=0)
    F_L = jnp.concatenate([rho_L_x*u_L,rho_L_x*u_L**2+p_L_x,rho_L_x*u_L*v_L,rho_L_x*u_L*w_L,
                           u_L*(E_L+p_L_x),rho_L_x*u_L*Y_L_x],axis=0)
    F_L_M = F_L + s_L*(U_L_M-U_L)
    U_R = jnp.concatenate([rho_R_x,rho_R_x*u_R,rho_R_x*v_R,rho_R_x*w_R,E_R,rho_R_x*Y_R_x],axis=0)
    U_R_M = rho_R_x*(s_R-u_R)/(s_R-s_M)*jnp.concatenate([jnp.ones_like(s_M),s_M,v_R,w_R,E_R/rho_R_x+(s_M-u_R)*(s_M+p_R_x/(rho_R_x*(s_R-u_R))),Y_R_x],axis=0)
    F_R = jnp.concatenate([rho_R_x*u_R,rho_R_x*u_R**2+p_R_x,rho_R_x*u_R*v_R,rho_R_x*u_R*w_R,
                           u_R*(E_R+p_R_x),rho_R_x*u_R*Y_R_x],axis=0)
    F_R_M = F_R + s_R*(U_R_M-U_R)
    
    cond1 = s_L>=0
    cond2 = jnp.logical_and(s_L<=0,s_M>=0)
    cond3 = jnp.logical_and(s_M<=0,s_R>=0)
    
    F1 = jnp.where(cond1,F_L,F_R)
    F2 = jnp.where(cond2,F_L_M,F1)
    F_HLLC = jnp.where(cond3,F_R_M,F2)
    
    #y-flux
    a_L = jnp.sqrt(gamma_L_y*p_L_y/rho_L_y)
    a_R = jnp.sqrt(gamma_R_y*p_R_y/rho_R_y)
    u_L,u_R = q_L_y[1:2],q_R_y[1:2]
    v_L,v_R = q_L_y[2:3],q_R_y[2:3]
    w_L,w_R = q_L_y[3:4],q_R_y[3:4]
    E_L = rho_L_y*h_L_y-p_L_y+0.5*rho_L_y*(u_L**2+v_L**2+w_L**2)
    E_R = rho_R_y*h_R_y-p_R_y+0.5*rho_R_y*(u_R**2+v_R**2+w_R**2)
    s_L = jnp.minimum(v_L-a_L,v_R-a_R)
    s_R = jnp.maximum(v_L+a_L,v_R+a_R)
    s_M = (p_R_y-p_L_y+rho_L_y*v_L*(s_L-v_L)-rho_R_y*v_R*(s_R-v_R))/(rho_L_y*(s_L-v_L)-rho_R_y*(s_R-v_R))
    U_L = jnp.concatenate([rho_L_y,rho_L_y*u_L,rho_L_y*v_L,rho_L_y*w_L,E_L,rho_L_y*Y_L_y],axis=0)
    U_L_M = rho_L_y*(s_L-v_L)/(s_L-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_L,s_M,w_L,E_L/rho_L_y+(s_M-v_L)*(s_M+p_L_y/(rho_L_y*(s_L-v_L))),Y_L_y],axis=0)
    G_L = jnp.concatenate([rho_L_y*v_L,rho_L_y*u_L*v_L,rho_L_y*v_L**2+p_L_y,rho_L_y*v_L*w_L,
                           v_L*(E_L+p_L_y),rho_L_y*v_L*Y_L_y],axis=0)
    G_L_M = G_L + s_L*(U_L_M-U_L)
    U_R = jnp.concatenate([rho_R_y,rho_R_y*u_R,rho_R_y*v_R,rho_R_y*w_R,E_R,rho_R_y*Y_R_y],axis=0)
    U_R_M = rho_R_y*(s_R-v_R)/(s_R-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_R,s_M,w_R,E_R/rho_R_y+(s_M-v_R)*(s_M+p_R_y/(rho_R_y*(s_R-v_R))),Y_R_y],axis=0)
    G_R = jnp.concatenate([rho_R_y*v_R,rho_R_y*u_R*v_R,rho_R_y*v_R**2+p_R_y,rho_R_y*v_R*w_R,
                           v_R*(E_R+p_R_y),rho_R_y*v_R*Y_R_y],axis=0)
    G_R_M = G_R + s_R*(U_R_M-U_R)
    
    cond1 = s_L>=0
    cond2 = jnp.logical_and(s_L<=0,s_M>=0)
    cond3 = jnp.logical_and(s_M<=0,s_R>=0)
    
    G1 = jnp.where(cond1,G_L,G_R)
    G2 = jnp.where(cond2,G_L_M,G1)
    G_HLLC = jnp.where(cond3,G_R_M,G2)
    
    #z-flux
    a_L = jnp.sqrt(gamma_L_z*p_L_z/rho_L_z)
    a_R = jnp.sqrt(gamma_R_z*p_R_z/rho_R_z)
    u_L,u_R = q_L_z[1:2],q_R_z[1:2]
    v_L,v_R = q_L_z[2:3],q_R_z[2:3]
    w_L,w_R = q_L_z[3:4],q_R_z[3:4]
    E_L = rho_L_z*h_L_z-p_L_z+0.5*rho_L_z*(u_L**2+v_L**2+w_L**2)
    E_R = rho_R_z*h_R_z-p_R_z+0.5*rho_R_z*(u_R**2+v_R**2+w_R**2)
    s_L = jnp.minimum(w_L-a_L,w_R-a_R)
    s_R = jnp.maximum(w_L+a_L,w_R+a_R)
    s_M = (p_R_z-p_L_z+rho_L_z*w_L*(s_L-w_L)-rho_R_z*w_R*(s_R-w_R))/(rho_L_z*(s_L-w_L)-rho_R_z*(s_R-w_R))
    U_L = jnp.concatenate([rho_L_z,rho_L_z*u_L,rho_L_z*v_L,rho_L_z*w_L,E_L,rho_L_z*Y_L_z],axis=0)
    U_L_M = rho_L_z*(s_L-w_L)/(s_L-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_L,v_L,s_M,E_L/rho_L_z+(s_M-w_L)*(s_M+p_L_z/(rho_L_z*(s_L-w_L))),Y_L_z],axis=0)
    H_L = jnp.concatenate([rho_L_z*w_L,rho_L_z*u_L*w_L,rho_L_z*v_L*w_L,rho_L_z*w_L**2+p_L_z,
                           w_L*(E_L+p_L_z),rho_L_z*w_L*Y_L_z],axis=0)
    H_L_M = H_L + s_L*(U_L_M-U_L)
    U_R = jnp.concatenate([rho_R_z,rho_R_z*u_R,rho_R_z*v_R,rho_R_z*w_R,E_R,rho_R_z*Y_R_z],axis=0)
    U_R_M = rho_R_z*(s_R-w_R)/(s_R-s_M)*jnp.concatenate([jnp.ones_like(s_M),u_R,v_R,s_M,E_R/rho_R_z+(s_M-w_R)*(s_M+p_R_z/(rho_R_z*(s_R-w_R))),Y_R_z],axis=0)
    H_R = jnp.concatenate([rho_R_z*w_R,rho_R_z*w_R*u_R,rho_R_z*w_R*v_R,rho_R_z*w_R**2+p_R_z,
                           w_R*(E_R+p_R_z),rho_R_z*w_R*Y_R_z],axis=0)
    H_R_M = H_R + s_R*(U_R_M-U_R)
    
    cond1 = s_L>=0
    cond2 = jnp.logical_and(s_L<=0,s_M>=0)
    cond3 = jnp.logical_and(s_M<=0,s_R>=0)
    
    H1 = jnp.where(cond1,H_L,H_R)
    H2 = jnp.where(cond2,H_L_M,H1)
    H_HLLC = jnp.where(cond3,H_R_M,H2)
    
    return F_HLLC,G_HLLC,H_HLLC

