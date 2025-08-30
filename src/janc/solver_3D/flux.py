import jax.numpy as jnp
from ..solver import aux_func
from .reconstruction import reconstruction_L_x_dict,reconstruction_R_x_dict,\
                            reconstruction_L_y_dict,reconstruction_R_y_dict,\
                            reconstruction_L_z_dict,reconstruction_R_z_dict,\
                            reconstruction_x_dict,reconstruction_y_dict,reconstruction_z_dict
from .finite_difference import d_dx_dict,d_dy_dict,d_dz_dict
from .riemann_solver import riemann_solver_dict
from .flux_splitting import split_flux_dict
from ..model import thermo_model, transport_model

solver_type = 'godunov'
interface_reconstruction = 'WENO5-JS'
riemann_solver = 'HLLC'
split_method = 'LF'
viscosity = 'off'
viscosity_discretization = 'CENTRAL6'


def set_flux_solver(flux_solver_config,transport_config=None):
    global solver_type,interface_reconstruction,riemann_solver,split_method,viscosity,viscosity_discretization
    solver_type = flux_solver_config['solver_type']
    if solver_type == 'godunov':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        riemann_solver = flux_solver_config['riemann_solver']
    elif solver_type == 'flux_splitting':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        split_method = flux_solver_config['split_method']
    else:
        raise KeyError("JANC only support 'godunov' and 'flux_splitting'")
    
    if flux_solver_config['viscosity'] == 'on':
        viscosity = 'on'
        transport_model.set_transport(transport_config,'3D')
        if 'viscosity_discretization' in flux_solver_config:
            viscosity_discretization = flux_solver_config['viscosity_discretization']
        
def godunov_flux(U,aux,metrics):
    rho,u,v,w,Y,p,a = aux_func.U_to_prim(U, aux)
    q = jnp.concatenate([rho,u,v,w,p,Y],axis=0)
    q_L_x = reconstruction_L_x_dict[interface_reconstruction](q)
    q_R_x = reconstruction_R_x_dict[interface_reconstruction](q)
    q_L_y = reconstruction_L_y_dict[interface_reconstruction](q)
    q_R_y = reconstruction_R_y_dict[interface_reconstruction](q)
    q_L_z = reconstruction_L_z_dict[interface_reconstruction](q)
    q_R_z = reconstruction_R_z_dict[interface_reconstruction](q)
    ξ_n_x,ξ_n_y,ξ_n_z = metrics['ξ-n_x'],metrics['ξ-n_y'],metrics['ξ-n_z']
    ξ_τ1_x,ξ_τ1_y,ξ_τ1_z = metrics['ξ-τ1_x'],metrics['ξ-τ1_y'],metrics['ξ-τ1_z']
    ξ_τ2_x,ξ_τ2_y,ξ_τ2_z = metrics['ξ-τ2_x'],metrics['ξ-τ2_y'],metrics['ξ-τ2_z']
    ξ11,ξ12,ξ13,ξ21,ξ22,ξ23,ξ31,ξ32,ξ33 = metrics['ξ11'],metrics['ξ12'],metrics['ξ13'],metrics['ξ21'],metrics['ξ22'],metrics['ξ23'],metrics['ξ31'],metrics['ξ32'],metrics['ξ33']
    η_n_x,η_n_y,η_n_z = metrics['η-n_x'],metrics['η-n_y'],metrics['η-n_z']
    η_τ1_x,η_τ1_y,η_τ1_z = metrics['η-τ1_x'],metrics['η-τ1_y'],metrics['η-τ1_z']
    η_τ2_x,η_τ2_y,η_τ2_z = metrics['η-τ2_x'],metrics['η-τ2_y'],metrics['η-τ2_z']
    η11,η12,η13,η21,η22,η23,η31,η32,η33 = metrics['η11'],metrics['η12'],metrics['η13'],metrics['η21'],metrics['η22'],metrics['η23'],metrics['η31'],metrics['η32'],metrics['η33']
    ζ_n_x,ζ_n_y,ζ_n_z = metrics['ζ-n_x'],metrics['ζ-n_y'],metrics['ζ-n_z']
    ζ_τ1_x,ζ_τ1_y,ζ_τ1_z = metrics['ζ-τ1_x'],metrics['ζ-τ1_y'],metrics['ζ-τ1_z']
    ζ_τ2_x,ζ_τ2_y,ζ_τ2_z = metrics['ζ-τ2_x'],metrics['ζ-τ2_y'],metrics['ζ-τ2_z']
    ζ11,ζ12,ζ13,ζ21,ζ22,ζ23,ζ31,ζ32,ζ33 = metrics['ζ11'],metrics['ζ12'],metrics['ζ13'],metrics['ζ21'],metrics['ζ22'],metrics['ζ23'],metrics['ζ31'],metrics['ζ32'],metrics['ζ33']
    u_L_x = q_L_x[1:2]*ξ_n_x + q_L_x[2:3]*ξ_n_y + q_L_x[3:4]*ξ_n_z
    u_R_x = q_R_x[1:2]*ξ_n_x + q_R_x[2:3]*ξ_n_y + q_R_x[3:4]*ξ_n_z
    v_L_x = q_L_x[1:2]*ξ_τ1_x + q_L_x[2:3]*ξ_τ1_y + q_L_x[3:4]*ξ_τ1_z
    v_R_x = q_R_x[1:2]*ξ_τ1_x + q_R_x[2:3]*ξ_τ1_y + q_R_x[3:4]*ξ_τ1_z
    w_L_x = q_L_x[1:2]*ξ_τ2_x + q_L_x[2:3]*ξ_τ2_y + q_L_x[3:4]*ξ_τ2_z
    w_R_x = q_R_x[1:2]*ξ_τ2_x + q_R_x[2:3]*ξ_τ2_y + q_R_x[3:4]*ξ_τ2_z
    
    v_L_y = q_L_y[1:2]*η_n_x + q_L_y[2:3]*η_n_y + q_L_y[3:4]*η_n_z
    v_R_y = q_R_y[1:2]*η_n_x + q_R_y[2:3]*η_n_y + q_R_y[3:4]*η_n_z
    u_L_y = q_L_y[1:2]*η_τ1_x + q_L_y[2:3]*η_τ1_y + q_L_y[3:4]*η_τ1_z
    u_R_y = q_R_y[1:2]*η_τ1_x + q_R_y[2:3]*η_τ1_y + q_R_y[3:4]*η_τ1_z
    w_L_y = q_L_y[1:2]*η_τ2_x + q_L_y[2:3]*η_τ2_y + q_L_y[3:4]*η_τ2_z
    w_R_y = q_R_y[1:2]*η_τ2_x + q_R_y[2:3]*η_τ2_y + q_R_y[3:4]*η_τ2_z
    
    w_L_z = q_L_z[1:2]*ζ_n_x + q_L_z[2:3]*ζ_n_y + q_L_z[3:4]*ζ_n_z
    w_R_z = q_R_z[1:2]*ζ_n_x + q_R_z[2:3]*ζ_n_y + q_R_z[3:4]*ζ_n_z
    u_L_z = q_L_z[1:2]*ζ_τ1_x + q_L_z[2:3]*ζ_τ1_y + q_L_z[3:4]*ζ_τ1_z
    u_R_z = q_R_z[1:2]*ζ_τ1_x + q_R_z[2:3]*ζ_τ1_y + q_R_z[3:4]*ζ_τ1_z
    v_L_z = q_L_z[1:2]*ζ_τ2_x + q_L_z[2:3]*ζ_τ2_y + q_L_z[3:4]*ζ_τ2_z
    v_R_z = q_R_z[1:2]*ζ_τ2_x + q_R_z[2:3]*ζ_τ2_y + q_R_z[3:4]*ζ_τ2_z
    q_L_x = q_L_x.at[1:4].set(jnp.concatenate([u_L_x,v_L_x,w_L_x]))
    q_R_x = q_R_x.at[1:4].set(jnp.concatenate([u_R_x,v_R_x,w_R_x]))
    q_L_y = q_L_y.at[1:4].set(jnp.concatenate([u_L_y,v_L_y,w_L_y]))
    q_R_y = q_R_y.at[1:4].set(jnp.concatenate([u_R_y,v_R_y,w_R_y]))
    q_L_z = q_L_z.at[1:4].set(jnp.concatenate([u_L_z,v_L_z,w_L_z]))
    q_R_z = q_R_z.at[1:4].set(jnp.concatenate([u_R_z,v_R_z,w_R_z]))
    F_interface,G_interface,H_interface = riemann_solver_dict[riemann_solver](q_L_x,q_R_x,q_L_y,q_R_y,q_L_z,q_R_z)
    F = jnp.concatenate([F_interface[0:1],F_interface[1:2]*ξ11+F_interface[2:3]*ξ12+F_interface[3:4]*ξ13,
                         F_interface[1:2]*ξ21+F_interface[2:3]*ξ22+F_interface[3:4]*ξ23,
                         F_interface[1:2]*ξ31+F_interface[2:3]*ξ32+F_interface[3:4]*ξ33,F_interface[4:]],axis=0)*metrics['ξ-dA']
    G = jnp.concatenate([G_interface[0:1],G_interface[1:2]*η11+G_interface[2:3]*η12+G_interface[3:4]*η13,
                         G_interface[1:2]*η21+G_interface[2:3]*η22+G_interface[3:4]*η23,
                         G_interface[1:2]*η31+G_interface[2:3]*η32+G_interface[3:4]*η33,G_interface[4:]],axis=0)*metrics['η-dA']
    H = jnp.concatenate([H_interface[0:1],H_interface[1:2]*ζ11+H_interface[2:3]*ζ12+H_interface[3:4]*ζ13,
                         H_interface[1:2]*ζ21+H_interface[2:3]*ζ22+H_interface[3:4]*ζ23,
                         H_interface[1:2]*ζ31+H_interface[2:3]*ζ32+H_interface[3:4]*ζ33,H_interface[4:]],axis=0)*metrics['ζ-dA']
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    dH = H[:,:,:,1:]-H[:,:,:,:-1]
    net_flux = (dF + dG + dH)/metrics['J']
    return -net_flux

def flux_splitting(U,aux,metrics):
    
    Fplus,Fminus = split_flux_dict[split_method](1,U,aux,metrics)
    Gplus,Gminus = split_flux_dict[split_method](2,U,aux,metrics)
    Hplus,Hminus = split_flux_dict[split_method](3,U,aux,metrics)
    Fp = reconstruction_L_x_dict[interface_reconstruction](Fplus)
    Fm = reconstruction_R_x_dict[interface_reconstruction](Fminus)
    Gp = reconstruction_L_y_dict[interface_reconstruction](Gplus)
    Gm = reconstruction_R_y_dict[interface_reconstruction](Gminus)
    Hp = reconstruction_L_z_dict[interface_reconstruction](Hplus)
    Hm = reconstruction_R_z_dict[interface_reconstruction](Hminus)
    F = Fp + Fm
    G = Gp + Gm
    H = Hp + Hm
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    dH = H[:,:,:,1:]-H[:,:,:,:-1]
    net_flux = (dF + dG + dH)/metrics['J']
    return -net_flux

advective_flux_dict = {'godunov':godunov_flux,
                       'flux_splitting':flux_splitting}    

def advective_flux(U,aux,metrics):
    return advective_flux_dict[solver_type](U,aux,metrics)


def viscous_flux_node(U, aux, metrics):
    ρ,u,v,w,Y,p,a = aux_func.U_to_prim(U,aux)
    T = aux[1:2]
    cp_k, _, h_k = thermo_model.get_thermo_properties(T[0])
    cp, _, _, _, _ = thermo_model.get_thermo(T,Y)
    Y = thermo_model.fill_Y(Y)
    u_ξ = d_dx_dict[viscosity_discretization](u,1.0);  u_η = d_dy_dict[viscosity_discretization](u,1.0);  u_ζ = d_dz_dict[viscosity_discretization](u,1.0)
    v_ξ = d_dx_dict[viscosity_discretization](v,1.0);  v_η = d_dy_dict[viscosity_discretization](v,1.0);  v_ζ = d_dz_dict[viscosity_discretization](v,1.0)
    w_ξ = d_dx_dict[viscosity_discretization](w,1.0);  w_η = d_dy_dict[viscosity_discretization](w,1.0);  w_ζ = d_dz_dict[viscosity_discretization](w,1.0)
    T_ξ = d_dx_dict[viscosity_discretization](T,1.0);  T_η = d_dy_dict[viscosity_discretization](T,1.0);  T_ζ = d_dz_dict[viscosity_discretization](T,1.0)
    Y_ξ = d_dx_dict[viscosity_discretization](Y,1.0);  Y_η = d_dy_dict[viscosity_discretization](Y,1.0);  Y_ζ = d_dz_dict[viscosity_discretization](Y,1.0)
    ξ_x, ξ_y, ξ_z = metrics['dξ_dx'], metrics['dξ_dy'], metrics['dξ_dz']
    η_x, η_y, η_z = metrics['dη_dx'], metrics['dη_dy'], metrics['dη_dz']
    ζ_x, ζ_y, ζ_z = metrics['dζ_dx'], metrics['dζ_dy'], metrics['dζ_dz']
    J = metrics['Jc']
    du_dx = ξ_x*u_ξ + η_x*u_η + ζ_x*u_ζ
    du_dy = ξ_y*u_ξ + η_y*u_η + ζ_y*u_ζ
    du_dz = ξ_z*u_ξ + η_z*u_η + ζ_z*u_ζ

    dv_dx = ξ_x*v_ξ + η_x*v_η + ζ_x*v_ζ
    dv_dy = ξ_y*v_ξ + η_y*v_η + ζ_y*v_ζ
    dv_dz = ξ_z*v_ξ + η_z*v_η + ζ_z*v_ζ

    dw_dx = ξ_x*w_ξ + η_x*w_η + ζ_x*w_ζ
    dw_dy = ξ_y*w_ξ + η_y*w_η + ζ_y*w_ζ
    dw_dz = ξ_z*w_ξ + η_z*w_η + ζ_z*w_ζ

    dT_dx = ξ_x*T_ξ + η_x*T_η + ζ_x*T_ζ
    dT_dy = ξ_y*T_ξ + η_y*T_η + ζ_y*T_ζ
    dT_dz = ξ_z*T_ξ + η_z*T_η + ζ_z*T_ζ
    
    dY_dx = ξ_x*Y_ξ + η_x*Y_η + ζ_x*Y_ζ
    dY_dy = ξ_y*Y_ξ + η_y*Y_η + ζ_y*Y_ζ
    dY_dz = ξ_z*Y_ξ + η_z*Y_η + ζ_z*Y_ζ
    mu,mu_t = transport_model.mu(ρ,T,metrics,du_dx,du_dy,du_dz,dv_dx,dv_dy,dv_dz,dw_dx,dw_dy,dw_dz)
    k = transport_model.kappa(mu, cp, mu_t)
    D_k = transport_model.D(mu,ρ,cp_k,mu_t)

    S11 = du_dx; S22 = dv_dy; S33 = dw_dz;
    S12 = 0.5 * (du_dy + dv_dx)
    S13 = 0.5 * (du_dz + dw_dx)
    S23 = 0.5 * (dv_dz + dw_dy)
    div_u = S11 + S22 + S33
    τ_xx = 2*mu*du_dx -2/3*mu*div_u
    τ_yy = 2*mu*dv_dy -2/3*mu*div_u
    τ_zz = 2*mu*dw_dz -2/3*mu*div_u
    τ_xy = 2*mu*S12
    τ_xz = 2*mu*S13
    τ_yz = 2*mu*S23

    qx = -k * dT_dx
    qy = -k * dT_dy
    qz = -k * dT_dz
    
    jx =  - ρ * D_k * dY_dx
    jy =  - ρ * D_k * dY_dy
    jz =  - ρ * D_k * dY_dz
    
    ex = jnp.sum(jx*h_k,axis=0,keepdims=True)
    ey = jnp.sum(jy*h_k,axis=0,keepdims=True)
    ez = jnp.sum(jz*h_k,axis=0,keepdims=True)

    # 3) 节点物理黏性通量向量
    Fv = jnp.concatenate([jnp.zeros_like(ρ),
                         τ_xx, τ_xy, τ_xz,
                         u*τ_xx + v*τ_xy + w*τ_xz - qx - ex, -jx[0:-1]], axis=0)

    Gv = jnp.concatenate([jnp.zeros_like(ρ),
                         τ_xy, τ_yy, τ_yz,
                         u*τ_xy + v*τ_yy + w*τ_yz - qy - ey, -jy[0:-1]], axis=0)

    Hv = jnp.concatenate([jnp.zeros_like(ρ),
                         τ_xz, τ_yz, τ_zz,
                         u*τ_xz + v*τ_yz + w*τ_zz - qz - ez, -jz[0:-1]], axis=0)
    F_hat = J * (ξ_x*Fv + ξ_y*Gv + ξ_z*Hv)
    G_hat = J * (η_x*Fv + η_y*Gv + η_z*Hv)
    H_hat = J * (ζ_x*Fv + ζ_y*Gv + ζ_z*Hv)
    return F_hat, G_hat, H_hat

def viscous_flux(U, aux, metrics):
    F_hat, G_hat, H_hat = viscous_flux_node(U, aux, metrics)
    F_interface = reconstruction_x_dict[viscosity_discretization](F_hat)
    G_interface = reconstruction_y_dict[viscosity_discretization](G_hat)
    H_interface = reconstruction_z_dict[viscosity_discretization](H_hat)
    dF = F_interface[:,1:,:]-F_interface[:,:-1,:]
    dG = G_interface[:,:,1:]-G_interface[:,:,:-1]
    dH = H_interface[:,:,:,1:]-H_interface[:,:,:,:-1]
    net_flux = (dF + dG + dH)/metrics['J']
    return net_flux

def NS_flux(U,aux,metrics):
    return advective_flux(U,aux,metrics) + viscous_flux(U, aux, metrics)

def Euler_flux(U,aux, metrics):
    return advective_flux(U,aux,metrics)

total_flux_dict = {'on':NS_flux,
                   'off':Euler_flux}

def total_flux(U,aux,metrics):
    return total_flux_dict[viscosity](U,aux,metrics)

