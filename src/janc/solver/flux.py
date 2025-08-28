import jax.numpy as jnp
from ..solver import aux_func
from .reconstruction import reconstruction_L_x_dict,reconstruction_R_x_dict,\
                            reconstruction_L_y_dict,reconstruction_R_y_dict,\
                            reconstruction_x_dict,reconstruction_y_dict
from .finite_difference import d_dx_dict,d_dy_dict
from .riemann_solver import riemann_solver_dict
from .flux_splitting import split_flux_dict
from ..model import thermo_model, transport_model

solver_type = 'godunov'
interface_reconstruction = 'WENO5-JS'
riemann_solver = 'HLLC'
split_method = 'LF'
viscosity = 'off'
viscosity_discretization = 'CENTRAL6'

def godunov_flux(U,aux,metrics,reconstruction_L_x,reconstruction_R_x,reconstruction_L_y,reconstruction_R_y,riemann_solver):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U, aux)
    ξ_n_x,ξ_n_y = metrics['ξ-n_x'],metrics['ξ-n_y']
    un = u*ξ_n_x + v*ξ_n_y
    ut = -u*ξ_n_y + v*ξ_n_x
    qx = jnp.concatenate([rho,un,ut,p,Y],axis=0)
    η_n_x,η_n_y = metrics['η-n_x'],metrics['η-n_y']
    un = u*η_n_x + v*η_n_y
    ut = u*η_n_y - v*η_n_x
    qy = jnp.concatenate([rho,ut,un,p,Y],axis=0)
    q_L_x = reconstruction_L_x(qx)
    q_R_x = reconstruction_R_x(qx)
    q_L_y = reconstruction_L_y(qy)
    q_R_y = reconstruction_R_y(qy)
    F_interface,G_interface = riemann_solver(q_L_x,q_R_x,q_L_y,q_R_y)
    F = jnp.concatenate([F_interface[0:1],F_interface[1:2]*ξ_n_x-F_interface[2:3]*ξ_n_y,
                         F_interface[1:2]*ξ_n_y+F_interface[2:3]*ξ_n_x,F_interface[3:]],axis=0)*metrics['ξ-dl']
    G = jnp.concatenate([G_interface[0:1],G_interface[1:2]*η_n_y+G_interface[2:3]*η_n_x,
                         -G_interface[1:2]*η_n_x+G_interface[2:3]*η_n_y,G_interface[3:]],axis=0)*metrics['η-dl']
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    net_flux = (dF + dG)/metrics['J']
    return -net_flux

def flux_splitting(U,aux,metrics,reconstruction_L_x,reconstruction_R_x,reconstruction_L_y,reconstruction_R_y,split_func):
    ξ_n_x,ξ_n_y = metrics['ξ-n_x'],metrics['ξ-n_y']
    Ux = jnp.concatenate([U[0:1],U[1:2]*ξ_n_x + U[2:3]*ξ_n_y, -U[1:2]*ξ_n_y + U[2:3]*ξ_n_x, U[3:]],axis=0)
    η_n_x,η_n_y = metrics['η-n_x'],metrics['η-n_y']
    Uy = jnp.concatenate([U[0:1],U[1:2]*η_n_y - U[2:3]*η_n_x, U[1:2]*η_n_x + U[2:3]*η_n_y, U[3:]],axis=0)
    Fplus,Fminus = split_func(1,Ux,aux)
    Gplus,Gminus = split_func(2,Uy,aux)
    Fp = reconstruction_L_x(Fplus)
    Fm = reconstruction_R_x(Fminus)
    Gp = reconstruction_L_y(Gplus)
    Gm = reconstruction_R_y(Gminus)
    F_interface = Fp + Fm
    G_interface = Gp + Gm
    F = jnp.concatenate([F_interface[0:1],F_interface[1:2]*ξ_n_x-F_interface[2:3]*ξ_n_y,
                         F_interface[1:2]*ξ_n_y+F_interface[2:3]*ξ_n_x,F_interface[3:]],axis=0)*metrics['ξ-dl']
    G = jnp.concatenate([G_interface[0:1],G_interface[1:2]*η_n_y+G_interface[2:3]*η_n_x,
                         -G_interface[1:2]*η_n_x+G_interface[2:3]*η_n_y,G_interface[3:]],axis=0)*metrics['η-dl']
    dF = F[:,1:,:]-F[:,:-1,:]
    dG = G[:,:,1:]-G[:,:,:-1]
    net_flux = (dF + dG)/metrics['J']
    return -net_flux

def viscous_flux_node(U, aux, metrics,d_dx,d_dy):
    ρ,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    T = aux[1:2]
    cp_k, _, h_k = thermo_model.get_thermo_properties(T[0])
    cp, _, _, _, _ = thermo_model.get_thermo(T,Y)
    Y = thermo_model.fill_Y(Y)
    du_dξ = d_dx(u,1.0);  du_dη = d_dy(u,1.0);
    dv_dξ = d_dx(v,1.0);  dv_dη = d_dy(v,1.0);
    dT_dξ = d_dx(T,1.0);  dT_dη = d_dy(T,1.0);
    dY_dξ = d_dx(Y,1.0);  dY_dη = d_dy(Y,1.0);
    dξ_dx, dξ_dy = metrics['dξ_dx'], metrics['dξ_dy']
    dη_dx, dη_dy = metrics['dη_dx'], metrics['dη_dy']
    J = metrics['Jc']
    du_dx = dξ_dx*du_dξ + dη_dx*du_dη
    du_dy = dξ_dy*du_dξ + dη_dy*du_dη
    dv_dx = dξ_dx*dv_dξ + dη_dx*dv_dη
    dv_dy = dξ_dy*dv_dξ + dη_dy*dv_dη
    dT_dx = dξ_dx*dT_dξ + dη_dx*dT_dη
    dT_dy = dξ_dy*dT_dξ + dη_dy*dT_dη
    dY_dx = dξ_dx*dY_dξ + dη_dx*dY_dη
    dY_dy = dξ_dy*dY_dξ + dη_dy*dY_dη
    mu,mu_t = transport_model.mu(ρ,T,J,du_dx,du_dy,None,dv_dx,dv_dy,None,None,None,None)
    k = transport_model.kappa(mu, cp, mu_t)
    D_k = transport_model.D(mu,cp_k,mu_t)
    λ = -2/3*mu
    S11 = du_dx; S22 = dv_dy;
    S12 = 0.5 * (du_dy + dv_dx)
    div_u = S11 + S22
    τ_xx = 2*mu*du_dx + λ*div_u
    τ_yy = 2*mu*dv_dy + λ*div_u
    τ_xy = 2*mu*S12
    qx = -k * dT_dx
    qy = -k * dT_dy
    jx =  - ρ * D_k * dY_dx
    jy =  - ρ * D_k * dY_dy
    ex = jnp.sum(jx*h_k,axis=0,keepdims=True)
    ey = jnp.sum(jy*h_k,axis=0,keepdims=True)
    Fv = jnp.concatenate([jnp.zeros_like(ρ),
                         τ_xx, τ_xy,
                         u*τ_xx + v*τ_xy - qx - ex, -jx[0:-1]], axis=0)
    Gv = jnp.concatenate([jnp.zeros_like(ρ),
                         τ_xy, τ_yy,
                         u*τ_xy + v*τ_yy - qy - ey, -jy[0:-1]], axis=0)
    F_hat = J * (dξ_dx*Fv + dξ_dy*Gv)
    G_hat = J * (dη_dx*Fv + dη_dy*Gv)
    return F_hat, G_hat

def viscous_flux_face(U, aux, metrics, d_dx, d_dy, reconstruction_x, reconstruction_y):
    F_hat, G_hat = viscous_flux_node(U, aux, metrics, d_dx, d_dy)
    F_interface = reconstruction_x(F_hat)
    G_interface = reconstruction_y(G_hat)
    dF = F_interface[:,1:,:]-F_interface[:,:-1,:]
    dG = G_interface[:,:,1:]-G_interface[:,:,:-1]
    net_flux = (dF + dG)/metrics['J']
    return net_flux



def set_flux_solver(flux_solver_config,transport_config=None):
    solver_type = flux_solver_config['solver_type']
    if solver_type == 'godunov':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        riemann_solver_type = flux_solver_config['riemann_solver']
        reconstruction_L_x = reconstruction_L_x_dict[interface_reconstruction]
        reconstruction_R_x = reconstruction_R_x_dict[interface_reconstruction]
        reconstruction_L_y = reconstruction_L_y_dict[interface_reconstruction]
        reconstruction_R_y = reconstruction_R_y_dict[interface_reconstruction]
        riemann_solver = riemann_solver_dict[riemann_solver_type]
        def advective_flux(U,aux,metrics):
            rho,u,v,Y,p,a = aux_func.U_to_prim(U, aux)
            ξ_n_x,ξ_n_y = metrics['ξ-n_x'],metrics['ξ-n_y']
            un = u*ξ_n_x + v*ξ_n_y
            ut = -u*ξ_n_y + v*ξ_n_x
            qx = jnp.concatenate([rho,un,ut,p,Y],axis=0)
            η_n_x,η_n_y = metrics['η-n_x'],metrics['η-n_y']
            un = u*η_n_x + v*η_n_y
            ut = u*η_n_y - v*η_n_x
            qy = jnp.concatenate([rho,ut,un,p,Y],axis=0)
            q_L_x = reconstruction_L_x(qx)
            q_R_x = reconstruction_R_x(qx)
            q_L_y = reconstruction_L_y(qy)
            q_R_y = reconstruction_R_y(qy)
            F_interface,G_interface = riemann_solver(q_L_x,q_R_x,q_L_y,q_R_y)
            F = jnp.concatenate([F_interface[0:1],F_interface[1:2]*ξ_n_x-F_interface[2:3]*ξ_n_y,
                                 F_interface[1:2]*ξ_n_y+F_interface[2:3]*ξ_n_x,F_interface[3:]],axis=0)*metrics['ξ-dl']
            G = jnp.concatenate([G_interface[0:1],G_interface[1:2]*η_n_y+G_interface[2:3]*η_n_x,
                                 -G_interface[1:2]*η_n_x+G_interface[2:3]*η_n_y,G_interface[3:]],axis=0)*metrics['η-dl']
            dF = F[:,1:,:]-F[:,:-1,:]
            dG = G[:,:,1:]-G[:,:,:-1]
            net_flux = (dF + dG)/metrics['J']
            return -net_flux
    elif solver_type == 'flux_splitting':
        interface_reconstruction = flux_solver_config['interface_reconstruction']
        split_method = flux_solver_config['split_method']
        reconstruction_L_x = reconstruction_L_x_dict[interface_reconstruction]
        reconstruction_R_x = reconstruction_R_x_dict[interface_reconstruction]
        reconstruction_L_y = reconstruction_L_y_dict[interface_reconstruction]
        reconstruction_R_y = reconstruction_R_y_dict[interface_reconstruction]
        split_func = split_flux_dict[split_method]
        def advective_flux(U,aux,metrics):
            ξ_n_x,ξ_n_y = metrics['ξ-n_x'],metrics['ξ-n_y']
            Ux = jnp.concatenate([U[0:1],U[1:2]*ξ_n_x + U[2:3]*ξ_n_y, -U[1:2]*ξ_n_y + U[2:3]*ξ_n_x, U[3:]],axis=0)
            #Ux = jnp.concatenate([U[0:1],U[1:2]*1.0 + U[2:3]*0.0, -U[1:2]*0.0 + U[2:3]*1.0, U[3:]],axis=0)
            η_n_x,η_n_y = metrics['η-n_x'],metrics['η-n_y']
            Uy = jnp.concatenate([U[0:1],U[1:2]*η_n_y - U[2:3]*η_n_x, U[1:2]*η_n_x + U[2:3]*η_n_y, U[3:]],axis=0)
            #Uy = jnp.concatenate([U[0:1],U[1:2]*1.0 - U[2:3]*0.0, U[1:2]*0.0 + U[2:3]*1.0, U[3:]],axis=0)
            Fplus,Fminus = split_func(1,Ux,aux)
            Gplus,Gminus = split_func(2,Uy,aux)
            Fp = reconstruction_L_x(Fplus)
            Fm = reconstruction_R_x(Fminus)
            Gp = reconstruction_L_y(Gplus)
            Gm = reconstruction_R_y(Gminus)
            F_interface = Fp + Fm
            G_interface = Gp + Gm
            #F = jnp.concatenate([F_interface[0:1],F_interface[1:2]*ξ_n_x-F_interface[2:3]*ξ_n_y,
                                 #F_interface[1:2]*ξ_n_y+F_interface[2:3]*ξ_n_x,F_interface[3:]],axis=0)*metrics['ξ-dl']
            F = jnp.concatenate([F_interface[0:1],F_interface[1:2]-F_interface[2:3]*0.0,
                                 F_interface[1:2]*0.0+F_interface[2:3],F_interface[3:]],axis=0)*metrics['ξ-dl']
            #G = jnp.concatenate([G_interface[0:1],G_interface[1:2]*η_n_y+G_interface[2:3]*η_n_x,
                                 #-G_interface[1:2]*η_n_x+G_interface[2:3]*η_n_y,G_interface[3:]],axis=0)*metrics['η-dl']
            G = jnp.concatenate([G_interface[0:1],G_interface[1:2]+G_interface[2:3]*0.0,
                                 -G_interface[1:2]*0.0+G_interface[2:3],G_interface[3:]],axis=0)*metrics['η-dl']
            dF = (F[:,1:,:]-F[:,:-1,:])
            dG = (G[:,:,1:]-G[:,:,:-1])
            net_flux = (dF + dG)/metrics['J']
            return -net_flux
    else:
        raise KeyError("JANC only support 'godunov' and 'flux_splitting'")
    
    if flux_solver_config['viscosity'] == 'on':
        transport_model.set_transport(transport_config)
        if 'viscosity_discretization' in flux_solver_config:
            viscosity_discretization = flux_solver_config['viscosity_discretization']
        else:
            viscosity_discretization = 'CENTRAL6'
        d_dx = d_dx_dict[viscosity_discretization]
        d_dy = d_dy_dict[viscosity_discretization]
        reconstruction_x = reconstruction_x_dict[viscosity_discretization]
        reconstruction_y = reconstruction_y_dict[viscosity_discretization]
        def viscous_flux_node(U, aux, metrics,d_dx,d_dy):
            ρ,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
            T = aux[1:2]
            cp_k, _, h_k = thermo_model.get_thermo_properties(T[0])
            cp, _, _, _, _ = thermo_model.get_thermo(T,Y)
            Y = thermo_model.fill_Y(Y)
            du_dξ = d_dx(u,1.0);  du_dη = d_dy(u,1.0);
            dv_dξ = d_dx(v,1.0);  dv_dη = d_dy(v,1.0);
            dT_dξ = d_dx(T,1.0);  dT_dη = d_dy(T,1.0);
            dY_dξ = d_dx(Y,1.0);  dY_dη = d_dy(Y,1.0);
            dξ_dx, dξ_dy = metrics['dξ_dx'], metrics['dξ_dy']
            dη_dx, dη_dy = metrics['dη_dx'], metrics['dη_dy']
            J = metrics['Jc']
            du_dx = dξ_dx*du_dξ + dη_dx*du_dη
            du_dy = dξ_dy*du_dξ + dη_dy*du_dη
            dv_dx = dξ_dx*dv_dξ + dη_dx*dv_dη
            dv_dy = dξ_dy*dv_dξ + dη_dy*dv_dη
            dT_dx = dξ_dx*dT_dξ + dη_dx*dT_dη
            dT_dy = dξ_dy*dT_dξ + dη_dy*dT_dη
            dY_dx = dξ_dx*dY_dξ + dη_dx*dY_dη
            dY_dy = dξ_dy*dY_dξ + dη_dy*dY_dη
            mu,mu_t = transport_model.mu(ρ,T,J,du_dx,du_dy,None,dv_dx,dv_dy,None,None,None,None)
            k = transport_model.kappa(mu, cp, mu_t)
            D_k = transport_model.D(mu,cp_k,mu_t)
            λ = -2/3*mu
            S11 = du_dx; S22 = dv_dy;
            S12 = 0.5 * (du_dy + dv_dx)
            div_u = S11 + S22
            τ_xx = 2*mu*du_dx + λ*div_u
            τ_yy = 2*mu*dv_dy + λ*div_u
            τ_xy = 2*mu*S12
            qx = -k * dT_dx
            qy = -k * dT_dy
            jx =  - ρ * D_k * dY_dx
            jy =  - ρ * D_k * dY_dy
            ex = jnp.sum(jx*h_k,axis=0,keepdims=True)
            ey = jnp.sum(jy*h_k,axis=0,keepdims=True)
            Fv = jnp.concatenate([jnp.zeros_like(ρ),
                                 τ_xx, τ_xy,
                                 u*τ_xx + v*τ_xy - qx - ex, -jx[0:-1]], axis=0)
            Gv = jnp.concatenate([jnp.zeros_like(ρ),
                                 τ_xy, τ_yy,
                                 u*τ_xy + v*τ_yy - qy - ey, -jy[0:-1]], axis=0)
            F_hat = J * (dξ_dx*Fv + dξ_dy*Gv)
            G_hat = J * (dη_dx*Fv + dη_dy*Gv)
            return F_hat, G_hat
        def viscous_flux(U,aux,metrics):
            F_hat, G_hat = viscous_flux_node(U, aux, metrics, d_dx, d_dy)
            F_interface = reconstruction_x(F_hat)
            G_interface = reconstruction_y(G_hat)
            dF = F_interface[:,1:,:]-F_interface[:,:-1,:]
            dG = G_interface[:,:,1:]-G_interface[:,:,:-1]
            net_flux = (dF + dG)/metrics['J']
            return net_flux
        def total_flux(U,aux,metrics):
            total_flux = advective_flux(U,aux,metrics) + viscous_flux(U,aux,metrics)
            return total_flux
    else:
        total_flux = advective_flux
    
    return total_flux











