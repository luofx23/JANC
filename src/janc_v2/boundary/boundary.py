import jax
import jax.numpy as jnp
from .boundary_padding import pad_1D,pad_2D,pad_3D,\
                        replace_lb_1D, replace_rb_1D, \
                        replace_lb_2D, replace_rb_2D, replace_ub_2D, replace_bb_2D,\
                        replace_lb_3D, replace_rb_3D, replace_ub_3D, replace_bb_3D,replace_fb_3D, replace_backb_3D
from .boundary_conditions_1D import left_bd_dict_1D,right_bd_dict_1D
from .boundary_conditions_2D import left_bd_dict_2D,right_bd_dict_2D,bottom_bd_dict_2D,top_bd_dict_2D
from .boundary_conditions_3D import left_bd_dict_3D,right_bd_dict_3D,bottom_bd_dict_3D,top_bd_dict_3D,front_bd_dict_3D,back_bd_dict_3D



boundary_func = {'left_boundary':None,
                 'right_boundary':None,
                 'top_boundary':None,
                 'bottom_boundary':None}

def set_boundary(boundary_config:dict,dim:str):
    global  boundary_func
    if dim == '1D':
        replace_lb = replace_lb_1D
        replace_rb = replace_rb_1D
        left_bd_dict = left_bd_dict_1D
        right_bd_dict = right_bd_dict_1D
        if callable(boundary_config['left_boundary']):
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                U_lb,aux_lb = padded_U[:,3:6],padded_aux[:,3:6]
                U_lb,aux_lb = boundary_config['left_boundary'](U_lb,aux_lb,metrics,theta)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
        elif boundary_config['left_boundary']=='periodic':
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
             def left_boundary(padded_U,padded_aux,metrics,theta=None):
                 U_lb,aux_lb = padded_U[:,3:6],padded_aux[:,3:6]
                 U_lb,aux_lb = left_bd_dict[boundary_config['left_boundary']](U_lb,aux_lb,metrics,theta)
                 U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                 return U_with_lb,aux_with_lb   
        boundary_func['left_boundary'] = left_boundary
        
        
        if callable(boundary_config['right_boundary']):
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3],padded_aux[:,-6:-3]
                U_rb,aux_rb = boundary_config['right_boundary'](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
        elif boundary_config['right_boundary']=='periodic':
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3],padded_aux[:,-6:-3]
                U_rb,aux_rb = right_bd_dict[boundary_config['right_boundary']](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb        
        boundary_func['right_boundary'] = right_boundary 
    
    
    
    if dim == '2D':
        replace_lb = replace_lb_2D
        replace_rb = replace_rb_2D
        replace_bb = replace_bb_2D
        replace_ub = replace_ub_2D
        left_bd_dict = left_bd_dict_2D
        right_bd_dict = right_bd_dict_2D
        bottom_bd_dict = bottom_bd_dict_2D
        top_bd_dict = top_bd_dict_2D
        if callable(boundary_config['left_boundary']):
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
                U_lb,aux_lb = boundary_config['left_boundary'](U_lb,aux_lb,metrics,theta)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
        elif boundary_config['left_boundary']=='periodic':
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
             def left_boundary(padded_U,padded_aux,metrics,theta=None):
                 U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
                 U_lb,aux_lb = left_bd_dict[boundary_config['left_boundary']](U_lb,aux_lb,metrics,theta)
                 U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                 return U_with_lb,aux_with_lb   
        boundary_func['left_boundary'] = left_boundary
        
        
        if callable(boundary_config['right_boundary']):
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
                U_rb,aux_rb = boundary_config['right_boundary'](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
        elif boundary_config['right_boundary']=='periodic':
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
                U_rb,aux_rb = right_bd_dict[boundary_config['right_boundary']](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb        
        boundary_func['right_boundary'] = right_boundary    
        
        
        if callable(boundary_config['bottom_boundary']):
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
                U_bb,aux_bb = boundary_config['bottom_boundary'](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        elif boundary_config['bottom_boundary']=='periodic':
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
                U_bb,aux_bb = bottom_bd_dict[boundary_config['bottom_boundary']](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        boundary_func['bottom_boundary'] = bottom_boundary
        
        
        if callable(boundary_config['top_boundary']):
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
                U_ub,aux_ub = boundary_config['top_boundary'](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        elif boundary_config['top_boundary'] == 'periodic':
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
                U_ub,aux_ub = top_bd_dict[boundary_config['top_boundary']](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        boundary_func['top_boundary'] = top_boundary
        
    if dim == '3D':
        replace_lb = replace_lb_3D
        replace_rb = replace_rb_3D
        replace_bb = replace_bb_3D
        replace_ub = replace_ub_3D
        replace_fb = replace_fb_3D
        replace_backb = replace_backb_3D
        left_bd_dict = left_bd_dict_3D
        right_bd_dict = right_bd_dict_3D
        bottom_bd_dict = bottom_bd_dict_3D
        top_bd_dict = top_bd_dict_3D
        front_bd_dict = front_bd_dict_3D
        back_bd_dict = back_bd_dict_3D
        if callable(boundary_config['left_boundary']):
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                U_lb,aux_lb = padded_U[:,3:6,3:-3,3:-3],padded_aux[:,3:6,3:-3,3:-3]
                U_lb,aux_lb = boundary_config['left_boundary'](U_lb,aux_lb,metrics,theta)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
        elif boundary_config['left_boundary']=='periodic':
            def left_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
             def left_boundary(padded_U,padded_aux,metrics,theta=None):
                 U_lb,aux_lb = padded_U[:,3:6,3:-3,3:-3],padded_aux[:,3:6,3:-3,3:-3]
                 U_lb,aux_lb = left_bd_dict[boundary_config['left_boundary']](U_lb,aux_lb,metrics,theta)
                 U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                 return U_with_lb,aux_with_lb   
        boundary_func['left_boundary'] = left_boundary
        
        
        if callable(boundary_config['right_boundary']):
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3,3:-3],padded_aux[:,-6:-3,3:-3,3:-3]
                U_rb,aux_rb = boundary_config['right_boundary'](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
        elif boundary_config['right_boundary']=='periodic':
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def right_boundary(padded_U,padded_aux,metrics,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3,3:-3],padded_aux[:,-6:-3,3:-3,3:-3]
                U_rb,aux_rb = right_bd_dict[boundary_config['right_boundary']](U_rb,aux_rb,metrics,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb        
        boundary_func['right_boundary'] = right_boundary    
            
            
        if callable(boundary_config['bottom_boundary']):
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6,3:-3],padded_aux[:,3:-3,3:6,3:-3]
                U_bb,aux_bb = boundary_config['bottom_boundary'](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        elif boundary_config['bottom_boundary']=='periodic':
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def bottom_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6,3:-3],padded_aux[:,3:-3,3:6,3:-3]
                U_bb,aux_bb = bottom_bd_dict[boundary_config['bottom_boundary']](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        boundary_func['bottom_boundary'] = bottom_boundary
        
        
        if callable(boundary_config['top_boundary']):
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3,3:-3],padded_aux[:,3:-3,-6:-3,3:-3]
                U_ub,aux_ub = boundary_config['top_boundary'](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        elif boundary_config['top_boundary'] == 'periodic':
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def top_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3,3:-3],padded_aux[:,3:-3,-6:-3,3:-3]
                U_ub,aux_ub = top_bd_dict[boundary_config['top_boundary']](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        boundary_func['top_boundary'] = top_boundary
        
        if callable(boundary_config['front_boundary']):
            def front_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:-3,3:6],padded_aux[:,3:-3,3:-3,3:6]
                U_bb,aux_bb = boundary_config['front_boundary'](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_fb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        elif boundary_config['front_boundary']=='periodic':
            def front_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def front_boundary(padded_U,padded_aux,metrics,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:-3,3:6],padded_aux[:,3:-3,3:-3,3:6]
                U_bb,aux_bb = front_bd_dict[boundary_config['front_boundary']](U_bb,aux_bb,metrics,theta)
                U_with_bb,aux_with_bb = replace_fb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        boundary_func['front_boundary'] = front_boundary
        
        
        if callable(boundary_config['back_boundary']):
            def back_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,3:-3,-6:-3],padded_aux[:,3:-3,3:-3,-6:-3]
                U_ub,aux_ub = boundary_config['back_boundary'](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_backb(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        elif boundary_config['back_boundary'] == 'periodic':
            def back_boundary(padded_U,padded_aux,metrics,theta=None):
                return padded_U,padded_aux
        else:
            def back_boundary(padded_U,padded_aux,metrics,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,3:-3,-6:-3],padded_aux[:,3:-3,3:-3,-6:-3]
                U_ub,aux_ub = back_bd_dict[boundary_config['back_boundary']](U_ub,aux_ub,metrics,theta)
                U_with_ub,aux_with_ub = replace_backb(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        boundary_func['back_boundary'] = back_boundary

def boundary_conditions_1D(U, aux, metrics, theta=None):
    U_periodic_pad,aux_periodic_pad = pad_1D(U,aux)
    U_with_lb,aux_with_lb = boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad, metrics, theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary_func['right_boundary'](U_with_lb,aux_with_lb,metrics,theta)
    return U_with_ghost_cell,aux_with_ghost_cell

def boundary_conditions_2D(U, aux, metrics, theta=None):
    U_periodic_pad,aux_periodic_pad = pad_2D(U,aux)
    U_with_lb,aux_with_lb = boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad, metrics, theta)
    U_with_rb,aux_with_rb = boundary_func['right_boundary'](U_with_lb,aux_with_lb,metrics,theta)
    U_with_bb,aux_with_bb = boundary_func['bottom_boundary'](U_with_rb,aux_with_rb,metrics,theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary_func['top_boundary'](U_with_bb,aux_with_bb,metrics,theta)
    return U_with_ghost_cell,aux_with_ghost_cell

def boundary_conditions_3D(U, aux, metrics, theta=None):
    U_periodic_pad,aux_periodic_pad = pad_3D(U,aux)
    U_with_lb,aux_with_lb = boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad, metrics, theta)
    U_with_rb,aux_with_rb = boundary_func['right_boundary'](U_with_lb,aux_with_lb,metrics,theta)
    U_with_bb,aux_with_bb = boundary_func['bottom_boundary'](U_with_rb,aux_with_rb,metrics,theta)
    U_with_ub,aux_with_ub = boundary_func['top_boundary'](U_with_bb,aux_with_bb,metrics,theta)
    U_with_fb,aux_with_fb = boundary_func['front_boundary'](U_with_ub,aux_with_ub,metrics,theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary_func['back_boundary'](U_with_fb,aux_with_fb,metrics,theta)
    return U_with_ghost_cell,aux_with_ghost_cell


