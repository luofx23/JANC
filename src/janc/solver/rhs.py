from . import flux, aux_func
from ..model import thermo_model,reaction_model
from ..boundary import boundary

point_implicit = 'off'
def set_rhs(thermo_config,reaction_config,flux_config,transport_config,boundary_config,source_config=None):
    global point_implicit
    thermo_model.set_thermo(thermo_config)
    reaction_source_terms = reaction_model.set_reaction(reaction_config)
    flux_func = flux.set_flux_solver(flux_config,transport_config)
    boundary.set_boundary(boundary_config)
    aux_func.set_source_terms(source_config)
    if reaction_config['is_detailed_chemistry']:
        point_implicit = 'on'
        def rhs(U,aux,metrics,dt,theta):
            U_with_ghost,aux_with_ghost = boundary.boundary_conditions(U,aux,metrics,theta)
            rhs = dt*(flux_func(U_with_ghost,aux_with_ghost,metrics) + aux_func.user_source(U, aux, theta))
            return rhs
    else:
        def rhs(U,aux,metrics,dt,theta):
            U_with_ghost,aux_with_ghost = boundary.boundary_conditions(U,aux,metrics,theta)
            rhs = dt*(flux_func(U_with_ghost,aux_with_ghost,metrics) + aux_func.user_source(U,aux,theta)) + reaction_source_terms(U,aux,dt,theta)
            return rhs
    return rhs
    





