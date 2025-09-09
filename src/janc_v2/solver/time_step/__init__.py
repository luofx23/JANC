from . import RK2,TVD_RK3

time_step_dict_flux_splitting = {
    "RK2": RK2.advance_one_step_flux_splitting,
    "TVD_RK3": TVD_RK3.advance_one_step_splitting
}

time_step_dict_flux_godunov = {
    "RK2": RK2.advance_one_step_godunov,
    "TVD_RK3": TVD_RK3.advance_one_step_godunov
}

