from . import RK2,TVD_RK3

time_step_dict = {
    "RK2_flux_splitting": RK2.advance_one_step_flux_splitting,
    "RK2_godunov": RK2.advance_one_step_godunov,
    "RK2_flux_splitting_amr": RK2.advance_one_step_flux_splitting_amr,
    "RK2_godunov_amr": RK2.advance_one_step_godunov_amr,
    "TVD_RK3_flux_splitting": TVD_RK3.advance_one_step_flux_splitting,
    "TVD_RK3_godunov": TVD_RK3.advance_one_step_godunov,
    "TVD_RK3_flux_splitting_amr": TVD_RK3.advance_one_step_flux_splitting_amr,
    "TVD_RK3_godunov_amr": TVD_RK3.advance_one_step_godunov_amr,
}


