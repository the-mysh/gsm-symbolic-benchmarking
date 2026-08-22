def make_bootstrap_names(n_boot: int, glmm_id: str):
    ext = "pkl"
    run_info = f"n_boot_{n_boot}__glmm{glmm_id}"
    return f"{run_info}.{ext}", f"{run_info}__wald.{ext}", f"{run_info}__checkpoints.{ext}"
