def run_residual_diagnostics(*args, **kwargs):
    from .residual_diagnostics import run_residual_diagnostics as _run_residual_diagnostics

    return _run_residual_diagnostics(*args, **kwargs)


def run_phase_pipeline(*args, **kwargs):
    from .server_phase_runner import run_phase_pipeline as _run_phase_pipeline

    return _run_phase_pipeline(*args, **kwargs)
