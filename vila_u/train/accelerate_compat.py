import inspect
from functools import wraps


def patch_accelerator_init_for_old_versions() -> None:
    try:
        import accelerate
    except ModuleNotFoundError:
        return

    accelerator_cls = accelerate.Accelerator
    if getattr(accelerator_cls, "_vila_u_patched_init", False):
        return

    original_init = accelerator_cls.__init__
    accepted_params = set(inspect.signature(original_init).parameters.keys())

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
        return original_init(self, *args, **filtered_kwargs)

    accelerator_cls.__init__ = wrapped_init
    accelerator_cls._vila_u_patched_init = True
