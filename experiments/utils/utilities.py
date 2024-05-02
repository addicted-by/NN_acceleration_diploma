import argparse
from typing import Callable, Optional, ParamSpec, TypeVar, overload
from functools import wraps
import os
import warnings
from functools import wraps
from typing import Any, Callable, List, Optional
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from catalyst.core import Callback
from . import pylogger, rich_utils


log = pylogger.get_pylogger(__name__)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "config.yaml",
}

def _get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None

T = TypeVar("T")
P = ParamSpec("P")

@overload
def rank_zero_only(
    fn: Callable[P, T]
) -> Callable[P, Optional[T]]: 
    ...

@overload
def rank_zero_only(
    fn: Callable[P, T], 
    default: T
) -> Callable[P, T]:
    ...

def rank_zero_only(
    fn: Callable[P, T], 
    default: Optional[T] = None
) -> Callable[P, Optional[T]]:
    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", _get_rank() or 0)
        print(rank)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn



def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg (DictConfig): Main config.
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info(
            "Disabling python warnings! <cfg.extras.ignore_warnings=True>"
        )
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info(
            "Printing config tree with Rich! <cfg.extras.print_config=True>"
        )
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        List[Callback]: List with all instantiated callbacks.
    """

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks



@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup).

    Args:
        path (str): File path.
        content (str): File content.
    """

    with open(path, "w+") as file:
        file.write(content)


def get_args_parser() -> argparse.ArgumentParser:
    """Get parser for additional Hydra's command line flags."""
    parser = argparse.ArgumentParser(
        description="Additional Hydra's command line flags parser."
    )

    parser.add_argument(
        "--config-path",
        "-cp",
        nargs="?",
        default=None,
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        nargs="?",
        default=None,
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        nargs="?",
        default=None,
        help="Adds an additional config dir to the config search path",
    )
    return parser


def register_custom_resolvers(
    version_base: str, config_path: str, config_name: str
) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is
    excepted to call before `hydra.main` decorator call.

    Replace resolver: To avoiding copying of loss and metric names in configs,
    there is custom resolver during hydra initialization which replaces
    `__loss__` to `loss.__class__.__name__` and `__metric__` to
    `main_metric.__class__.__name__` For example: ${replace:"__metric__/valid"}
    Use quotes for defining internal value in ${replace:"..."} to avoid grammar
    problems with hydra config parser.

    Args:
        version_base (str): Hydra version base.
        config_path (str): Hydra config path.
        config_name (str): Hydra config name.

    Returns:
        Callable: Decorator that registers custom resolvers before running
            main function.
    """

    # parse additional Hydra's command line flags
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    if args.config_path:
        config_path = args.config_path
    if args.config_dir:
        config_path = args.config_dir
    if args.config_name:
        config_name = args.config_name

    # register of replace resolver
    if not OmegaConf.has_resolver("replace"):
        with initialize_config_dir(
            version_base=version_base, config_dir=config_path
        ):
            cfg = compose(
                config_name=config_name, return_hydra_config=True, overrides=[]
            )
        GlobalHydra.instance().clear()

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapper

    return decorator