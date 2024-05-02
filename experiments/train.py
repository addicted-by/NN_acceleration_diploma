import hydra
from omegaconf import DictConfig
import utils


@utils.register_custom_resolvers(**utils._HYDRA_PARAMS)
@hydra.main(**utils._HYDRA_PARAMS)
def main(cfg: DictConfig):
    utils.print_config_tree(cfg)