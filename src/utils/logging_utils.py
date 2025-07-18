from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["policy"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # hparams["databuffer"] = cfg["databuffer"]
    # hparams["modelbuffer"] = cfg["modelbuffer"]
    hparams["env"] = cfg["env"]
    hparams["policy"] = cfg["policy"]
    hparams["algo"] = cfg["rl_model"] #d 
    hparams["trainer"] = cfg["trainer"] #d
    hparams["dynamics_model"] = cfg['model']
    hparams["callbacks"] = cfg.get("callbacks") #d
    # hparams["extras"] = cfg.get("extras")
    # hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags") 
    hparams["ckpt_path"] = cfg.get("ckpt_path") #d
    hparams["seed"] = cfg.get("seed") #d

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
