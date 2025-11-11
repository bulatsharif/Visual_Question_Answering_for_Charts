import wandb
from omegaconf import DictConfig, OmegaConf

class WandbLogger:
    """
    Simple wrapper to perform logging in W&B.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the wandb session, if logging is on in the config.
        """
        self.cfg = cfg
        self.is_active = getattr(cfg.eval, "wandb_log", False)
        self.run = None

        if self.is_active:
            wandb_entity = "b-sharipov-innopolis-university"
            wandb_project = "chart-vqa-evaluation"

            self.run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            print(f"Setting up W&B logging, Run ID: {self.run.id}")

    def log(self, data: dict):
        """
        Logs the metrics if session is active.
        """
        if self.is_active and self.run:
            self.run.log(data)

    def finish(self):
        """
        Finish the session if it's active.
        """
        if self.is_active and self.run:
            self.run.finish()