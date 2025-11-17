import wandb
from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    """
    Simple wrapper to perform logging in W&B.
    """

    def __init__(self, cfg: DictConfig, section: str = "eval", project: str | None = None):
        """
        Initialize the wandb session, if logging is on in the config.
        """
        self.cfg = cfg
        self.section = section
        self.section_cfg = getattr(cfg, section, None)
        self.is_active = bool(self.section_cfg and getattr(self.section_cfg, "wandb_log", False))
        self.run = None

        if self.is_active:
            wandb_entity = "b-sharipov-innopolis-university"
            if project is None:
                project = "chart-vqa-evaluation" if section == "eval" else "chart-vqa-training"

            self.run = wandb.init(
                entity=wandb_entity,
                project=project,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            print(f"Setting up W&B logging for '{section}', Run ID: {self.run.id}")

    def log(self, data: dict):
        """
        Logs the metrics if session is active.
        """
        if self.is_active and self.run:
            self.run.log(data)

    def log_artifact(self, name: str, path, type_name: str = "model", metadata: dict | None = None):
        """
        Logs a local file as a W&B artifact, if logging is active.
        """
        if not (self.is_active and self.run):
            return

        artifact = wandb.Artifact(name=name, type=type_name, metadata=metadata)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)
    
    def load_artifact(self, artifact_path: str, type_name: str = "model") -> str:
        """
        Loads an artifact from W&B and returns the local path to the downloaded artifact.
        """
        if not (self.is_active and self.run):
            raise RuntimeError("W&B logging is not active; cannot load artifact.")

        api = wandb.Api()
        artifact = api.artifact(artifact_path, type=type_name)
        artifact_dir = artifact.download()
        return artifact_dir

    def finish(self):
        """
        Finish the session if it's active.
        """
        if self.is_active and self.run:
            self.run.finish()
