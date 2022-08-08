from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch


class Save(pl.Callback):
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y-%H-%M")
        dir_name = Path("/srv/galene0/ot44/checkpoints/dfrdd/")
        dir_name.mkdir(exist_ok=True)
        file_name = f"{date_time}-{trainer.current_epoch}-{trainer.global_step}.pt"
        torch.save(pl_module.state_dict(), dir_name / file_name)
