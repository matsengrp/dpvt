import torch
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def custom_collate(items):
    """
    Args:
        items is a list of (input, output) pairs, where `input` is an ete3.Tree and
        `output` is a float
    """
    return [item[0] for item in items], torch.tensor([item[1] for item in items])


class Wrap:
    def __init__(
        self,
        train_data,
        val_data,
        model,
        log_path,
        batch_size=1024,
        epochs=200,
    ):
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, collate_fn=custom_collate
        )
        self.val_loader = DataLoader(
            val_data, batch_size=batch_size, collate_fn=custom_collate
        )
        self.model = model  # currently TraverseNN
        self.log_path = log_path
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, final_checkpoint):
        # use pytorch lightning
        logger = TensorBoardLogger("lightning_logs", name=self.log_path)
        checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1)
        trainer = L.Trainer(
            logger=logger,
            max_epochs=self.epochs,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(self.model, self.train_loader, self.val_loader)
        trainer.save_checkpoint(final_checkpoint)
