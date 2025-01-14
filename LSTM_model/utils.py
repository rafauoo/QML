from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from LSTM_model.config import LSTM_config as lstm_conf


def get_trainer() -> Trainer:
    """Inits logger, checkpoint callback and returns the trainer.

    :return: configured trainer object
    :rtype: Trainer
    """
    try:
        logger = TensorBoardLogger("LSTM_model/lightning_logs", name="authenticity")

        checkpoint_callback = ModelCheckpoint(
            dirpath="LSTM_model/checkpoints",
            filename="best-checkpoint",
            save_top_k=1,  # only the best model
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        progress_bar = RichProgressBar()

        trainer = Trainer(
            logger=logger,
            callbacks=[
                checkpoint_callback,
                progress_bar,
            ],
            max_epochs=lstm_conf.num_epochs,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=5,
        )
        return trainer
    except Exception as e:
        print(f"Error in get_trainer: {e}")
        raise
