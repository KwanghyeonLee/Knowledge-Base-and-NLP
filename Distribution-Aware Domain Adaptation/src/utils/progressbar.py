from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar


class CustomProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__()

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        **kwargs
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return

        assert self.progress is not None

        if trainer.sanity_checking:
            if self.val_sanity_progress_bar_id is not None:
                self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)

            self.val_sanity_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                self.sanity_check_description,
                visible=False,
            )
        else:
            if self.val_progress_bar_id is not None:
                self.progress.update(self.val_progress_bar_id, advance=0, visible=False)

            self.val_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                f"DataLoader {dataloader_idx}: {self.validation_description}",
                visible=False,
            )

        self.refresh()

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return

        if self.test_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.test_progress_bar_id, advance=0, visible=False)
        self.test_progress_bar_id = self._add_task(
            self.total_test_batches_current_dataloader,
            f"DataLoader {dataloader_idx}: {self.test_description}",
        )
        self.refresh()

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return

        if self.predict_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.predict_progress_bar_id, advance=0, visible=False)
        self.predict_progress_bar_id = self._add_task(
            self.total_predict_batches_current_dataloader,
            f"DataLoader {dataloader_idx}: {self.predict_description}",
        )
        self.refresh()
