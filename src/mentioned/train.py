import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from mentioned.model import LitMentionDetector, ModelRegistry
from mentioned.data import DataRegistry


def train(
    model_factory: str = "model_v1",
    data_factory: str = "litbank_mentions",
    repo_id: str = "kadarakos/mention-detector-poc-dry-run",
    project_name: str = "mention-detector-poc",
    encoder_id: str = "distilroberta-base",
    patience: int = 5,
    val_interval: int = 1000,
    stop_criterion: str = "val_f1_mention",
    max_epochs: int | None = None,
):
    if max_epochs is None:
        max_epochs = 1000
    data = DataRegistry.get(data_factory)()
    model = ModelRegistry.get(model_factory)(data, encoder_id)
    wandb_logger = WandbLogger(
        project=project_name,
        name=encoder_id,
    )
    best_checkpoint = ModelCheckpoint(
        monitor=stop_criterion,
        mode="max",
        save_top_k=1,
        filename=f"best-{stop_criterion}",
        verbose=True,
    )
    early_stopper = EarlyStopping(
        monitor=stop_criterion,
        min_delta=0.01,
        patience=patience,
        verbose=True,
        mode="max",
    )
    trainer = Trainer(
        max_epochs=max_epochs,      # Now configurable
        val_check_interval=val_interval,
        callbacks=[early_stopper, best_checkpoint],
        logger=wandb_logger,
        accelerator="auto",
    )
    print(f"Starting Trainer for {max_epochs} epochs.")
    trainer.fit(
        model=model,
        train_dataloaders=data.train_loader,
        val_dataloaders=data.val_loader,
    )
    trainer.test(dataloaders=data.test_loader, ckpt_path="best", weights_only=False)
    print(f"Pushing best model to: {repo_id}")
    fresh_bundle = ModelRegistry.get(model_factory)(data, encoder_id)
    labeler = getattr(fresh_bundle, "mention_labeler", None)
    l2id = getattr(fresh_bundle, "label2id", None)

    best_model = LitMentionDetector.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        tokenizer=fresh_bundle.tokenizer,
        encoder=fresh_bundle.encoder,
        mention_detector=fresh_bundle.mention_detector,
        label2id=l2id,
        mention_labeler=labeler,
        weights_only=False,
    )
    best_model.push_to_hub(repo_id, private=True)
    wandb.finish()

    print("Verifying Hub upload by pulling and re-evaluating...")
    remote_model = LitMentionDetector.from_pretrained(
        repo_id,
        tokenizer=fresh_bundle.tokenizer,
        encoder=fresh_bundle.encoder,
        mention_detector=fresh_bundle.mention_detector,
        label2id=l2id,
        mention_labeler=labeler,
    )

    verify_trainer = Trainer(accelerator="auto", logger=False)
    verify_trainer.test(model=remote_model, dataloaders=data.test_loader)
