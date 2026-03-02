import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from mentioned.model import LitMentionDetector, ModelRegistry
from mentions.data import DataRegistry


def train(
    model_factory: str = "model_v1",
    data_factory: str = "litbank_mentions",
    repo_id: str = "kadarakos/mention-detector-poc-dry-run",
    project_name: str = "mention-detector-poc",
    encoder_id: str = "distilroberta-base",
    patience: int = 5,
    val_interval: int = 1000,
):
    data = DataRegistry.get(data_factory)()
    model = ModelRegistry.get(model_factory)(data, encoder_id)
    wandb_logger = WandbLogger(
        project=project_name,
        name=encoder_id,
    )
    # Save only the best model for the PoC purposes.
    best_checkpoint = ModelCheckpoint(
        monitor="val_f1_mention",
        mode="max",
        save_top_k=1,
        filename="best-mention-f1",
        verbose=True,
    )
    early_stopper = EarlyStopping(
        monitor="val_f1_mention",
        min_delta=0.01,
        patience=patience,
        verbose=True,
        mode="max",
    )
    trainer = Trainer(
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        callbacks=[early_stopper, best_checkpoint],
        logger=wandb_logger,
    )
    print("Starting Trainer.")
    trainer.fit(
        model=model,
        train_dataloaders=data.train_loader,
        val_dataloaders=data.val_loader,
    )
    trainer.test(dataloaders=data.test_loader, ckpt_path="best", weights_only=False)
    print(f"Pushing best model to: {repo_id}")
    fresh_model = ModelRegistry.get(model_factory)(data, encoder_id)
    best_model = LitMentionDetector.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        tokenizer=fresh_model.tokenizer,
        encoder=fresh_model.encoder,
        mention_detector=fresh_model.mention_detector,
        weights_only=False,
    )
    best_model.push_to_hub(repo_id, private=True)
    wandb.finish()

    print("Testing pulling and evaluating the model.")
    fresh_model = ModelRegistry.get(model_factory)(data, encoder_id)
    remote_model = LitMentionDetector.from_pretrained(
        repo_id,
        tokenizer=fresh_model.tokenizer,
        encoder=fresh_model.encoder,
        mention_detector=fresh_model.mention_detector,
    )
    verify_trainer = Trainer(accelerator="auto", logger=False)
    verify_trainer.test(model=remote_model, dataloaders=data.test_loader)
