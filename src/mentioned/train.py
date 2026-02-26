import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from mentioned.model import make_model_v1, LitMentionDetector
from mentions.data import make_litbank


def train():
    train_loader, val_loader, test_loader = make_litbank()
    model = make_model_v1()
    wandb_logger = WandbLogger(
        project="mention-detector-poc",
        name="distilroberta-frozen-encoder",
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
        patience=5,
        verbose=True,
        mode="max",
    )
    trainer = Trainer(
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[early_stopper, best_checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.test(dataloaders=test_loader, ckpt_path="best", weights_only=False)
    fresh_model = make_model_v1()
    best_model = LitMentionDetector.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        tokenizer=fresh_model.tokenizer,
        encoder=fresh_model.encoder,
        mention_detector=fresh_model.mention_detector,
        weights_only=False,
    )
    best_model.push_to_hub("kadarakos/mention-detector-poc-dry-run", private=True)
    wandb.finish()

    ### Test pull:
    fresh_model = make_model_v1()
    repo_id = "kadarakos/mention-detector-poc-dry-run"
    remote_model = LitMentionDetector.from_pretrained(
        repo_id,
        tokenizer=fresh_model.tokenizer,
        encoder=fresh_model.encoder,
        mention_detector=fresh_model.mention_detector,
    )

    # 3. Final Verification
    verify_trainer = Trainer(accelerator="auto", logger=False)
    verify_trainer.test(model=remote_model, dataloaders=test_loader)
