from time import time
from pathlib import Path
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter





def preproc_data():
    from data import split_data
    split_data('../data/hin-eng/hin.txt', '../data/hin-eng')


from data import IndicDataset, PadSequence
import model as M


def gen_model_loaders(config):
    model, tokenizers = M.build_model(config)

    pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True),
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False),
                           batch_size=config.eval_size,
                           shuffle=False,
                           collate_fn=pad_sequence)
    return model, tokenizers, train_loader, eval_loader


import numpy as np
from train_util import run_train
from config import replace, preEnc, preEncDec
import pytorch_lightning as pl
from pytorch_lightning import Trainer

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    #print (f'preds: {pred_flat}')
    #print (f'labels: {labels_flat}')

    return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)

class MyLightninModule(pl.LightningModule):
    def __init__(self):
        super(MyLightninModule, self).__init__()
        self.rconf=preEncDec
        self.model = gen_model_loaders(self.rconf)[0]
        #self.criterion = get_criterion()

    def forward(self, x,y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        #y_hat = self.forward(x)
        loss = self.forward(x,y)[0]
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x,y)[1]
        logits = y_hat.detach().cpu().numpy()
        label_ids = y.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        return {'val_loss': self.forward(x,y)[0], 'correct': tmp_eval_accuracy}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([torch.tensor(x['correct']) for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'avg_val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.rconf.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=self.rconf.lr)
        return [optimizer], [scheduler]

    #@pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return gen_model_loaders(self.rconf)[2]

    #@pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return gen_model_loaders(self.rconf)[3]


def main():
    epochs = 10
    #num_class = 10
    rconf = preEncDec
    output_path = '../output/lightning'
    tokenizers= gen_model_loaders(rconf)[1]


    model = MyLightninModule()

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=epochs,
        default_save_path=output_path,

        # use_amp=False,
    )
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")
    #model.save(tokenizers, rconf.model_output_dirs)


if __name__ == '__main__':
    preproc_data()
    main()
