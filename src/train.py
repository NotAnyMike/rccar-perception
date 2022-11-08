import pytorch_lightning as pl
import torch.nn.functional as F

from models import get_unet, get_resnet50, build_3d_cnn_pytorch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='data')
    parser.add_argument('--img-shape', type=int, nargs='+', default=[224, 224])

    return parser.parse_args()


class MainModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # MSE loss
        loss = F.mse_loss(y_hat, y)

        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(folder, img_shape):
    # Create train_dataset and val_dataset
    in_dims = [3,] + img_shape

    datamodule = YoloDataModule(folder, img_shape)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    arch = get_resnet50(in_dims, 2)
    model = MainModel(arch)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    train(folder=args.folder, img_shape=args.img_shape)
