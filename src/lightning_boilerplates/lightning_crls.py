import logging
import os
from argparse import Namespace

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision.datasets import DatasetFolder
from pytorch_lightning.core import LightningModule
from tqdm import tqdm

import utils.helpers as H
from data.lidc import LIDCNodulesDataset
from models.rls import RLSModule, init_levelset


log = logging.getLogger("lightning_boilerplates.crls")


class CRLSModel(LightningModule):
    def __init__(self, config: Namespace):
        super().__init__()
        self.metaconf = config.metaconf
        self.hparams = Namespace(**config.hyperparams)
        self.dataset_params = Namespace(**config.dataloaders["train"])

        inp_image_size = [self.dataset_params.params["cube_voxelsize"]] * 2
        self.rls_model = RLSModule(inp_image_size)

    def forward(self, input, hidden):
        return self.rls_model(input, hidden)

    def configure_optimizers(self):
        lr = self.hparams.lr
        alpha = self.hparams.alpha
        return torch.optim.RMSprop(self.rls_model.parameters(), lr=lr, alpha=alpha)

    def prepare_data(self):
        """Prepare and save dataset as TensorDataset to improve training speed.
        """

        self.generic_dataset = LIDCNodulesDataset(**self.dataset_params.params)
        log.info(f"DATASET SIZE: {len(self.generic_dataset)}")

        tensor_dataset_path = os.path.join(
            self.metaconf["ws_path"], "tensor_datasets", self.dataset_params.tensor_dataset_name
        )

        # compare configs, if not same, refresh dataset
        current_config_snapshot_exists = H.config_snapshot(
            "dataset_params", self.dataset_params.params, "src/data/aux/.dataset_config_snapshot.json",
        )
        if not current_config_snapshot_exists:
            H.makedirs(tensor_dataset_path)
            _tqdm_kwargs = {"desc": "Preparing TensorDataset", "total": len(self.generic_dataset)}
            for i, sample in tqdm(enumerate(self.generic_dataset), **_tqdm_kwargs):
                f_folder_path = os.path.join(tensor_dataset_path, "0")
                H.makedirs(f_folder_path)
                f_path = os.path.join(tensor_dataset_path, "0", f"nodule_{i}.pt")
                save_nodules = {"nodule": sample["nodule"], "mask": sample["mask"]}
                torch.save(save_nodules, f_path)

        self.dataset = DatasetFolder(tensor_dataset_path, torch.load, ("pt"))
        self.dataset.norm = self.generic_dataset.norm

        train_inds, val_inds, test_inds = H.train_val_holdout_split(self.dataset)
        self.train_sampler = SubsetRandomSampler(train_inds)
        self.val_sampler = SubsetRandomSampler(val_inds)
        self.test_subset = Subset(self.dataset, test_inds)

    def train_dataloader(self):
        dl = DataLoader(
            self.dataset,
            sampler=self.train_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )
        return dl

    # def val_dataloader(self):
    #     dl = DataLoader(
    #         self.dataset,
    #         self.val_sampler,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=4,
    #         shuffle=True,
    #     )
    #     return dl

    # def test_dataloader(self):
    #     dl = DataLoader(self.test_subset)
    #     return dl

    def loss_f(self, y_hat, y):
        return F.nll_loss(F.log_softmax(y_hat, 1), y.squeeze().long())

    def training_step(self, batch, batch_idx):
        nodules, masks = batch[0]["nodule"], batch[0]["mask"]
        nodules, masks = (
            nodules[:, :, nodules.size(2) // 2, :, :],
            masks[:, :, masks.size(2) // 2, :, :]
        )

        hiddens = init_levelset(nodules.shape[-2:]).repeat(nodules.size(0), 1, 1, 1)
        for t in range(self.hparams.num_T):
            outputs, hiddens = self.forward(nodules, hiddens)

        loss = self.loss_f(outputs, masks)

        tqdm_dict = {"loss": loss}
        output = {
            "imgs": nodules,
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }
        return output

    def training_step_end(self, output):
        if self.global_step % 20 == 0:
            imgs = output["imgs"]
            imgs_in_hu = self.dataset.norm.denorm(imgs)
            grid = torchvision.utils.make_grid(imgs_in_hu, 4, normalize=True)
            self.logger.experiment.add_image(f"input_images", grid, self.global_step)
        del output["imgs"]
        return output

    def on_epoch_end(self):
        pass
        # sample_imgs_in_hu = self.dataset.norm.denorm(self.forward(z))
        # grid = torchvision.utils.make_grid(sample_imgs_in_hu, 4, normalize=True)
        # self.logger.experiment.add_image(f"generated_images", grid, self.current_epoch)
