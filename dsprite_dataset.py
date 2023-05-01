from typing import List, Optional, Sequence, Union
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import sys
import os
import urllib.request
from pytorch_lightning import LightningDataModule

class DspritesDataset(Dataset):
    def __init__(
        self,
        dataset_url="https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        val_set=False
    ):
        data_dir = "./Data/dsprite/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(dataset_url, filename=os.path.join(data_dir, "data.npz"))
        for file in glob.glob(f"{data_dir}/*.npz"):
            # Look at the first npz file and break
            data = np.load(file)
            self.image_ndarray = data["imgs"]       # 737280 x 64 x 64, uint8
            break
        self.val_set = val_set
        
        print(f"Shape of image_ndarray = {self.__len__(), self.image_ndarray.shape[1:]}")
    

    def __len__(self):
        if self.val_set:
            return 1000
        else:
            return self.image_ndarray.shape[0]
    
    def __getitem__(self, index):
        return self.image_ndarray[index].reshape((1, 64, 64)).astype(np.float32), 0.0 # dummy datat to prevent breaking 

    def dsprite_dataloader(self,
        batch_size=8, 
        dataset_dir="/Users/shuagarw/repos/deep-generative-models/src/main/vae/dataset",
        num_workers=1,

    ):
        dsprite_dataset = DspritesDataset(dataset_dir=dataset_dir)
        data_loader = DataLoader(
            dsprite_dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        return data_loader


class DspriteVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        # self._has_setup_TrainerFn = True
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
        # self._has_setup_TrainerFn.FITTING = True

    def setup(self, stage=None) -> None:
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        
        self.train_dataset = DspritesDataset()
        
        # Replace CelebA with your dataset
        self.val_dataset = DspritesDataset(val_set=True)
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     
