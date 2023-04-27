from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import sys
import os
import urllib.request

class DspritesDataset(Dataset):
    def __init__(
        self,
        dataset_url="https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
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

        print(f"Shape of image_ndarray = {self.image_ndarray.shape}")
    

    def __len__(self):
        return self.image_ndarray.shape[0]
    
    def __getitem__(self, index):
        return self.image_ndarray[index].reshape((1, 64, 64)).astype(np.float32)

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


