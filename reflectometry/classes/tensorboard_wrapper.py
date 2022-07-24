import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from os.path import join
from utils import convert_to_uint8
import os
import matplotlib.pyplot as plt

class TensorBoardWrapper(object):
    def __init__(self, I_gt, title=None):
        if title is None:
            self.train_id = datetime.now().strftime("%d%m-%H%M-%S")
        else:
            self.train_id = title

        self.writer = SummaryWriter(log_dir=f"checkpoints/{self.train_id}")
        self.folder = join("checkpoints", self.train_id)
        # I_copy = np.copy(I_gt)
        # I_copy[I_copy>255] = 255
        # I_copy = I_copy.astype(np.uint8)
        I_copy = convert_to_uint8(I_gt, max_factor=0.05)
        self.writer.add_image(f"ground_truth", np.transpose(I_copy, (2,0,1)))

    def update(self, I_res, loss, rel_dist1, res_Ks, res_n, iter):
        # I_copy = np.copy(I_res)
        # I_copy[I_copy>255] = 255
        # I_copy = I_copy.astype(np.uint8)
        I_copy = convert_to_uint8(I_res, max_factor=0.02)
        self.writer.add_image(f"simulated_images", np.transpose(I_copy, (2,0,1)), global_step=iter)
        self.writer.add_scalar("loss", loss, global_step=iter)
        self.writer.add_scalar("relative_dist1", rel_dist1, global_step=iter)
        self.writer.add_scalar("Ks_opt", res_Ks, global_step=iter)
        self.writer.add_scalar("n_opt", res_n, global_step=iter)
        # plt.figure()
        # plt.imshow(I_copy)
        # plt.show()

    def update_albedo_image(self, I_res, loss, rel_dist1, albedo_image, iter):
        I_copy = convert_to_uint8(I_res, max_factor=0.05)
        self.writer.add_image(f"simulated_images", np.transpose(I_copy, (2,0,1)), global_step=iter)
        self.writer.add_image(f"albedo_image_res", np.transpose(albedo_image, (2,0,1)), global_step=iter)
        self.writer.add_scalar("loss", loss, global_step=iter)
        self.writer.add_scalar("relative_dist1", rel_dist1, global_step=iter)
        np.save(join("checkpoints",self.train_id,f"albedo_{iter}.npy"), albedo_image)
