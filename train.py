import torch
import numpy as np
import argparse

from src.config.configloading import load_config
from src.render import run_network
from src.trainer import Trainer

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/CCTA.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

        self.l2_loss = torch.nn.MSELoss(reduction='mean')

    def compute_loss(self, data, global_step, idx_epoch):
        loss = {"loss": 0.}

        projs = data.projs #.reshape(-1)
        image_pred = run_network(self.voxels, self.net, self.netchunk)
        train_output = image_pred.squeeze()[None, ...] #.transpose(1,4).squeeze(4)

        train_projs_one = self.ct_projector_first.forward_project(train_output)
        train_projs_two = self.ct_projector_second.forward_project(train_output)

        train_projs = torch.cat((train_projs_one,train_projs_two), 1)

        loss["loss"] = self.l2_loss(train_projs, projs)

        return loss

trainer = BasicTrainer()
trainer.start()
