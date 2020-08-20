import os
import numpy as np
import torch
import torch.nn as nn
import sys
from itertools import chain

from edflow import TemplateIterator, get_logger

try:
    from iterators.utils import (
        set_gpu,
        set_random_state,
        set_requires_grad,
        pt2np,
        convert_logs2numpy,
        calculate_gradient_penalty
    )
except ModuleNotFoundError:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cur_path = cur_path.replace("\\", "/")
    path = cur_path[:cur_path.rfind('/')]
    sys.path.append(path)
    from iterators.utils import (
        set_gpu,
        set_random_state,
        set_requires_grad,
        pt2np,
        convert_logs2numpy,
        calculate_gradient_penalty
    )


class Style_Transfer_Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.logger = get_logger("Iterator")
        #set device
        self.device = set_gpu(config)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        
        self.config = config
        set_random_state(self.config["random_seed"])

        self.batch_size = config["batch_size"]
        assert self.batch_size == 1, "The current implementation only works for batchsize 1"

        self.logger.info(f"{model}")
        self.model = model.to(self.device)

        self.optimizer_G = torch.optim.Adam(chain(self.model.c_enc.parameters(), self.model.s_enc.parameters(), self.model.dec.parameters()), lr=config["optimizer"]["lr"], betas=(.5, .999))
        D_lr_factor = self.config["optimizer"]["D_lr_factor"] if "D_lr_factor" in config["optimizer"] else 1
        self.optimizer_D = torch.optim.Adam(self.model.disc.parameters(), lr=D_lr_factor * self.config["optimizer"]["lr"], betas=(.5, .999))
        self.logger.debug("Learning rate for generator: {}\nLearning rate for disriminator: {}".format(self.config["optimizer"]["lr"], self.config["optimizer"]["lr"]*D_lr_factor))

        self.real_label = torch.ones(4).float().to(self.device)
        self.fake_label = torch.zeros(4).float().to(self.device)

        self.zero = torch.tensor(0).float().to(self.device)

    def criterion(self, content_images, style_images, output, style_label):
        """Calculates losses according to models in and output

        Args:
            content_images (torch.Tensor):  Input of content encoder
            style_images (torch.Tensor): Input of style encoder
            output (torch.Tensor): Output of model by content_ and style_images
            style_label (torch.Tensor): Label of style of artist (one hot encoded)
        Returns:
            dict: Losses of the model seperated into "generator and discriminator
        """
        losses = {}

        ######### Generator Loss #########
        #import weights
        adv_weight = self.config["losses"]["adv_weight"]
        rec_weight = self.config["losses"]["rec_weight"]
        fp_cont_weight = self.config["losses"]["fp_cont_weight"]
        fpt_style_weight = self.config["losses"]["fpt_style_weight"]
        fpt_margin = self.config["losses"]["fpt_margin"]
        fpd_weight = self.config["losses"]["fpd_weight"]


        losses["generator"] = {}
        #adversarial loss
        losses["generator"]["adv"] = torch.mean(nn.BCELoss()(self.model.disc(output, style_label), self.real_label))
        #reconstruction loss
        losses["generator"]["rec"] = torch.mean(nn.MSELoss()(output, torch.cat((content_images, content_images))))
        #Fixpoint content loss
        losses["generator"]["fp_cont"] = torch.mean(nn.MSELoss()(self.model.c_enc(output), self.model.c_enc(torch.cat((content_images, content_images)))))
        #Fixpoint triplet style loss
        fpt1 = nn.MSELoss()(self.model.s[:1], self.model.s_enc(output[:1]))
        fpt2 = nn.MSELoss()(self.model.s[:1], self.model.s_enc(output[2:3]))
        losses["generator"]["fpt_style"] = torch.max(torch.stack([self.zero, fpt_margin + fpt1 - fpt2]))
        #Fixpoint disentanglement loss
        fpd1 = nn.MSELoss()(self.model.s_enc(output[:1]), self.model.s_enc(output[1:2]))
        fpd2 = nn.MSELoss()(self.model.s_enc(output[:1]), self.model.s[:1])
        losses["generator"]["fpd"] = torch.max(torch.stack([self.zero, fpd1 - fpd2]))

        losses["generator"]["total"] = adv_weight * losses["generator"]["adv"] \
                                        + rec_weight * losses["generator"]["rec"] \
                                        + fp_cont_weight * losses["generator"]["fp_cont"] \
                                        + fpt_style_weight * losses["generator"]["fpt_style"] \
                                        + fpd_weight * losses["generator"]["fpd"]

        ######### Discriminator Loss #########
        gp_weight = self.config["losses"]["gp_weight"] if "gp_weight" in self.config["losses"] else 0

        d_real = self.model.disc(style_images.detach(), style_label).view(-1)
        d_fake = self.model.disc(output.detach(), style_label).view(-1)

        losses["discriminator"] = {}
        losses["discriminator"]["outputs_real"] = np.mean(d_real.detach().cpu().numpy())
        losses["discriminator"]["outputs_fake"] = np.mean(d_fake.detach().cpu().numpy())
        losses["discriminator"]["fake"] = torch.mean(nn.BCELoss()(d_fake, torch.zeros_like(d_fake).to(self.device)))
        losses["discriminator"]["real"] = torch.mean(nn.BCELoss()(d_real, torch.ones_like(d_real).to(self.device)))
        losses["discriminator"]["total"] = losses["discriminator"]["fake"] + losses["discriminator"]["real"]
        if gp_weight > 0:
            losses["discriminator"]["gp"] = calculate_gradient_penalty(self.model.disc, torch.cat((style_images, style_images)), output.detach(), self.device, style_label)
            losses["discriminator"]["total"] += gp_weight * losses["discriminator"]["gp"]
        return losses

    def step_op(self, model, **kwargs):

        #Get data from dataloader and push to device
        content1 = torch.tensor(kwargs["content1"]).float().to(self.device)
        content2 = torch.tensor(kwargs["content2"]).float().to(self.device)
        content_images = torch.cat([content1, content2])
        
        style1 = torch.tensor(kwargs["style1"]).float().to(self.device)
        style2 = torch.tensor(kwargs["style2"]).float().to(self.device)
        style_images = torch.cat([style1, style2])
        style_label = torch.tensor(kwargs["label"]).float().to(self.device)
        artist = kwargs["artist"]
        self.logger.debug("content_images.shape: {}".format(content_images.shape))
        self.logger.debug("style_images.shape: {}".format(style_images.shape))
        
        #Calculate output
        output = self.model(content_images, style_images)
        self.logger.debug("output.shape: {}".format(output.shape))
        
        #Calculate losses
        losses = self.criterion(content_images, style_images, output, style_label)

        def train_op():
            """Training step including log generation for evaluation"""
            #Generator update
            set_requires_grad([self.model.disc], False)
            self.optimizer_G.zero_grad()
            losses["generator"]["total"].backward()
            self.optimizer_G.step()

            #Discriminator update
            set_requires_grad([self.model.disc], True)
            self.optimizer_D.zero_grad()
            losses["discriminator"]["total"].backward()
            self.optimizer_D.step()

        def log_op():
            logs = self.prepare_logs(losses, content_images, style_images, output)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    
    def prepare_logs(self, losses, content_images, style_images, output):
        """Return a log dictionary with all instersting data to log."""
        logs = {
            "images": {},
            "scalars": {
                **losses
            }
        }
        ############
        ## images ##
        ############
        content_images = pt2np(content_images)
        style_images = pt2np(style_images)
        output = pt2np(output)
        logs["images"].update({"content_images": content_images})
        logs["images"].update({"style_images": style_images})
        logs["images"].update({"output": output})
        # convert to numpy
        logs = convert_logs2numpy(logs)
        return logs

    def save(self, checkpoint_path):
        """This function is used to save all weights of the model as well as the optimizers.'sketch_decoder' refers to the decoder of the face2sketch network and vice versa. 

        Args:
            checkpoint_path (str): Path where the weights are saved. 
        """
        state = {}
        state["model"] = self.model.state_dict()
        state["optimizer_D"] = self.optimizer_D.state_dict()
        state["optimizer_G"] = self.optimizer_G.state_dict()
        torch.save(state, checkpoint_path)
    
    def restore(self, checkpoint_path):
        """This function is used to load all weights of the model from a previous run.

        Args:
            checkpoint_path (str): Path from where the weights are loaded.
        """
        state = torch.load(checkpoint_path)
        self.model.load_state(state_dict=state["model"])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D.load_state_dict(state['optimizer_D'])