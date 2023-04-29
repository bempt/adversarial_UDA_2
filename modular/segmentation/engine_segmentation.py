"""
Contains functions for training and testing a PyTorch model.
"""
import torch

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

from modular.segmentation import metrics_segmentation
from modular.utils import save_model
import modular.segmentation.data_setup_segmentation as seg_data


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"

        return False
    

def seg_update_results(epoch, results, seg_train_loss, seg_train_acc, seg_train_mIoU, seg_val_loss, seg_val_acc, seg_val_mIoU):
    print(
        f"Epoch: {epoch+1} | "
        f"seg_train_loss: {seg_train_loss:.4f} | "
        f"seg_train_acc: {seg_train_acc:.4f} | "
        f"seg_train_mIoU: {seg_train_mIoU:.4f} | "
        f"seg_val_loss: {seg_val_loss:.4f} | "
        f"seg_val_acc: {seg_val_acc:.4f} | "
        f"seg_val_mIoU: {seg_val_mIoU:.4f}"
    )

    results["seg_train_loss"].append(seg_train_loss)
    results["seg_train_acc"].append(seg_train_acc)
    results["seg_train_mIoU"].append(seg_train_mIoU)
    results["seg_val_loss"].append(seg_val_loss)
    results["seg_val_acc"].append(seg_val_acc)
    results["seg_val_mIoU"].append(seg_val_mIoU)

    

def seg_update_writer(seg_train_loss, seg_train_acc, seg_train_mIoU, seg_val_loss, seg_val_acc, seg_val_mIoU, writer=None, epoch=None):
    # See if there's a writer, if so, log to it
    if writer and epoch is not None:
        # Add results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"seg_train_loss": seg_train_loss,
                                            "seg_val_loss": seg_val_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"seg_train_acc": seg_train_acc,
                                            "seg_val_acc": seg_val_acc}, 
                           global_step=epoch)
        writer.add_scalars(main_tag="mIoU", 
                           tag_scalar_dict={"seg_train_mIoU": seg_train_mIoU,
                                            "seg_val_mIoU": seg_val_mIoU}, 
                           global_step=epoch)

        # Close the writer
        writer.close()

def seg_train_step(seg_model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               seg_loss_fn: torch.nn.Module, 
               seg_optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    seg_model.train()

    # Setup train loss and train accuracy values
    seg_train_loss, seg_train_acc, seg_train_mIoU = 0, 0, 0

    # Loop through data loader data batches
    for i, data in enumerate(tqdm(dataloader)):

      # Send data to target device
      image_tiles, mask_tiles = data

      # if patch:
      #     bs, n_tiles, c, h, w = image_tiles.size()

      #     image_tiles = image_tiles.view(-1,c, h, w)
      #     mask_tiles = mask_tiles.view(-1, h, w)

      image = image_tiles.to(device); mask = mask_tiles.to(device);

      # 1. Forward pass
      output = seg_model(image)

      # 2. Calculate  and accumulate loss
      loss = seg_loss_fn(output, mask)
      seg_train_loss += loss.item() 

      # 3. Optimizer zero grad
      seg_optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      seg_optimizer.step()

      # Calculate and accumulate metrics across all batches
      seg_train_acc += metrics_segmentation.pixel_accuracy(output, mask)
      seg_train_mIoU += metrics_segmentation.mIoU(output, mask)

    # Adjust metrics to get average loss and accuracy per batch 
    seg_train_loss = seg_train_loss / len(dataloader)
    seg_train_acc = seg_train_acc / len(dataloader)
    seg_train_mIoU = seg_train_mIoU / len(dataloader)

    return seg_train_loss, seg_train_acc, seg_train_mIoU

def seg_val_step(seg_model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              seg_loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    seg_model.eval() 

    # Setup val loss and val accuracy values
    val_loss, val_acc, val_mIoU = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for i, data in enumerate(tqdm(dataloader)):
            
            # Send data to target device
            image_tiles, mask_tiles = data

            # if patch:
            #         bs, n_tiles, c, h, w = image_tiles.size()

            #         image_tiles = image_tiles.view(-1,c, h, w)
            #         mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device); mask = mask_tiles.to(device);

            # 1. Forward pass
            output = seg_model(image)

            # 2. Calculate and accumulate loss
            loss = seg_loss_fn(output, mask)  
            val_loss += loss.item()

            # Calculate and accumulate metrics
            val_acc += metrics_segmentation.pixel_accuracy(output, mask)
            val_mIoU += metrics_segmentation.mIoU(output, mask)

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    val_mIoU = val_mIoU / len(dataloader)
    return val_loss, val_acc, val_mIoU

def seg_train(seg_model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          seg_optimizer: torch.optim.Optimizer,
          seg_loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device, 
          writer: torch.utils.tensorboard.writer.SummaryWriter, # new parameter to take in a writer
          target_dir: str,
          seg_model_name: str
          ) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"seg_train_loss": [],
               "seg_train_acc": [],
               "seg_train_mIoU": [],
               "seg_val_loss": [],
               "seg_val_acc": [],
                "seg_val_mIoU": []
    }

    es = EarlyStopping()

    # Loop through training and testing steps for a number of epochs
    done = False
    for epoch in range(epochs):
        if not done:
            train_loss, train_acc, train_mIoU = seg_train_step(seg_model=seg_model,
                                            dataloader=train_dataloader,
                                            seg_loss_fn=seg_loss_fn,
                                            seg_optimizer=seg_optimizer,
                                            device=device)
            val_loss, val_acc, val_mIoU = seg_val_step(seg_model=seg_model,
                                            dataloader=val_dataloader,
                                            seg_loss_fn=seg_loss_fn,
                                            device=device)
            
            print(f"Early Stopping: {es.status}")
            if es(seg_model, val_loss): done = True

            # Update results dictionary and print what's happening
            seg_update_results(epoch, results, train_loss, train_acc, train_mIoU, val_loss, val_acc, val_mIoU)

            ### New: Use the writer parameter to track experiments ###
            # See if there's a writer, if so, log to it
            seg_update_writer(train_loss, train_acc, train_mIoU, val_loss, val_acc, val_mIoU, writer, epoch)

            # save model
            save_model(seg_model, target_dir, seg_model_name)
    ### End new ###

    # Return the filled results at the end of the epochs
    return results

def adv_forward_pass(seg_model: torch.nn.Module,
                        disc_model: torch.nn.Module,
                        image: torch.Tensor,
                        mask_onehot: torch.Tensor,
                        device: torch.device):
    seg_output = seg_model(image)
    disc_output_real = disc_model(mask_onehot)
    disc_output_fake = disc_model(seg_output)

    # Concatenate real and fake outputs and domain labels
    disc_output = torch.cat((disc_output_real, disc_output_fake), dim=0)

    # Batch labels (source=0, target=1), same size as output_D
    domain_labels_real = torch.ones(disc_output_real.size(0), 1).to(device)
    domain_labels_fake = torch.zeros(disc_output_fake.size(0), 1).to(device)
    domain_labels = torch.cat((domain_labels_real, domain_labels_fake), dim=0)
    return disc_output_real, disc_output_fake, disc_output


def adv_train_step(seg_model: torch.nn.Module, 
                   disc_model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               seg_loss_fn: torch.nn.Module, 
               disc_loss_fn: torch.nn.Module,
               seg_optimizer: torch.optim.Optimizer,
               disc_optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    # Put model in train mode
    seg_model.train()
    disc_model.train()

    # Setup train loss and train accuracy values
    seg_train_loss, seg_train_acc, seg_train_mIoU = 0, 0, 0
    disc_train_loss, disc_train_acc = 0, 0

    # Loop through data loader data batches
    for i, data in enumerate(tqdm(dataloader)):

        # Send data to target device
        image_tiles, mask_tiles = data

        # if patch:
        #     bs, n_tiles, c, h, w = image_tiles.size()

        #     image_tiles = image_tiles.view(-1,c, h, w)
        #     mask_tiles = mask_tiles.view(-1, h, w)

        batch_size = image_tiles.size(0)

        image = image_tiles.to(device); mask = mask_tiles.to(device);

        mask_onehot = F.one_hot(mask, num_classes=seg_data.n_classes).permute(0, 3, 1, 2).float()
        mask_onehot = mask_onehot.to(device)

        ################################
        # Discriminator training phase #
        ################################

        # segmentation model gradients are turned off
        for param in seg_model.parameters():
            param.requires_grad = False
        for param in disc_model.parameters():
            param.requires_grad = True

        # 1. Forward pass
        seg_output, disc_output_real, disc_output_fake, disc_output = adv_forward_pass(seg_model, disc_model, image, mask_onehot, device)

        # 2. Calculate  and accumulate loss
        disc_loss = disc_loss_fn(disc_output, domain_labels) # domain classification loss
        disc_train_loss += disc_loss.item()

        # 3. Optimizer zero grad
        disc_optimizer.zero_grad()

        # 4. Loss backward
        disc_loss.backward()

        # 5. Optimizer step
        disc_optimizer.step()

        disc_train_acc += metrics_segmentation.disc_accuracy(disc_output_real, disc_output_fake, batch_size)

        ###############################
        # Segmentation training phase #
        ###############################

        # turn off discriminator gradients
        for param in seg_model.parameters():
            param.requires_grad = True
        for param in disc_model.parameters():
            param.requires_grad = False

        # 1. Forward pass
        seg_output = seg_model(image)
        disc_output = disc_model(seg_output)

        # batch labels (source=0, target=1), same size as output_D
        domain_labels = torch.ones(disc_output.size(0), 1).to(device)

        # 2. Calculate  and accumulate loss
        seg_loss = seg_loss_fn(seg_output, mask)
        disc_loss = disc_loss_fn(disc_output, domain_labels)
        total_seg_loss = seg_loss + disc_loss

        seg_train_loss += total_seg_loss.item() 

        # 3. Optimizer zero grad
        seg_optimizer.zero_grad()

        # 4. Loss backward
        total_seg_loss.backward()

        # 5. Optimizer step
        seg_optimizer.step()

        # Calculate and accumulate metrics across all batches
        seg_train_acc += metrics_segmentation.pixel_accuracy(seg_output, mask)
        seg_train_mIoU += metrics_segmentation.mIoU(seg_output, mask)

    # make sure gradients are turned on for both models
    for param in seg_model.parameters():
        param.requires_grad = True
    for param in disc_model.parameters():
        param.requires_grad = True
    
    # Adjust metrics to get average loss and accuracy per batch 
    seg_train_loss = seg_train_loss / len(dataloader)
    seg_train_acc = seg_train_acc / len(dataloader)
    seg_train_mIoU = seg_train_mIoU / len(dataloader)

    return seg_train_loss, seg_train_acc, seg_train_mIoU


def adv_val_step(seg_model: torch.nn.Module, 
                disc_model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              seg_loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    seg_model.eval() 
    disc_model.eval()

    # Setup val loss and val accuracy values
    seg_val_loss, seg_val_acc, seg_val_mIoU = 0, 0, 0
    disc_val_loss, disc_val_acc = 0, 0


    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for i, data in enumerate(tqdm(dataloader)):
            
            # Send data to target device
            image_tiles, mask_tiles = data

            # if patch:
            #         bs, n_tiles, c, h, w = image_tiles.size()

            #         image_tiles = image_tiles.view(-1,c, h, w)
            #         mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device); mask = mask_tiles.to(device);

            # 1. Forward pass
            seg_output = seg_model(image)
            # ADD ADV_FORWARD_PASS HERE

            # 2. Calculate and accumulate loss
            loss = seg_loss_fn(seg_output, mask)  
            seg_val_loss += loss.item()

            # Calculate and accumulate metrics
            seg_val_acc += metrics_segmentation.pixel_accuracy(seg_output, mask)
            seg_val_mIoU += metrics_segmentation.mIoU(seg_output, mask)

    # Adjust metrics to get average loss and accuracy per batch 
    seg_val_loss = seg_val_loss / len(dataloader)
    seg_val_acc = seg_val_acc / len(dataloader)
    seg_val_mIoU = seg_val_mIoU / len(dataloader)

    return seg_val_loss, seg_val_acc, seg_val_mIoU