"""
Contains functions for training and testing a PyTorch model.
"""
import torch

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter

from modular.segmentation import metrics_segmentation

'''
https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
'''
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def print_epoch_results(epoch, train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU):
    """
    Prints the results of an epoch.
    
    Parameters:
        epoch (int): The current epoch number.
        train_loss (float): The training loss.
        train_acc (float): The training accuracy.
        train_mIoU (float): The training mean Intersection over Union (mIoU).
        test_loss (float): The testing loss.
        test_acc (float): The testing accuracy.
        test_mIoU (float): The testing mean Intersection over Union (mIoU).
    """
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"train_mIoU: {train_mIoU:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f} | "
        f"test_mIoU: {test_mIoU:.4f}"
    )

def update_results(results, train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU):
    """Updates a dictionary of results with training and testing metrics for an epoch.

    Args:
      results: A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                  train_acc: [...],
                  train_mIoU: [...],
                  test_loss: [...],
                  test_acc: [...],
                  test_mIoU: [...]} 
      train_loss: A float indicating the training loss for the current epoch.
      train_acc: A float indicating the training accuracy for the current epoch.
      train_mIoU: A float indicating the training mIoU for the current epoch.
      test_loss: A float indicating the testing loss for the current epoch.
      test_acc: A float indicating the testing accuracy for the current epoch.
      test_mIoU: A float indicating the testing mIoU for the current epoch.

    Returns:
      None. The results dictionary is updated in place.
    """
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["train_mIoU"].append(train_mIoU)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    results["test_mIoU"].append(test_mIoU)

def update_writer(train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU, writer=None, epoch=None):
    # See if there's a writer, if so, log to it
    if writer and epoch is not None:
        # Add results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc}, 
                           global_step=epoch)
        writer.add_scalars(main_tag="mIoU", 
                           tag_scalar_dict={"train_mIoU": train_mIoU,
                                            "test_mIoU": test_mIoU}, 
                           global_step=epoch)

        # Close the writer
        writer.close()

def seg_train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc, train_mIoU = 0, 0, 0

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
      output = model(image)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(output, mask)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate metrics across all batches
      train_acc += metrics_segmentation.pixel_accuracy(output, mask)
      train_mIoU += metrics_segmentation.mIoU(output, mask)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_mIoU = train_mIoU / len(dataloader)
    return train_loss, train_acc, train_mIoU

def seg_test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc, test_mIoU = 0, 0, 0

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
            output = model(image)

            # 2. Calculate and accumulate loss
            loss = loss_fn(output, mask)  
            test_loss += loss.item()

            # Calculate and accumulate metrics
            test_acc += metrics_segmentation.pixel_accuracy(output, mask)
            test_mIoU += metrics_segmentation.mIoU(output, mask)

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    test_mIoU = test_mIoU / len(dataloader)
    return test_loss, test_acc, test_mIoU

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device, 
          writer: torch.utils.tensorboard.writer.SummaryWriter # new parameter to take in a writer
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      val_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "train_mIoU": [],
               "test_loss": [],
               "test_acc": [],
                "test_mIoU": []
    }

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc, train_mIoU = seg_train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc, test_mIoU = seg_test_step(model=model,
                                          dataloader=val_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        if early_stopper.early_stop(test_loss):
            print("Early stopping")
            break

        # Print out what's happening
        print_epoch_results(epoch, train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU)

        # Update results dictionary
        update_results(results, train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        update_writer(results, train_loss, train_acc, train_mIoU, test_loss, test_acc, test_mIoU, writer=None, epoch=None)


    ### End new ###

    # Return the filled results at the end of the epochs
    return results