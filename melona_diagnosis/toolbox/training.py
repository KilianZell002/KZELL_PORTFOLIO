import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from toolbox import utils

class Trainer():
    def __init__(self, model, device, task='segmentation'):
        """
        Initialize the Trainer with the provided PyTorch model and processing device.

        Parameters:
            model: The PyTorch model to be trained.
            device (str): The device on which the model and data will be processed ('cuda' or 'cpu').
            task (str): The type of task for which the Trainer is configured ('segmentation' or 'classification').

        Note:
            The Trainer initializes the loss criterion based on the specified task.
            For 'segmentation' task, it uses the BCEDiceLoss criterion.
            For 'classification' task, it uses the BCELoss criterion.
        """
        # Set the provided model and device
        self.model = model
        self.device = device
        self.task = task

        if task == 'segmentation':
          # Initialize BCEDiceLoss criterion using the provided device and move it to the device
          self.criterion = utils.BCEDiceLoss(self.device).to(device)
        elif task == 'classification':
          # Initialize BCELoss criterion and move it to the device
          self.criterion = nn.BCELoss().to(device)
        
    def train(self, epochs, trainloader, testloader=None, model_name=None, learning_rate=0.001, weight_decay=0., clip_gradient=False, metadata=False, verbose=True):
        """
        Train the model.

        Args:
            epochs (int): Number of epochs for the training session.
            trainloader (torch.utils.data.DataLoader): Training dataloader.
            testloader (torch.utils.data.DataLoader): Test dataloader (optional).
            model_name (str): Name of the file to save the model checkpoint.
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            clip_gradient (bool): If True, clip gradients during training.
            metadata (bool): If True, use metadata during training.
            verbose (bool): If True, print training progress; otherwise, suppress output.

        Returns:
            list: List of training losses at every epoch.
            list: List of test losses at every epoch (empty list if testloader is not provided).
        """

        # Lista to store training and test loss for each epoch
        losses_train = []
        losses_test = []

        # Set Adam as the optimizer. Link to all optimizers in torch.optim: https://pytorch.org/docs/stable/optim.html
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Reduce learning rate on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.85, patience=2, verbose=verbose)

        if verbose == True:
          print('Starting Training Process')

        # Epoch Loop
        for epoch in range(epochs):
          # Start the time to compute s/epch
          start_time = time.time()

          # Set the model to training mode
          self.model.train()  
          
          # Training a single epoch
          epoch_loss = self._train_epoch(trainloader, clip_gradient=clip_gradient, metadata=metadata)

          # Reduce LR On Plateau
          self.scheduler.step(epoch_loss)
            
          # Get test loss over epochs
          if testloader is not None:
              self.model.eval()  # Set the model to evaluation mode
              test_loss = 0  # Initialize the variable to accumulate test loss for the epoch
              
              # Loop through batches in the testloader
              for _, data in enumerate(testloader):
                  image = data['image'].to(self.device)
                  
                  if self.task == 'segmentation':
                      gt = data['mask'].to(self.device)
                  elif self.task == 'classification':
                      gt = data['label'].to(self.device).float()
                      if metadata == True:
                          meta = data['metadata'].to(self.device)
                
                  if metadata == False:
                    output = self.model(image)  # Forward pass to get model predictions
                  elif metadata == True:
                    output = self.model(image, metadata=meta)  # Forward pass to get model predictions
                  
                  loss_value = self.criterion(output, gt)  # Calculate the loss between predictions and ground truth
                  test_loss += loss_value.item()  # Accumulate the test loss for the batch

              av_test_loss = test_loss / len(testloader.dataset)  # Calculate average test loss per sample
          else: 
              av_test_loss = None

          # End time count for the epoch
          end_time = time.time()

          # Save the model every 5 epochs
          if epoch % 5 == 0 and model_name is not None:
              torch.save(self.model.state_dict(), model_name)
              print(f'Model saved at {model_name}')

          # Print losses if verbose is True
          if verbose == True:
              if testloader is not None:
                  print(f'Epoch: {epoch+1:03d}, Train Loss: {epoch_loss:.7f}, Test Loss: {av_test_loss:.7f}, Time: {end_time - start_time:.3f} sec/epch')
              else:
                  print(f'Epoch: {epoch+1:03d}, Train Loss: {epoch_loss:.7f}, Time: {end_time - start_time:.3f} sec/epch')
            
          # Append training and test loss for the epoch to the list
          losses_train.append(epoch_loss)
          losses_test.append(av_test_loss)
        
        # Save the final model
        torch.save(self.model.state_dict(), model_name)
        if verbose == True:
          print(f'Model saved at {model_name}')

        # Return the lists of training and test losses
        return losses_train, losses_test
    
    def _train_epoch(self, trainloader, clip_gradient=False, metadata=False):
        """
        Train the model for one epoch.

        Args:
            trainloader (torch.utils.data.DataLoader): Training dataloader for the optimizer.
            clip_gradient (bool): If True, clip gradients during training.
            metadata (bool): If True, use metadata during training.

        Returns:
            float: Loss calculated for each epoch.
        """

        # Initialize epoch-level loss, batch-level loss, and batch iteration counters
        epoch_loss, batch_iteration = 0, 0

        # Training loop
        for _, data in enumerate(trainloader):
            
            # Increment the batch iteration counter
            batch_iteration += 1

            # Move input data and target mask to the device
            image = data['image'].to(self.device)
            
            if self.task == 'segmentation':
              gt = data['mask'].to(self.device)
            if self.task == 'classification':
              gt = data['label'].to(self.device).float()
              if metadata == True:
                 meta = data['metadata'].to(self.device)

            # Clear the gradients of the optimizer
            self.optimizer.zero_grad()
            
            if metadata == False:
                # Forward pass: calculate predicted output
                output = self.model(image)
            if metadata == True:
                # Forward pass: calculate predicted output
                output = self.model(image, metadata=meta)

            # Calculate the loss value using the specified criterion
            loss_value = self.criterion(output, gt)

            # Backward pass: compute gradients
            loss_value.backward()

            if clip_gradient == True:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update the model's parameters (perform optimization step)
            self.optimizer.step()

            # Update the running training loss
            epoch_loss += loss_value.item()

        # Calculate the average epoch loss by dividing by the total number of batches and batch size
        epoch_loss = epoch_loss / (batch_iteration * trainloader.batch_size)

        # Return the average epoch loss
        return epoch_loss

    def predict(self, data, threshold=0.5, task='segmentation', metadata=False):
        """ 
        Calculate the output mask on a single input data.

        Args:
          data (dict): A dictionary containing 'image' and 'mask' tensors.
          threshold (float): Threshold for binarizing the output mask (default is 0.5).
          task (str): Task type, either 'segmentation' or 'classification' (default is 'segmentation').
          metadata (bool): if 'True', integrates metadata to the model for prediction.

        Returns:
          tuple: A tuple containing the input image tensor, ground truth mask or label tensor, and predicted output mask tensor.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Extract the image tensor from the data dictionary and move it to the device
        image_tensor = data['image'].to(self.device)

        # Reshape the image tensor to match the model's input size
        image_tensor = image_tensor.view((-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))

        if metadata == True:
           meta = data['metadata'].to(self.device)
           meta = meta.view((-1, meta.shape[0]))

        # Forward pass to get the model's prediction
        if metadata == False:
           output = self.model(image_tensor).detach().cpu()
        elif metadata == True:
           output = self.model(image_tensor, metadata=meta).detach().cpu()

        if task == 'segmentation':
            # Binarize the output mask using the specified threshold
            output = torch.where(output > threshold, 1.0, 0.0)
            # Return the input image, ground truth mask, and predicted output mask
            return data['image'], data['mask'], output

        elif task == 'classification':
            # Binarize the output label using the specified threshold
            output = int((output > threshold).float().item())
            # Return the input image, ground truth label, and predicted output label
            return data['image'], data['label'], output
        
    def load_model(self, model_path, verbose=True):
        """
        Load the trained model's state dictionary from a saved checkpoint file.

        Args:
            model_path (str): Path to the saved model checkpoint.
            verbose (bool): If True, print a message indicating successful model loading. Default is True.

        Returns:
            None
        """
        # Use map_location to load on the CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        
        if verbose:
            print('Model successfully loaded.')