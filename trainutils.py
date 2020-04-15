import os
import torch
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, params):

    # Load parameters
    num_epochs   = params["num_epochs"]
    loss_fn      = params["loss_fn"]
    optimizer    = params["optimizer"]
    training_loader = params["training_loader"]
    testing_loader  = params["testing_loader"]
    lr_scheduler = params["lr_scheduler"]
    weights_path = params["weights_path"]

    # Create Loss History - for plotting
    loss_history = {
        "train": [],
        "test"  : [],
    }

    # Create Metric History - for plotting
    metric_history = {
        "train": [],
        "test"  : [],
    }

    # Initialize 
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Train for num_epochs
    for epoch in range(num_epochs):

        # Get learning rate
        current_lr = get_lr(optimizer)

        # Print Stats
        print('Epoch {}/{}, current_lr={}'.format(epoch, num_epochs -1, current_lr))

        # Train the model
        model.train()
        train_loss, train_metric = calculate_loss(model, loss_fn, training_loader, optimizer)

        # Append Training Loss and Metrics
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # Set to eval mode and evaluate
        model.eval()
        with torch.no_grad():
            test_loss, test_metric = calculate_loss(model, loss_fn, testing_loader)
        
        # If loss is lower, save weights
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weights_path)
            print("Saved best model weights")

        # Store test/validation loss and metrics 
        loss_history["test"].append(test_loss)
        metric_history["test"].append(test_metric)

        # Update Learning Rate
        lr_scheduler.step(test_loss)
        if current_lr != get_lr(optimizer):
            print("Loading best model weights")
            model.load_state_dict(best_model_weights)
        
        # Print Loss Statistics 
        print("Train Loss: %.6f, Dev Loss: %.6f, Accuracy: %.2f"%(train_loss, test_loss, 100*test_metric))
        print("-"*10)
    
    model.load_state_dict(best_model_weights)
    return model, loss_history, metric_history

# Obtain Updated Learning Rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Calculate Accuracy
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# Compute Loss
def loss_batch(loss_fn, output, target, opt=None):

    # reshape output
    b, _, c = output.shape
    output = output.view(b,c)

    loss = loss_fn(output, target)

    with torch.no_grad():
        metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def calculate_loss(model, loss_fn, dataset_loader, optimizer = None):

    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_loader.dataset)

    for x_batch, y_batch in tqdm_notebook(dataset_loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Plug in x
        output = model(x_batch)
        
        # Calculate loss and append
        loss_b, metric_b = loss_batch(loss_fn, output, y_batch, optimizer)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric

def plot_loss(loss_hist, metric_hist):
    num_epochs = len(loss_hist["train"])

    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs+1), loss_hist["test"], label="test")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), metric_hist["test"], label="test")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()
    
def train_model_mse(model, params):

    # Load parameters
    num_epochs   = params["num_epochs"]
    loss_fn      = params["loss_fn"]
    optimizer    = params["optimizer"]
    training_loader = params["training_loader"]
    testing_loader  = params["testing_loader"]
    lr_scheduler = params["lr_scheduler"]
    weights_path = params["weights_path"]

    # Create Loss History - for plotting
    loss_history = {
        "train": [],
        "test"  : [],
    }

    # Create Metric History - for plotting
    metric_history = {
        "train": [],
        "test"  : [],
    }

    # Initialize 
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Train for num_epochs
    for epoch in range(num_epochs):

        # Get learning rate
        current_lr = get_lr(optimizer)

        # Print Stats
        print('Epoch {}/{}, current_lr={}'.format(epoch, num_epochs -1, current_lr))

        # Train the model
        model.train()
        train_loss, train_metric = calculate_loss_mse(model, loss_fn, training_loader, optimizer)

        # Append Training Loss and Metrics
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # Set to eval mode and evaluate
        model.eval()
        with torch.no_grad():
            test_loss, test_metric = calculate_loss_mse(model, loss_fn, testing_loader)
        
        # If loss is lower, save weights
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weights_path)
            print("Saved best model weights")

        # Store test/validation loss and metrics 
        loss_history["test"].append(test_loss)
        metric_history["test"].append(test_metric)

        # Update Learning Rate
        lr_scheduler.step(test_loss)
        if current_lr != get_lr(optimizer):
            print("Loading best model weights")
            model.load_state_dict(best_model_weights)
        
        # Print Loss Statistics 
        print("Train Loss: %.6f, Dev Loss: %.6f, Accuracy: %.2f"%(train_loss, test_loss, 100*test_metric))
        print("-"*10)
    
    model.load_state_dict(best_model_weights)
    return model, loss_history, metric_history

def calculate_loss_mse(model, loss_fn, dataset_loader, optimizer = None):

    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_loader.dataset)

    for x_batch, y_batch in tqdm_notebook(dataset_loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Plug in x
        output = model(x_batch)
        output = output.squeeze()
        
        # Calculate loss and append
        loss_b, metric_b = loss_batch_mse(loss_fn, output, y_batch, optimizer)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric

# Compute Loss
def loss_batch_mse(loss_fn, output, target, opt=None):

    # reshape output
    #b, _, c = output.shape
    #output = output.view(b,c).argmax(axis=1).type(torch.FloatTensor)

    loss = loss_fn(output, target.type(torch.FloatTensor))

    with torch.no_grad():
        metric_b = metrics_batch_mse(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def metrics_batch_mse(output, target):
    pred = output.type(torch.IntTensor)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects