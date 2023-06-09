# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18

num_classes = 8
num_epochs = 1
torch.set_printoptions(linewidth=200)


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


"""
1.  Define and build a PyTorch Dataset
"""


class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        # dlist[i] = {
        #     b'batch_label': b'training batch 1 of 5',
        #     b'labels': 8047 labels
        #     b'data': 8047 images
        #     b'filenames': 8047 filenames
        # }
        dlist = [unpickle(data_file) for data_file in data_files]
        self.images = [image for data in dlist for image in data[b"data"]]
        self.labels = [label for data in dlist for label in data[b"labels"]]
        self.transform = transform if transform else lambda x: x
        self.target_transform = (
            target_transform if target_transform else lambda x: x
        )

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset.

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        image = torch.tensor(self.images[idx]).reshape(3, 32, 32).float()
        label = self.labels[idx]
        return image, label


def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """

    return lambda x: x


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    return CIFAR10(data_files, transform=transform)


"""
2.  Build a PyTorch DataLoader
"""


def build_dataloader(
    dataset, loader_params={"batch_size": 32, "shuffle": True}
):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader.

    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those
    respective parameters in the PyTorch DataLoader class.

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    return DataLoader(dataset, **loader_params)


"""
3. (a) Build a neural network class.
"""


class FinetuneNet(torch.nn.Module):
    def __init__(self, pretrained=False, pretrained_path="resnet18.pt"):
        """
        Initialize your neural network here. Remember that you will be
        performing finetuning in this network so follow these steps:

        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s).
        """
        super().__init__()
        self.model = resnet18()
        if pretrained:
            print("Loading pretrained model from {}".format(pretrained_path))
            self.model.load_state_dict(torch.load(pretrained_path))
            for param in self.model.parameters():
                param.requires_grad = False
        # Change the shape of the last layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        return self.model.forward(x)


"""
3. (b)  Build a model
"""


def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet(trained)
    return net


"""
4.  Build a PyTorch optimizer
"""


def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "Adam":
        return torch.optim.Adam(params=model_params, **hparams)
    if optim_type == "SGD":
        return torch.optim.SGD(params=model_params, **hparams)
    raise NotImplementedError(f"Optimizer {optim_type} not implemented!")


"""
5. Training loop for model
"""


def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    for images, labels in train_dataloader:
        # 1.  The model makes a prediction.
        output = model.forward(images)

        # 2.  Calculate the error in the prediction (loss).
        loss = loss_fn(output, labels)
        print(f"Predictions: {torch.argmax(output, 1)}")
        print(f"Labels:      {labels}")
        print(f"Loss:        {loss}")
        print("-" * 80)

        # 3.  Zero the gradients of the optimizer.
        optimizer.zero_grad()

        # 4.  Perform backpropagation on the loss.
        loss.backward()

        # 5.  Step the optimizer.
        optimizer.step()


"""
6. Testing loop for model
"""


def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            outputs = model.forward(images)
            prediction = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        return correct / total


"""
7. Full model training and testing
"""


def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    train_dataloader = build_dataloader(
        build_dataset(
            [
                "cifar10_batches/data_batch_1",
                "cifar10_batches/data_batch_2",
                "cifar10_batches/data_batch_3",
                "cifar10_batches/data_batch_4",
                "cifar10_batches/data_batch_5",
            ],
        )
    )
    test_dataloader = build_dataloader(
        build_dataset(["cifar10_batches/test_batch"])
    )
    model = build_model(True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.model.parameters(), {"lr": 0.0005})

    for epoch in range(num_epochs):
        print(f"========== Epoch {epoch} ==========")
        train(train_dataloader, model, loss_fn, optimizer)
        print(f"Accuracy: {test(test_dataloader, model)}")
    return model


if __name__ == "__main__":

    def check_data():
        files = [
            "cifar10_batches/data_batch_1",
            "cifar10_batches/data_batch_2",
        ]
        dataset = build_dataset(files)
        print("length of dataset: {}".format(len(dataset)))
        image, label = dataset[0]
        print("image type: {}".format(type(image)))
        print("label type: {}".format(type(label)))
        print("image shape: {}".format(image.shape))
        print("label: {}".format(label))
        dataloader = build_dataloader(dataset)
        for image, label in dataloader:
            print("image type (dataloader): {}".format(type(image)))
            print("label type (dataloader): {}".format(type(label)))
            print("image shape (dataloader): {}".format(image.shape))
            print("label (dataloader): {}".format(label))
            break

    def check_model():
        model = run_model()
        test_dataloader = build_dataloader(
            build_dataset(
                ["cifar10_batches/test_batch"],
                transform=get_preprocess_transform("train"),
            ),
        )
        test(test_dataloader, model)

    check_model()
