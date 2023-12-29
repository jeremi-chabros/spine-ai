# source ~/venv-metal/bin/activate
# bokeh serve --show training.py

from data_loader import ScanDataset
from model import NIMA
from tornado import gen
from threading import Thread
from functools import partial
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.io import curdoc
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append("/Users/jjc/Research/CNOC_Spine/All/code")


sys.path.append('/Users/jjc/Research/CNOC_Spine/All/code/gradcam')


@gen.coroutine
def update(new_data):
    source.stream(new_data)


def get_transforms():
    """Returns transformations for datasets."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, val_transform, test_transform


def initialize_model(config, device, base_model):
    """Initializes and returns the model."""
    model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(
            config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' %
              config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(
            model.features, device_ids=config.gpu_ids)
    model = model.to(device)

    return model


def main(config):

   # Define device
    device = torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device.type == "mps":
        torch.device("mps")
        print("Using MPS device.")
    else:
        print("MPS device not found. Using CPU.")

    train_transform, val_transform, _ = get_transforms()

    trainset = ScanDataset(csv_file=config.train_csv_file,
                           root_dir=config.train_img_path, transform=train_transform)
    valset = ScanDataset(csv_file=config.val_csv_file,
                         root_dir=config.val_img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                                             shuffle=False, num_workers=config.num_workers)

    base_model = models.resnet50(weights="DEFAULT", progress=False)
    model = initialize_model(config, device, base_model)
    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    # optimizer = optim.SGD([
    #     {'params': model.features.parameters(), 'lr': conv_base_lr},
    #     {'params': model.classifier.parameters(), 'lr': dense_lr}],
    #     momentum=0.9
    # )
    optimizer = optim.Adam(
        model.parameters(), lr=config.conv_base_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.lr_decay_rate)

    wpos = 0.73  # majority weight
    wneg = 1.58  # minority weight
    weight_binary = wpos/wneg
    weights = torch.tensor([wpos, wneg]).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.181, 1.458]).to(device)) # higher weight for positive minority class
    # criterion = torch.nn.BCELoss(weight=torch.tensor(weight_binary).to(device))
    # criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
    # criterion = torch.nn.BCELoss(weight=weight)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []

        previous_filename = None
        for epoch in range(config.warm_start_epoch, config.epochs):
            saveflag = 0
            correct_train = 0
            total_train = 0
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                labels = labels.view(-1, 2)
                outputs = model(images)
                outputs = outputs.view(-1, 2)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
#                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()
            if epoch % config.lr_decay_freq == 0:
                scheduler.step()

#                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))

            avg_loss = sum(batch_losses) / (len(trainset) //
                                            config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training EMD loss: %.4f' %
                  (epoch + 1, avg_loss))

            # Get the index of the max value
            _, predicted_train = torch.max(outputs, 1)
            _, true_labels_train = torch.max(labels, 1)
            correct_train += (predicted_train ==
                              true_labels_train).sum().item()
            total_train += labels.size(0)

            train_accuracy = correct_train / total_train
            print(
                f'Epoch {epoch + 1} Training Accuracy: {100*train_accuracy:.2f}%')

            # exponetial learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * \
                    config.lr_decay_rate ** ((epoch + 1) /
                                             config.lr_decay_freq)
                dense_lr = dense_lr * \
                    config.lr_decay_rate ** ((epoch + 1) /
                                             config.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )

            # do validation after each epoch
            batch_val_losses = []
            correct_val = 0
            total_val = 0
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                labels = labels.view(-1, 2)
                with torch.no_grad():
                    outputs = model(images)
                val_outputs = outputs.view(-1, 2)
                val_loss = criterion(val_outputs, labels)
#                val_loss = emd_loss(labels, outputs)
                # val_loss = weighted_binary_cross_entropy(outputs, labels, weights=weights)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / \
                (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            _, predicted_val = torch.max(val_outputs, 1)
            _, true_labels_val = torch.max(labels, 1)
            correct_val += (predicted_val == true_labels_val).sum().item()
            total_val += labels.size(0)
            val_accuracy = correct_val / total_val
            print(
                f'Epoch {epoch + 1} Validation Accuracy: {100*val_accuracy:.2f}%')

            if train_accuracy + val_accuracy == 2.0 or val_accuracy == 1.0:
                saveflag = 1

            print('Epoch %d completed. Averaged BCE loss on val set: %.4f. Inital val loss : %.4f.' % (
                epoch + 1, avg_val_loss, init_val_loss))

            new_data = {'epochs': [epoch],
                        'trainlosses': [avg_loss],
                        'vallosses': [avg_val_loss]}
            doc.add_next_tick_callback(partial(update, new_data))

            if avg_val_loss < init_val_loss or saveflag:
                init_val_loss = avg_val_loss
                filename = f'epoch-{epoch + 1}.pkl'
                full_path = os.path.join(config.ckpt_path, filename)
                print(f'Saving model...')
                torch.save(model.state_dict(), full_path)
                print('Done.\n')

                if previous_filename and os.path.exists(previous_filename):
                    os.remove(previous_filename)

                previous_filename = full_path

                count = 0
            else:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val BCE loss has not decreased in %d epochs. Training terminated.' %
                          config.early_stopping_patience)
                    break

        print('Training completed.')

        # # plot train and val loss
        # epochs = range(1, epoch + 2)
        # plt.plot(epochs[0:-1], train_losses, 'b-', label='train loss')
        # plt.plot(epochs[0:-1], val_losses, 'g-', label='val loss')
        # plt.title('BCE loss')
        # plt.legend()
        # plt.savefig('./loss.png')


# if __name__ == '__main__':
parser = argparse.ArgumentParser()
# input parameters
# parser.add_argument('--train_img_path', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/train26weeks')
# parser.add_argument('--val_img_path', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/val26weeks')
# parser.add_argument('--test_img_path', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/test26weeks')
# parser.add_argument('--train_csv_file', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/train26weeks.csv')
# parser.add_argument('--val_csv_file', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/val26weeks.csv')
# parser.add_argument('--test_csv_file', type=str,
#                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/test26weeks.csv')

parser.add_argument('--train_img_path', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/train26weeks_f')
parser.add_argument('--val_img_path', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/val26weeks_f')
parser.add_argument('--test_img_path', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/test26weeks_f')
parser.add_argument('--train_csv_file', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/train26weeks_f.csv')
parser.add_argument('--val_csv_file', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/val26weeks_f.csv')
parser.add_argument('--test_csv_file', type=str,
                    default='/Users/jjc/Research/SpineAI/Images/test26weeks_f.csv')

# training parameters
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--conv_base_lr', type=float, default=.0000001)
parser.add_argument('--dense_lr', type=float, default=.0000001)
# parser.add_argument('--conv_base_lr', type=float, default=.00001)
# parser.add_argument('--dense_lr', type=float, default=.00001)
parser.add_argument('--lr_decay_rate', type=float, default=0.95)
parser.add_argument('--lr_decay_freq', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)

# misc
parser.add_argument('--ckpt_path', type=str,
                    default='/Users/jjc/Research/CNOC_Spine/All/ckpts/')
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--gpu_ids', type=list, default=None)
parser.add_argument('--warm_start', type=bool, default=False)
parser.add_argument('--warm_start_epoch', type=int, default=216)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--save_fig', type=bool, default=True)

config, unknown = parser.parse_known_args()

source = ColumnDataSource(data={'epochs': [],
                                'trainlosses': [],
                                'vallosses': []}
                          )
plot = figure()
plot.line(x='epochs', y='trainlosses',
            color='green', alpha=0.8, line_width=2, legend_label="Training",
            source=source)
plot.line(x='epochs', y='vallosses',
            color='red', alpha=0.8, line_width=2, legend_label="Validation",
            source=source)

plot.xaxis.axis_label = "Epoch"
plot.yaxis.axis_label = "BCE Loss"

# Displaying the legend
plot.legend.location = "top_right"
doc = curdoc()
doc.add_root(plot)

thread = Thread(target=main, args=(config,))
thread.start()
