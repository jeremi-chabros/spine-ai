# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import torchmetrics

from data_loader import ScanDataset
from model import NIMA
from gradcam.utils import visualize_cam
from gradcam import GradCAM

from sklearn.calibration import calibration_curve


def overlay_images(original, cam_output, alpha=0.5):
    return (alpha * original) + ((1 - alpha) * cam_output)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()])

    base_model = models.resnet50(weights="DEFAULT", progress=False)
    model = NIMA(base_model)

    # Load pretrained model if required
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(
            config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' %
              config.warm_start_epoch)

    model = model.to(device)

    if config.test:
        print('Testing')

        # Load model
        model_path = os.path.join(
            config.ckpt_path, f'epoch-{config.warm_start_epoch}.pkl')
        model.load_state_dict(torch.load(model_path))

        # Set target layer
        target_layer = model.features

        # Prepare test dataset and loader
        testset = ScanDataset(csv_file=config.test_csv_file,
                              root_dir=config.test_img_path,
                              transform=val_transform)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        model.eval()
        gradcam = GradCAM(model, target_layer)

        ypreds = []
        ylabels = []
        im_ids = []
        count = 0

        for data in test_loader:
            im_id = data['img_id']
            im_name = os.path.split(im_id[0])
            base_name = os.path.splitext(im_name[1])
            image = data['image'].to(device)
            labels = data['annotations'].to(device).long()

            output = model(image)
            output = output.view(-1, 2)
            bpred = output.to(torch.device("cpu"))
            cpred = bpred.data.numpy()
            blabel = labels.to(torch.device("cpu"))
            clabel = blabel.data.numpy()

            # pred_rd = cpred[0]
            # label_rd = clabel[0]

            prediction = [cpred[0][0], cpred[0][1]]
            labl = [clabel[0][0][0], clabel[0][1][0]]

            # pred_dd = np.argmax(prediction, axis=0)
            # label_dd = np.argmax(labl, axis=0)

            threshold = 0.25
            if prediction[1] >= threshold:
                pred_dd = 1  # Positive
            else:
                pred_dd = 0  # Negative

            label_dd = np.argmax(labl, axis=0)

            # Determine TP, TN, FP, FN
            if pred_dd == label_dd:
                if pred_dd == 0:
                    addon = "TP"  # True Positive
                else:
                    addon = "TN"  # True Negative
            else:
                if pred_dd == 0:
                    addon = "FP"  # False Positive
                else:
                    addon = "FN"  # False Negative

            # GradCAM visualization
            # mask, _ = gradcam(image, success)
            # heatmap, cam_result = visualize_cam(mask, image)
            # overlaid_image = overlay_images(image.squeeze(0).cpu(), cam_result.squeeze(0).cpu())
            # im = transforms.ToPILImage()(overlaid_image)

            # UNCOMMENT
            if config.generate_heatmaps:
                mask, _ = gradcam(image)
                heatmap, result = visualize_cam(mask, image)
                im = transforms.ToPILImage()(result)

                # mask, _ = gradcam(image)
                # heatmap, cam_result = visualize_cam(mask, image)
                # overlaid_image = overlay_images(image.squeeze(0).cpu(), cam_result.squeeze(0).cpu())
                # im = transforms.ToPILImage()(overlaid_image)

                im.save(
                    f"/Users/jjc/Research/CNOC_Spine/All/heatmaps/{base_name[0]}_{addon}.jpg")

            ypreds.append(cpred)
            ylabels.append(clabel)
            im_name = os.path.split(im_id[0])
            im_ids.append(im_name[1])
            count = count+1

    # Convert predictions and labels to tensors
    apreds = torch.Tensor(ypreds).squeeze()
    alabels = torch.Tensor(ylabels).squeeze()

    # Set values below the threshold to a small number, so they won't be chosen by argmax
    # threshold = 0.23
    threshold = 0.1
    apreds[apreds[:, 1] >= threshold, 1] = float('inf')
    apreds[apreds[:, 1] < threshold, 0] = float('inf')
    pred_labels = torch.logical_not(torch.argmax(apreds, dim=1))
    # For true labels, you probably don't need thresholding.
    true_labels = torch.logical_not(torch.argmax(alabels, dim=1))

    # pred_labels = torch.argmax(apreds, dim=1)
    # true_labels = torch.argmax(alabels, dim=1)

    # df = pd.DataFrame(data={'Label': true_labels, "Predict": pred_labels})
    df = pd.DataFrame(data={'Label': true_labels,
                      'Predict': pred_labels, 'FileName': im_ids})
    df.to_csv("./results.csv")

    TP = ((pred_labels == 1) & (true_labels == 1)).float().sum().item()
    TN = ((pred_labels == 0) & (true_labels == 0)).float().sum().item()
    FP = ((pred_labels == 1) & (true_labels == 0)).float().sum().item()
    FN = ((pred_labels == 0) & (true_labels == 1)).float().sum().item()

    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    tasktype = "binary"
    acc = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=2)
    spec = torchmetrics.classification.Specificity(task=tasktype)
    f1 = torchmetrics.classification.F1Score(task=tasktype)
    bprecision = torchmetrics.classification.Precision(task=tasktype)
    brec = torchmetrics.classification.Recall(task=tasktype)
    auroc = torchmetrics.classification.AUROC(task=tasktype)
    broc = torchmetrics.classification.ROC(task=tasktype)
    cm = torchmetrics.ConfusionMatrix(task=tasktype)
    prcurve = torchmetrics.classification.BinaryPrecisionRecallCurve()

    # Compute metrics
    print(f'Accuracy: {acc(pred_labels, true_labels):.4f}')
    print(f'Specificity: {spec(pred_labels, true_labels):.4f}')
    print(f'F1 Score: {f1(pred_labels, true_labels):.4f}')
    print(f'Precision: {bprecision(pred_labels, true_labels):.4f}')
    print(f'Recall: {brec(pred_labels, true_labels):.4f}')
    print(f'AUC-ROC: {auroc(apreds, alabels):.4f}')

    # Plot ROC
    broc.update(apreds, alabels.long())
    fig, ax = broc.plot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/ROC.svg",
                dpi=300, bbox_inches='tight')

    # Plot Confusion Matrix
    # cm.update(pred_labels, true_labels)
    cm.update(true_labels, pred_labels)

    fig, ax = cm.plot()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/confusion_matrix.svg",
                dpi=300, bbox_inches='tight')

    prcurve.update(apreds, alabels.long())
    fig, ax = prcurve.plot(score=True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    fig.savefig("/Users/jjc/Research/CNOC_Spine/All/PR_curve.svg",
                dpi=300, bbox_inches='tight')

    def plot_calibration_curve(true_labels, predicted_probs, n_bins=5):
        prob_true, prob_pred = calibration_curve(
            true_labels, predicted_probs, n_bins=n_bins)

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.savefig("calibration_curve.png", dpi=300, bbox_inches='tight')
        # plt.show()

    apreds = torch.Tensor(ypreds).squeeze()
    alabels = torch.Tensor(ylabels).squeeze()
    preds = (apreds[:, 0]).numpy()
    labels = (alabels[:, 0]).numpy()

    # print(preds)
    # print(labels)
    plot_calibration_curve(labels, preds)

    dist = []
    for i in np.linspace(0, 1, 21):
        apreds = torch.Tensor(ypreds).squeeze()
        alabels = torch.Tensor(ylabels).squeeze()
        threshold = i
        apreds[apreds[:, 1] >= threshold, 1] = float('inf')
        apreds[apreds[:, 1] < threshold, 0] = float('inf')
        pred_labels = torch.logical_not(torch.argmax(apreds, dim=1))
        true_labels = torch.logical_not(torch.argmax(alabels, dim=1))
        TP = ((pred_labels == 1) & (true_labels == 1)).float().sum().item()
        TN = ((pred_labels == 0) & (true_labels == 0)).float().sum().item()
        FP = ((pred_labels == 1) & (true_labels == 0)).float().sum().item()
        FN = ((pred_labels == 0) & (true_labels == 1)).float().sum().item()
        dist.append(sqrt((1 - (TP/(TP+FN)))**2 + (FP/(FP + TN))**2))

    udf = np.linspace(0, 1, 21)
    mv = dist.index(min(dist))
    print(udf[mv])
    # print(dist)
    # print(dist.index(min(dist)))
    # print(min(dist))

    plt.figure(figsize=(6, 6))
    plt.plot(np.linspace(0, 1, 21), dist, marker='o', linewidth=1)
    plt.xlabel('Probability threshold')
    plt.ylabel('Distance from perfect classifier')
    plt.title('Distance-based Threshold Optimization')
    plt.legend()
    plt.savefig("dt_curve.png", dpi=300, bbox_inches='tight')

    # Save results
    np.savez('test_results.npz', Label=ylabels, Predict=ypreds)
    df = pd.DataFrame(data={'Label': ylabels, "Predict": ypreds})
    print(df.dtypes)
    df.to_pickle("./test_results.pkl")
    df.to_csv("./test_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters

    # parser.add_argument('--train_img_path', type=str,
    #                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/all')
    # parser.add_argument('--val_img_path', type=str,
    #                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/all')
    # parser.add_argument('--test_img_path', type=str,
    #                     default='/Users/jjc/Research/CNOC_Spine/All/imgs/all')

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

    # parser.add_argument('--train_img_path', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/all_copy')
    # parser.add_argument('--val_img_path', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/all_copy')
    # parser.add_argument('--test_img_path', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/all_copy')

    # parser.add_argument('--train_csv_file', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/train26weeks.csv')
    # parser.add_argument('--val_csv_file', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/val26weeks.csv')
    # parser.add_argument('--test_csv_file', type=str,
    #                     default='/Users/jjc/Research/SpineAI/Images/test26weeks.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=.001)
    parser.add_argument('--dense_lr', type=float, default=.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str,
                        default='/Users/jjc/Research/CNOC_Spine/All/ckpts/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=216)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=True)
    parser.add_argument('--generate_heatmaps', type=bool, default=False)

    config, unknown = parser.parse_known_args()
    main(config)
