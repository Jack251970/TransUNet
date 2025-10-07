import numpy as np
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

from datasets.mask_visualization import show_anns


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


# image: B, H, W, C
# label: B, H, W
def test_single_volume_for_tooth(args, image, label, net, classes, patch_size=[256, 256], test_save_path=None,
                                 case=None, tooth_id=0, z_spacing=1):
    image = image.squeeze(0)  # H, W, C
    label = label.squeeze(0)  # H, W

    # Resize if needed
    x, y = image.shape[:2]
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=3)  # bilinear for RGB

    # Convert to torch tensors
    image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # C, H, W
    image = image.unsqueeze(0)  # 1, C, H, W

    # prediction
    # dice_loss = DiceLoss(args.num_classes)
    net.eval()
    out = None  # [224, 224]
    prediction = None  # [H, W]
    with torch.no_grad():
        input = image.float().cuda()  # [1, 3, 224, 224]
        outputs = net(input)  # [1, 5, 224, 224]
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  # [224, 224]
        out = out.cpu().detach().numpy()  # [224, 224]
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    prediction = prediction  # [H, W] ndarray
    label = label.cpu().detach().numpy()  # [H, W] ndarray

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

    # TODO: Test visualization
    # # Visualize
    # image = image.squeeze(0).permute(1, 2, 0).numpy()  # (H,W,C)
    # label = label  # (H,W)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.title("Image")
    # plt.imshow(image.astype(np.uint8))
    # plt.axis('off')
    # plt.subplot(1, 3, 2)
    # plt.title("Label")
    # plt.imshow(label, cmap='jet', vmin=0, vmax=8)
    # plt.axis('off')
    # plt.subplot(1, 3, 3)
    # plt.title("Prediction")
    # plt.imshow(prediction, cmap='jet', vmin=0, vmax=8)
    # plt.axis('off')
    # plt.show()
    #
    # # Visualize with show_anns
    # vis_img = show_anns(image.astype(np.uint8), out, tooth_id=tooth_id, borders=True)
    # plt.figure(figsize=(5, 5))
    # plt.title(f"Visualization for Tooth ID {tooth_id}")
    # plt.imshow(vis_img.astype(np.uint8))
    # plt.axis('off')
    # plt.show()

    return metric_list
