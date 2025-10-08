import argparse
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_toothsegmdataset import ToothSegmDataset
from datasets.mask_visualization import show_anns
from utils import test_single_volume_for_tooth
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/ToothSegmDataset/trainset_valset', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='ToothSegmDataset', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./data/ToothSegmDataset/trainset_valset', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # CHANGE: Use val here
    db_test = args.Dataset(base_dir=args.volume_path, split="val")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name, tooth_id = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'], sampled_batch['tooth_id']
        metric_i = test_single_volume_for_tooth(args, image, label, model, classes=args.num_classes,
                                                patch_size=[args.img_size, args.img_size], test_save_path=test_save_path,
                                                case=case_name, tooth_id=tooth_id, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        # dice, hd95, mpa, iou_fg, iou_bg, miou
        case_metrics = np.mean(metric_i, axis=0)
        logging.info(
            'idx %d case %s mean_dice %.4f mean_hd95 %.4f mPA %.4f IoU_FG %.4f IoU_BG %.4f mIoU %.4f' %
            (i_batch, case_name, case_metrics[0], case_metrics[1], case_metrics[2],
             case_metrics[3], case_metrics[4], case_metrics[5])
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            'Mean class %d dice %.4f hd95 %.4f mPA %.4f IoU_FG %.4f IoU_BG %.4f mIoU %.4f' %
            (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2],
             metric_list[i - 1][3], metric_list[i - 1][4], metric_list[i - 1][5])
        )
    mean_metrics = np.mean(metric_list, axis=0)
    performance, mean_hd95 = mean_metrics[0], mean_metrics[1]
    mean_mpa, mean_iou_fg, mean_iou_bg, mean_miou = mean_metrics[2], mean_metrics[3], mean_metrics[4], mean_metrics[5]
    logging.info(
        'Testing performance in best val model: mean_dice: %.4f mean_hd95: %.4f mean_mPA: %.4f mean_IoU_FG: %.4f mean_IoU_BG: %.4f mean_mIoU: %.4f' %
        (performance, mean_hd95, mean_mpa, mean_iou_fg, mean_iou_bg, mean_miou)
    )
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'ToothSegmDataset': {
            'Dataset': ToothSegmDataset,
            'volume_path': './data/ToothSegmDataset/trainset_valset',
            'list_dir': './data/ToothSegmDataset/trainset_valset',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    if not os.path.exists(snapshot):
        raise Exception(f"You need to train the model first! {snapshot}")
    print('loading model from %s' % snapshot)
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = f'./test_log/{dataset_name}/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # inference(args, net, test_save_path)

    # Output testset mask.jpg & visualization
    test_path = './data/ToothSegmDataset/testset'
    save_path = './data/ToothSegmDataset/testset_pred'
    os.makedirs(save_path, exist_ok=True)

    net.eval()
    with torch.no_grad():
        for tooth_id in os.listdir(test_path):
            tooth_dir = os.path.join(test_path, tooth_id)
            if not os.path.isdir(tooth_dir):
                continue

            os.makedirs(os.path.join(save_path, tooth_id), exist_ok=True)
            rgb_files = sorted([f for f in os.listdir(tooth_dir) if f.endswith("_rgb.jpg")])
            for rgb_file in rgb_files:
                # Read and prepare image (BGR -> RGB)
                img_path = os.path.join(tooth_dir, rgb_file)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img_rgb.shape[:2]

                # Resize to model expected size if needed
                if (orig_h, orig_w) != (args.img_size, args.img_size):
                    proc_img = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_img = img_rgb

                # To tensor
                tensor = torch.from_numpy(proc_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()

                # Forward
                output = net(tensor)  # [1, num_classes, H, W]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

                # Restore to original size if resized
                if (orig_h, orig_w) != (args.img_size, args.img_size):
                    pred = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # Save mask
                mask_img = pred.astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, tooth_id, rgb_file.replace('_rgb.jpg', '_mask.jpg')), mask_img)

                # Visualization using original RGB image
                vis_img = show_anns(img_rgb.astype(np.uint8), pred, tooth_id=int(tooth_id), borders=True)
                plt.figure(figsize=(5, 5))
                plt.title(f"Visualization for Tooth ID {tooth_id}")
                plt.imshow(vis_img.astype(np.uint8))
                plt.axis('off')
                plt.savefig(os.path.join(save_path, tooth_id, rgb_file.replace('_rgb.jpg', '_vis.jpg')))
                plt.close()

                # Visualize original image and prediction side by side
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Image")
                plt.imshow(img_rgb.astype(np.uint8))
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("Prediction")
                plt.imshow(pred.astype(np.uint8), cmap='jet', vmin=0, vmax=8)
                plt.axis('off')
                plt.savefig(os.path.join(save_path, tooth_id, rgb_file.replace('_rgb.jpg', '_pred.png')))
                plt.close()
