import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current_value: float):
        if self.best is None or current_value < self.best - self.min_delta:
            self.best = current_value
            self.num_bad_epochs = 0
            return True  # improved
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
            return False  # no improvement


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)  # TODO: Use args.seed


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # Early stopping init
    early_stopper = EarlyStopping(patience=100, min_delta=0.0)
    logging.info(f"Early stopping enabled (patience={early_stopper.patience}, "
                 f"min_delta={early_stopper.min_delta})")

    iter_num = 0
    best_loss = float("inf")
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # print(image_batch.shape, label_batch.shape)  # torch.Size([24, 1, 224, 224]) torch.Size([24, 224, 224])

            outputs = model(image_batch)

            # print(outputs.shape, label_batch.shape)  # torch.Size([24, 9, 224, 224]) torch.Size([24, 224, 224])

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            just_improved = early_stopper.step(loss)
            if just_improved:
                save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if early_stopper.should_stop:
                logging.info("Early stopping triggered. Stopping training.")
                break

        # epoch_avg_loss = epoch_loss_sum / max(1, epoch_batches)
        # writer.add_scalar('epoch/avg_loss', epoch_avg_loss, epoch_num)
        # logging.info(f"Epoch {epoch_num} avg_loss={epoch_avg_loss:.6f}")
        #
        # improved = False
        # if epoch_avg_loss < best_loss:
        #     best_loss = epoch_avg_loss
        #     improved = True
        #     save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info(f"New best model saved to {save_mode_path}")
        #
        # just_improved = early_stopper.step(epoch_avg_loss)
        # if just_improved and not improved:
        #     save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if early_stopper.should_stop:
            logging.info("Early stopping triggered. Stopping training.")
            break

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_toothsegm(args, model, snapshot_path):
    from datasets.dataset_toothsegmdataset import ToothSegmDataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ToothSegmDataset(base_dir=args.root_path, split="train",
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = ToothSegmDataset(base_dir=args.root_path, split="val",
                              transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # Early stopping init
    early_stopper = EarlyStopping(patience=1000, min_delta=0.0)
    logging.info(f"Early stopping enabled (patience={early_stopper.patience}, "
                 f"min_delta={early_stopper.min_delta})")

    iter_num = 0
    best_loss = float("inf")

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # TODO: Ask why block: [23,0,0], thread: [755,0,0] Assertion `t >= 0 && t < n_classes` failed?
            # Check label values before moving to CUDA
            if not torch.all((label_batch >= 0) & (label_batch < num_classes)):
                print("Invalid label values found:", label_batch.unique())
                continue
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # print(image_batch.shape, label_batch.shape)  # torch.Size([24, 3, 224, 224]) torch.Size([24, 224, 224])

            outputs = model(image_batch)

            # print(outputs.shape, label_batch.shape)  # torch.Size([24, 4, 224, 224]) torch.Size([24, 224, 224])

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            # Get validation loss and use it for early stopping
            if iter_num % 5 == 0:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for j_batch, val_sampled_batch in enumerate(valloader):
                        val_image_batch, val_label_batch = val_sampled_batch['image'], val_sampled_batch['label']
                        val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()

                        val_outputs = model(val_image_batch)

                        val_loss_ce = ce_loss(val_outputs, val_label_batch[:].long())
                        val_loss_dice = dice_loss(val_outputs, val_label_batch, softmax=True)
                        val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice

                        val_loss_sum += val_loss.item()
                        val_batches += 1

                avg_val_loss = val_loss_sum / max(1, val_batches)
                writer.add_scalar('info/val_loss', avg_val_loss, iter_num)
                logging.info(f'Validation loss after {iter_num} iterations: {avg_val_loss:.6f}')
                model.train()

                just_improved = early_stopper.step(avg_val_loss)
                if just_improved:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                if early_stopper.should_stop:
                    logging.info("Early stopping triggered. Stopping training.")
                    break

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        if early_stopper.should_stop:
            logging.info("Early stopping triggered. Stopping training.")
            break

    writer.close()
    return "Training Finished!"
