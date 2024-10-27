# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    # for i, (sampled_batch, names) in enumerate(loader, 1):
    # for i, sampled_batch in enumerate(loader):
    for i, (sampled_batch, names) in enumerate(loader, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks = sampled_batch['image'], sampled_batch['label']




        images, masks = images.cuda(), masks.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        # preds = model(images)
        ds4,ds3,ds2,ds1,preds = model(images)   #######ds
        output_loss = criterion(preds, masks.float())  #output Loss
        ds4_loss = criterion(ds4,masks.float())
        ds3_loss = criterion(ds3,masks.float())
        ds2_loss = criterion(ds2,masks.float())
        ds1_loss = criterion(ds1,masks.float())

        # out_loss = output_loss+ds4_loss+ds3_loss+ds2_loss+ds1_loss  #Test_session_03.11_08h52。dice_pred 0.78279
        out_loss = output_loss + 0.4 * ds4_loss + 0.4 * ds3_loss + 0.1 * ds2_loss + 0.1 * ds1_loss  ## 0.7828??? #Test_session_03.11_09h17。dice_pred 0.7957148

        # test_session = "Test_session_03.11_20h58"测试0.7739
        # test_session = "Test_session_03.11_20h57"测试测试0.7739
        # out_loss = output_loss + 0.1 * ds4_loss + 0.1 * ds3_loss + 0.4 * ds2_loss + 0.4 * ds1_loss   # 0.76847??? ##UNet_encoder_D_avg_decoer_scSE1_ds  #Test_session_03.11_20h57，train0.8094测试0.7739


        # out_loss = output_loss + 0.25 * ds4_loss + 0.25 * ds3_loss + 0.25 * ds2_loss + 0.25 * ds1_loss   ##0.7810???  ##UNet_encoder_D_avg_decoer_scSE1_ds  #Test_session_03.11_21h20  dice:0.8032
        ##用UNet_encoder_D_decoer_scSE1_ds网络，深度监督损失系数都用0.25   #Test_session_03.11_21h40  训练0.8106测试0.7677033


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()


        # train_iou = 0
        train_iou = iou_on_batch(masks,preds)
        train_dice = criterion._show_dice(preds, masks.float())

        batch_time = time.time() - end
        # train_acc = acc_on_batch(masks,preds)
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)
    return average_loss, train_dice_avg
