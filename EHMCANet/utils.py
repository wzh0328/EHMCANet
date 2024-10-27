import numpy as np
from sklearn.metrics import roc_auc_score,jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from torch.optim.optimizer import Optimizer
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from scipy.ndimage import zoom
from medpy import metric
import SimpleITK as sitk



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    smooth = 1e-5
    boundary_IOU = 0
    for i in range(pred.squeeze().shape[0]):
        pred_boundary = mask_to_boundary(np.uint8(pred[i].squeeze()))
        gt_boundary = mask_to_boundary(np.uint8(gt[i].squeeze()))
        boundary_inter = np.sum(pred_boundary * gt_boundary)
        boundary_union = np.sum(pred_boundary + gt_boundary) - boundary_inter
        boundary_IOU += (boundary_inter + smooth) / (boundary_union + smooth) / pred.squeeze().shape[0]
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95, boundary_IOU
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, boundary_IOU
    else:
        return 0, 0, boundary_IOU

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def _adaptive_size(target):
    target = torch.from_numpy(target).float()
    kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
    padding_out[:, 1:-1, 1:-1] = target
    h, w = 3, 3

    Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1))
    for i in range(Y.shape[0]):
        Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),
                                  padding=1)

    Y = Y * target
    Y[Y == 5] = 0
    C = torch.count_nonzero(Y)
    S = torch.count_nonzero(target)
    smooth = 1e-5
    return (C + smooth)/(S + smooth).item()

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
    big_list = []
    small_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        if _adaptive_size(label == i) >= 0.2:
            small_list.append(metric_list[-1][0])
        else:
            big_list.append(metric_list[-1][0])

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list, big_list, small_list
    # return L1,L2,L3,L4

class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(),
                                      padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha,
                    0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


class TverskyLoss(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_true, alpha=0.7, beta=0.3):

        y_pred = torch.softmax(y_pred, dim=1)
        y_true = self._one_hot_encoder(y_true)
        loss = 0
        for i in range(1, self.classes):
            p0 = y_pred[:, i, :, :]
            ones = torch.ones_like(p0)
            #p1: prob that the pixel is of class 0
            p1 = ones - p0
            g0 = y_true[:, i, :, :]
            g1 = ones - g0
            #terms in the Tversky loss function combined with weights
            tp = torch.sum(p0 * g0)
            fp = alpha * torch.sum(p0 * g1)
            fn = beta * torch.sum(p1 * g0)
            #add to the denominator a small epsilon to prevent the value from being undefined
            EPS = 1e-5
            num = tp
            den = tp + fp + fn + EPS
            result = num / den
            loss += result
        return 1 - loss / self.classes



class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
                # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        outputs_soft = F.softmax(outputs, dim=1)
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:,self.idx,...]
        dc = torch.from_numpy(gt_sdf[:,self.idx,...]).cuda()
        multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss




class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        # print('logit00.shape',logit.shape)
        # print('truth00.shape', truth.shape)
        # logit = logit.mean(dim=1)
        # logit = logit.contiguous().view(4,-1)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        # print('logit11.shape', logit.shape)
        # print('truth11.shape', truth.shape)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


####
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
        target = torch.squeeze(target, dim=2)   #把target torch.Size([24, 1, 1, 224, 224])变成([24, 1, 224, 224])
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask = masks[i].cpu().detach().numpy()
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)

def iou_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    ious = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 1
        pred_tmp[pred_tmp<0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp>0] = 1
        mask_tmp[mask_tmp<=0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    dices = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 1
        pred_tmp[pred_tmp<0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp>0] = 1
        mask_tmp[mask_tmp<=0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)

def save_on_batch(images1, masks, pred, names, vis_path):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp>=0.5] = 255
        pred_tmp[pred_tmp<0.5] = 0
        mask_tmp[mask_tmp>0] = 255
        mask_tmp[mask_tmp<=0] = 0

        cv2.imwrite(vis_path+ names[i][:-4]+"_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path+names[i][:-4]+"_gt.jpg", mask_tmp)



class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
