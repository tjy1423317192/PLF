from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import torch.nn.functional as F


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class PLF(nn.Module):
    """PLF adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, num_classes=1000,
                 momentum=0.999, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        self.use_quantile = True
        self.clip_thresh = True
        self.p_model = torch.ones(self.num_classes) / self.num_classes
        self.label_hist = torch.ones(self.num_classes) / self.num_classes
        self.time_p = self.p_model.mean() + 0.2
        self.dataset_name = "cifar10_c"

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "plf requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.ap_pre = 0

    @torch.no_grad()
    def update(self, probs_x_ulb):

        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)

        if self.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs, 0.8)
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()

        if self.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.view(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1) if softmax_x_ulb else logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing        #model是学生  model_ema是教师  model_anhor是源模型
    def forward_and_adapt(self, x, model, optimizer):
        outputs_test = self.model(x)
        outputs_aug_test = self.model(self.transform(x))
        standard_ema = self.model_ema(x)
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        N = 32
        outputs_emas = []
        for i in range(N):
            outputs_ = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0) < self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        mask1 = self.masking(outputs_ema, softmax_x_ulb=True).unsqueeze(1)

        ap_alter = torch.nn.functional.softmax(outputs_ema + outputs_test, dim=1).max(1)[0].mean(0) - self.ap_pre
        self.ap_pre = torch.nn.functional.softmax(standard_ema, dim=1).max(1)[0].mean(0)
        decay_factor = torch.tensor(0.7)
        if ap_alter < 0:
            self.time_p = exponential_decay_threshold(self.time_p, ap_alter, decay_factor)

        loss1, _ = entropy_loss(mask1, outputs_test, self.p_model, self.label_hist)
        #         loss2 = (0.4 * symmetric_cross_entropy(outputs_test, outputs_ema * mask1) + 0.4 * symmetric_cross_entropy(outputs_aug_test, outputs_ema * mask1)).mean(0) + 0.2 * softmax_entropy(outputs_mask, outputs_ema).mean(0)
        loss2 = (0.5 * symmetric_cross_entropy(outputs_test, outputs_ema * mask1) + 0.5 * symmetric_cross_entropy(
            outputs_aug_test, outputs_ema * mask1)).mean(0)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        return outputs_ema + outputs_test


@torch.jit.script
def exponential_decay_threshold(original_threshold, confidence_change, decay_factor):
    # 计算新阈值
    new_threshold = original_threshold * torch.exp(-decay_factor * confidence_change)
    return new_threshold


def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def symmetric_cross_entropy(x, x_ema):  # -> torch.Tensor:
    return -(0.5) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s * mask

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:  # isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
