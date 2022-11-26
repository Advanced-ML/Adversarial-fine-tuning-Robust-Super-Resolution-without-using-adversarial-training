import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch import linalg as LA
from Losses import losses as ls 

#--------------------------------------PGD-----------------------------------------------------------
def gradiente(x_adv_aux, step_size, grad_norm):
    with torch.no_grad():
        gradient_num = x_adv_aux.grad * step_size.view(-1,1,1,1)
        gradient_den = LA.norm(x_adv_aux.grad.view(x_adv_aux.shape[0], -1), grad_norm, dim=-1) + 1e-12
        gradient = gradient_num * ((1/gradient_den).view(-1,1,1,1))
    return gradient

def make_step_SwinIR(x_adversarial, model, loss_fn, y, step_size, grad_norm,window_size, border, scale):
    x_adv_aux = x_adversarial.clone().detach().requires_grad_(True)
    x_adversarial_aux = x_adversarial.clone().detach()
    _, _, h_old, w_old = x_adv_aux.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    x_adv_aux_1 = torch.cat([x_adv_aux, torch.flip(x_adv_aux, [2])], 2)[:, :, :h_old + h_pad, :]
    x_adv_aux_2 = torch.cat([x_adv_aux_1, torch.flip(x_adv_aux_1, [3])], 3)[:, :, :, :w_old + w_pad]
    prediction = model(x_adv_aux_2)
    prediction = prediction[..., :h_old * scale, :w_old * scale]
    loss = loss_fn(prediction, y, crop_border=border)
    loss.backward()
    gradient = gradiente(x_adv_aux, step_size, grad_norm)
    x_adversarial_aux += gradient
    return x_adversarial_aux.clone().detach(), loss.detach().requires_grad_(False)

def projection(x, x_adversarial, eps, ball_norm):
    if ball_norm == float('inf'):
        x_adversarial = torch.max(torch.min(x_adversarial, x + eps), x - eps)
    else: 
        delta = x_adversarial - x
        norms = delta.view(delta.shape[0], -1).norm(ball_norm, dim=1)
        outside_deltas = norms <= eps
        norms[outside_deltas] = eps
        delta = (delta/norms.view(-1,1,1,1) + 1e-12) * eps
        x_adversarial = x + delta
    return x_adversarial



def pgd_SwinIR(model, x, y, loss_fn, num_steps, step_size, grad_norm, eps, ball_norm, window_size, border, scale, clamp=(0,1)):
    x_adversarial = x.clone().detach().requires_grad_(False).to(x.device)
    step_counter = 0
    step_size = torch.ones(x.shape[0]).to(x.device) * step_size
    x_max = None
    loss_max = None
    while True:
        x_adversarial_0 = x_adversarial.clone().detach().requires_grad_(False).to(x.device)
        x_adversarial, loss = make_step_SwinIR(x_adversarial, model, loss_fn, y, step_size, grad_norm,window_size, border, scale)
        x_adversarial = projection(x, x_adversarial, eps, ball_norm)
        step_counter += 1
        x_adversarial = x_adversarial.clamp(*clamp)
        if loss_max == None: 
            x_max = x_adversarial_0
            loss_max = loss
        else:
            x_max = x_max * ((loss_max>=loss).view(-1,1,1,1)) + x_adversarial_0 * ((loss_max<loss).view(-1,1,1,1))
            loss_max[loss>loss_max] = loss[loss>loss_max]
        if step_counter == num_steps:
            break

    return x_max.detach()


def pgd_with_momentum_SwinIR(model, x, y, loss_fn, num_steps, step_size, grad_norm, eps, ball_norm, window_size, border, scale, clamp=(0,1), alpha = 0.75):
    x_adversarial_0 = x.clone().detach().requires_grad_(False).to(x.device)
    step_size = torch.ones(x.shape[0]).to(x.device) * step_size
    _, _, h_old, w_old = x_adversarial_0.size()
    x_adversarial_1, loss_0 = make_step_SwinIR(x_adversarial_0, model, loss_fn,  y, step_size, grad_norm,window_size, border, scale)
    x_adversarial_1 = projection(x, x_adversarial_1, eps, ball_norm)
    x_adversarial_2 = None
    x_max = None
    x_max_0 = None
    step_counter = 1
    pred_1 = model(x_adversarial_1)
    pred_1 = pred_1[..., :h_old * scale, :w_old * scale]
    loss_1 = loss_fn(pred_1, y)
    loss_1 = loss_1.data
    loss1_ = loss_1.clone().detach()
    loss_max = loss_0.clone().requires_grad_(False)
    loss_max[loss_0<loss_1] = loss1_[loss_0<loss_1]
    x_max = x_adversarial_0 * ((loss_0>loss_1).view(-1,1,1,1)) + x_adversarial_1 * ((loss_0<loss_1).view(-1,1,1,1))
    while True:
        z, loss_1 = make_step_SwinIR(x_adversarial_1, model, loss_fn,  y, step_size, grad_norm, window_size, border, scale)
        z = projection(x, z, eps, ball_norm)
        x_adv_2_aux = x_adversarial_1 + (alpha*(z - x_adversarial_1)) + ((1-alpha)*(x_adversarial_1 -  x_adversarial_0))
        x_adversarial_2 = projection(x, x_adv_2_aux, eps, ball_norm)
        x_adversarial_2 = x_adversarial_2.clamp(*clamp)
        if x_max_0 == None:
            loss_max = loss_1.clone().detach()
            x_max = x_adversarial_1.clone().requires_grad_(False)
            x_max_0 = x_adversarial_0.clone().requires_grad_(False)
        else:
            x_max = x_max * ((loss_max>=loss_1).view(-1,1,1,1)) + x_adversarial_1 * ((loss_max<loss_1).view(-1,1,1,1))
            x_max_0 = x_max_0 * ((loss_max>=loss_1).view(-1,1,1,1)) + x_adversarial_0 * ((loss_max<loss_1).view(-1,1,1,1))
        loss1_ = loss_1.clone().detach()
        loss_max[loss_1>loss_max] = loss1_[loss_1>loss_max]

        x_adversarial_0 = x_adversarial_1.clone().detach()
        x_adversarial_1 = x_adversarial_2.clone().detach()
        loss_0 = loss_1.clone().detach()
        step_counter += 1
        if step_counter == num_steps:
            break
    return x_max.detach()
