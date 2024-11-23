import torch
import torchvision

import numpy as np
from PIL import Image

import my_models_attacks.inversefed_for_gradient_attacks as inversefed_for_gradient_attacks
from my_models_attacks.inversefed_for_gradient_attacks import reconstruction_algorithms

from collections import defaultdict
import datetime
import time
import os

from copy import deepcopy

def reconstruction_gradient_attack(client_model_act, server_model, dim_imgs, mean, std, num_images, labels):

    output = None
    stats = None

    gradients = [torch.clone(param.grad).detach() if param.grad is not None else torch.zeros_like(param) for param in client_model_act.parameters()]#No se si aquí unir el parámetro del cliente con el del server

    config = dict(signed=True,#Modificable
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=4000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')
    rec_machine = inversefed_for_gradient_attacks.GradientReconstructor(server_model, (mean, std), config, num_images=num_images)
    output, stats = rec_machine.reconstruct(gradients, labels, img_shape = dim_imgs) #Aqui el shape cambia segun el tamaño de las imagenes
    return output, stats

def reconstruction_gradient_attack_server(data_adv, server_model, loss_fn, local_lr, 
                                          local_steps, dim_imgs, mean, std, num_images):

    output = None
    stats = None

    ground_truth, labels = [], []
    for i in range(num_images):
        ground_truth.append(data_adv[i][0])
        labels.append(torch.as_tensor(data_adv[i][1],))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)

    server_model.zero_grad()
    target_loss, _, _ = loss_fn(server_model(ground_truth), labels)

    input_parameters = reconstruction_algorithms.loss_steps(server_model, ground_truth, labels, 
                                                        lr=local_lr, local_steps=local_steps,
                                                                   use_updates = True)

    input_parameters = [p.detach() for p in input_parameters]

    config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=1,
              optim='adam',
              restarts=8,
              max_iterations=24000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')
    rec_machine = inversefed_for_gradient_attacks.FedAvgReconstructor(server_model, (mean, std), local_steps, local_lr, config,
                                             use_updates=True, num_images=num_images)
    
    output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape = dim_imgs) #Aqui el shape cambia segun el tamaño de las imagenes
    return output, stats

def get_meanstd(trainset_ori):
    trainset = deepcopy(trainset_ori)
    cc = torch.cat([trainset[i][0].reshape(1, -1) for i in range(len(trainset))], dim=1)#Que las otras 2 dimensiones sean modificables
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()

    return data_mean, data_std