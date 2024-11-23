from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from networks_execution import *
from networks_models import *
from process_data import *

from globals_for_inference import *
import globals_for_inference as globals_inf


from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.pool import collect_client_diff_weights_pt
from flex.pool import fed_avg
from flex.pool import set_aggregated_diff_weights_pt
from flex.pool import deploy_server_model_pt
import numpy as np
from flexclash.data import data_poisoner
from flexclash.model import model_poisoner, model_inference
from flex.data.lazy_indexable import LazyIndexable
from flex.actors.actors import FlexActors, FlexRoleManager
from flex.data import FedDataset

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)

from my_poison_attacks import backdoorss, clean_label, dirty_label
from my_models_attacks.inversefed_for_gradient_attacks import *
from my_models_attacks import inversed_gradient

from my_models_attacks.moda import *
import my_models_attacks.ownershps as ow_atck
from my_models_attacks import label_inference as li
from wmDataset import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes = 5)


net_config = ExecutionNetwork()



@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    criterion, model, optimizer = net_config.for_fd_server_model_config()

    server_flex_model["model"] = model.to(device)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = criterion
    server_flex_model["optimizer_func"] = optimizer
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model

def train(client_flex_model: FlexModel, client_data: Dataset):

    train_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())#mnist_transform()
    client_dataloader = DataLoader(train_dataset, batch_size = 256)
    model = client_flex_model["model"]

    state_dict_v = deepcopy(model.state_dict())
    gradients = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in model.named_parameters()}


    model = model.to(device)
    client_flex_model["previous_model"] = deepcopy(
        model
    )  # Required to use `collect_client_diff_weights_pt` primitive

    client_flex_model["previous_model"].load_state_dict(state_dict_v)
    for name, param in client_flex_model["previous_model"].named_parameters():
        if name in gradients:
            param.grad = gradients[name]
    optimizer = client_flex_model["optimizer_func"]


    #optimizer.param_groups[0]['params']= model.parameters() Esto hay que ver si trabaja con los parametros que agrega o si no
    criterion = client_flex_model["criterion"]
    net_config.trainNetwork(local_epochs = 1, criterion = criterion, optimizer = optimizer,
                            momentum = 0.9, lr = 0.005, trainloader = client_dataloader, testloader= None, 
                            model=model)


    return client_flex_model

def train_adv_malicius(client_flex_model: FlexModel, client_data: Dataset):

    train_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())#mnist_transform()
    client_dataloader = DataLoader(train_dataset, batch_size = 256)
    model = client_flex_model["model"]

    state_dict_v = deepcopy(model.state_dict())
    gradients = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in model.named_parameters()}


    model = model.to(device)
    client_flex_model["previous_model"] = deepcopy(
        model
    )  # Required to use `collect_client_diff_weights_pt` primitive

    client_flex_model["previous_model"].load_state_dict(state_dict_v)
    for name, param in client_flex_model["previous_model"].named_parameters():
        if name in gradients:
            param.grad = gradients[name]
    optimizer = client_flex_model["optimizer_func"]


    #optimizer.param_groups[0]['params']= model.parameters() Esto hay que ver si trabaja con los parametros que agrega o si no
    criterion = client_flex_model["criterion"]
    global grad_per_data_samples
    net_config.trainAdv(local_epochs = 1, criterion = criterion, optimizer = optimizer,
                            momentum = 0.9, lr = 0.005, trainloader = client_dataloader, testloader= None, 
                            model=model)


    return client_flex_model

def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):#falta poner esto
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform = mnist_transform())
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target.long()).item())
            pred = output.data.max(1, keepdim=True)[1]
            #_, pred = output.max(dim=1)
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()#Ver esto
            #test_acc += (pred==target.data.view_as(pred)).sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

def clean_up_models(client_model: FlexModel, _):
    import gc

    client_model.clear()
    gc.collect()

@deploy_server_model
def deploy_serv(server_flex_model: FlexModel): 

    gradients = []
    i=0
    for param in server_flex_model["model"].parameters():
        if param.grad is None:
            i+=1
    #print(i)


    state_dict_save = deepcopy(server_flex_model["model"].state_dict())
    grad_save = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in server_flex_model["model"].named_parameters()}
    new_model = deepcopy(server_flex_model)
    new_model["model"].load_state_dict(state_dict_save)
    for name, param in new_model["model"].named_parameters():
        if name in grad_save:
            param.grad = grad_save[name]
    
    
    return new_model

@model_inference
def label_inference(client_model: FlexModel, client_data: Dataset):
    model_client= deepcopy(client_model["model"])
    test_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())
    dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=True, pin_memory=False
    )
    for data, target in tqdm(dataloader):
        li.label_inference_attack_model(model_client, client_model["criterion"],data, target)
    
    return client_model

n_malicius_client = 1
def train_n_rounds(n_rounds, clients_per_round=20):
    pool = FlexPool.client_server_pool(
        fed_dataset= flex_data, server_id=server_id, init_func=build_server_model
    )
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients

        adv_clients_pool = selected_clients.select(n_malicius_client)
        selected_adv_clients = adv_clients_pool.clients

        #print(f"Selected clients for this round: {len(selected_clients)}")
        #print(f"Selected clients for this round: {selected_clients._models.values()}")
        # Deploy the server model to the selected clients
        pool.servers.map(deploy_serv, selected_clients)

        global grad_per_data_samples
        global data_samples
        global target_samples

        for idc in selected_adv_clients.actor_ids:
            print("Cliente: ",idc)
            globals_inf.grad_per_data_samples [idc] = []
            globals_inf.data_samples [idc] = []
            globals_inf.target_samples [idc] = []
        
        test_actors = FlexActors()
        test_models= {}
        test_new_data = FedDataset()

        for actor_ids in selected_clients.actor_ids:
            if actor_ids not in selected_adv_clients:
                test_actors[actor_ids] = selected_clients._actors[actor_ids]
                test_models[actor_ids] = selected_clients._models[actor_ids]
                test_new_data[actor_ids] = selected_clients._data[actor_ids]
        test_clients_pool = FlexPool(flex_actors = test_actors, flex_data = test_new_data, flex_models = test_models)
        test_clients = test_clients_pool.clients

        #for i in selected_clients._actors:
        #    print("Para comprobar",i)

        # Each selected client trains her model
        test_clients.map(train)
        selected_adv_clients.map(train_adv_malicius)
        selected_adv_clients.map(label_inference)

        #for id, value in main_test.grad_per_data_samples.items():
        #    print("Id", id, "Cantidad de batch en grads", len(main_test.grad_per_data_samples[id]))
        #    print("Id", id, "Cantidad de batch en datas", len(main_test.data_samples[id]))
        #    print("Id", id, "Cantidad de batch en targets", len(main_test.target_samples[id]))

        actors = FlexActors()
        models= {}
        new_data = FedDataset()

        for actor_ids in selected_clients.actor_ids:
            #print("Para comprobar 2:",actor_ids)
            if actor_ids in selected_adv_clients.actor_ids:
                actors[actor_ids] = selected_adv_clients._actors[actor_ids]
                models[actor_ids] = selected_adv_clients._models[actor_ids]
                new_data[actor_ids] = selected_adv_clients._data[actor_ids]
            elif actor_ids in test_clients.actor_ids:
                actors[actor_ids] = test_clients._actors[actor_ids]
                models[actor_ids] = test_clients._models[actor_ids]
                new_data[actor_ids] = test_clients._data[actor_ids]
        new_pool = FlexPool(flex_actors = actors, flex_data = new_data, flex_models = models)
        clients_to_agregate = new_pool.clients

        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_client_diff_weights_pt, clients_to_agregate)
        pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_diff_weights_pt, pool.servers)
        # Optional: evaluate the server model
        metrics = pool.servers.map(evaluate_global_model)
        #metricsB = pool.servers.map(evaluate_global_model_bad_data)
        # Optional: clean-up unused memory
        selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
        #lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        #print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(1, clients_per_round = 2)
