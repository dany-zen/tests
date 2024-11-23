from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from networks_execution import *
from networks_models import *
from process_data import *

from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.pool import collect_client_diff_weights_pt
from flex.pool import fed_avg
from flex.pool import set_aggregated_diff_weights_pt
from flex.pool import deploy_server_model_pt
import numpy as np
from flexclash.data import data_poisoner, data_poisoner_all
from flexclash.model import model_poisoner, model_inference
from flex.data.lazy_indexable import LazyIndexable

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)

from my_poison_attacks import backdoorss, clean_label, dirty_label, colision_features
from my_models_attacks.inversefed_for_gradient_attacks import *
from my_models_attacks import inversed_gradient

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

flex_data, server_id = load_and_preprocess_horizontal(dataname = "mnist", trasnform = False, nodes = 2) #Carga de datos sin transform

label_base= 2
label_target = 3
dataname = "mnist"

@data_poisoner_all
def colision_poison(x_datas, y_datas):
    print("Entra aca")
    poison, labels_poison = colision_features.run_attack_caract_colision(x_datas, y_datas, dataset = dataname, pret = True, number_to_poison = 5, target_class = label_target
                                                 ,taget_class_ind = label_target - 1, base_class = label_base, n_class = 10, iter = 10, sim_coeff = 256, wmark = 0.3, lr = 1)
    return poison, labels_poison
num_labels_datas = 10

client_to_poison = list(flex_data.keys())

flex_data.apply(colision_poison, node_ids = client_to_poison)

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

    train_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())
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
    criterion = client_flex_model["criterion"]
    net_config.trainNetwork(local_epochs = 5, criterion = criterion, optimizer = optimizer,
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
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            #_, pred = output.max(dim=1)
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()#Ver esto
            #test_acc += (pred==target.data.view_as(pred)).sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

@deploy_server_model
def deploy_serv(server_flex_model: FlexModel): 

    gradients = []
    i=0
    for param in server_flex_model["model"].parameters():
        if param.grad is None:
            i+=1
    print(i)


    state_dict_save = deepcopy(server_flex_model["model"].state_dict())
    grad_save = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in server_flex_model["model"].named_parameters()}
    new_model = deepcopy(server_flex_model)
    new_model["model"].load_state_dict(state_dict_save)
    for name, param in new_model["model"].named_parameters():
        if name in grad_save:
            param.grad = grad_save[name]
    
    return new_model

def clean_up_models(client_model: FlexModel, _):
    import gc

    client_model.clear()
    gc.collect()

def train_n_rounds(n_rounds, clients_per_round=20):
    pool = FlexPool.client_server_pool(
        fed_dataset= flex_data, server_id=server_id, init_func=build_server_model
    )
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients

        #adv_clients_pool = selected_clients.select(n_malicius_client)
        #selected_adv_clients = adv_clients_pool.clients

        #print(f"Selected clients for this round: {len(selected_clients)}")
        #print(f"Selected clients for this round: {selected_clients._models.values()}")
        # Deploy the server model to the selected clients
        pool.servers.map(deploy_serv, selected_clients)
        # Each selected client trains her model
        selected_clients.map(train)
        #selected_adv_clients.map(execute_ownership_attack)#Este son los maliciosos, aquí viene el ataque

        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_client_diff_weights_pt, selected_clients)
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

#train_n_rounds(1, clients_per_round=1)