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
from flexclash.data import data_poisoner
from flexclash.model import model_poisoner, model_inference
from flex.data.lazy_indexable import LazyIndexable
from flex.actors.actors import FlexActors, FlexRoleManager
from flex.data import FedDataset
from art.estimators.classification import PyTorchClassifier
from my_models_attacks import tabular_atribute_inference as inf_att_tab
from my_models_attacks import membership_inference as mem_inf
from my_models_attacks import model_inversion as mod_inv
from my_models_attacks import param_manipulated_attacks as param_maniplt

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes = 5)

adv_data = flex_data[server_id]

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

adv_cl = [0]
@param_maniplt.manipulated_weights_attack
def inferencer(aggregated_weights_as_list: list): #aggregated_weights_as_list: list Esto va para el agregador

    data = adv_data.to_torchvision_dataset(transform = mnist_transform())
    criterion, model, optimizer = net_config.for_fd_server_model_config() #Para esto hay que hacer un modelo adversario configurable
    w_dict = model.state_dict()
    dataloader = DataLoader(
        data, batch_size = len(data), shuffle=True, pin_memory=False
    )
    data_list = []
    labels = []
    for data, label in tqdm(dataloader):
        data_list = np.array(data)
        labels = label.tolist()

    x = data_list
    print(x.shape)
    #y = np.array(labels, dtype = int)
    y = labels
    print(set(y))

    #x, y = data.to_numpy()
    if len(x[0].shape) == 2:
        x = np.expand_dims(x, axis = 1)
    elif len(x[0].shape) == 3 and 3 > x[0].shape[-1]:
        x = np.transpose(x, (0,3,1,2))
    sizes = x[0].shape
    n_labels = len(set(y))
    print(n_labels)
    num_instances = 10 #Esto cambia, lo puse pa que sea igual al # de clases
    chan = sizes[0]
    dimen1 = sizes[1]
    dimen2 = sizes[2]

    x_base = np.zeros((num_instances, chan, dimen1, dimen2)) + np.mean(x)
    #x_base = np.random.uniform(0, 1, (10, 28, 28, 1))
    y_pos_label = np.arange(n_labels)

    model = model
    optim = optimizer
    crit = criterion
    max = np.max(x)
    min = np.min(x)
    #x_base = np.random.uniform(0, 1, (num_instances, chan, dimen1, dimen2)) + np.mean(x)

    for client_index in range(len(aggregated_weights_as_list)):
        if client_index in adv_cl:
            client = aggregated_weights_as_list[client_index]
            for layer_key, new in zip(w_dict, client):
                try:
                    if len(new) != 0:
                        w_dict[layer_key].add_(new)
                except TypeError:
                    w_dict[layer_key].add_(new)
            model.load_state_dict(w_dict)
            #print(client)
            x_infer = mod_inv.mi_face_model_inversion(x_base, y_pos_label, n_labels, sizes, model, crit, optim, max, min)
    return aggregated_weights_as_list
"""""
    x, y = data.to_numpy()
    if len(x[0].shape) == 2:
        x = np.expand_dims(x, axis = 1)
    elif len(x[0].shape) == 3 and 3 > x[0].shape[-1]:
        x = np.transpose(x, (0,3,1,2))
    sizes = x[0].shape
    n_labels = len(set(y))
    num_instances = 10 #Esto cambia, lo puse pa que sea igual al # de clases
    chan = sizes[0]
    dimen1 = sizes[1]
    dimen2 = sizes[2]

    x_base = np.zeros((num_instances, chan, dimen1, dimen2)) + np.mean(x, axis=0)
    y_pos_label = np.arange(n_labels)

    model = client_model["model"]
    optim = client_model["optimizer_func"]
    crit = client_model["criterion"]
    max = np.max(x)
    min = np.min(x)

    x_infer = mod_inv.mi_face_model_inversion(x_base, y_pos_label, n_labels, sizes, model, crit, optim, max, min)
"""
    #return client_model

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


        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_client_diff_weights_pt, selected_clients)
        pool.aggregators.map(inferencer)
        pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_diff_weights_pt, pool.servers)
        # Optional: evaluate the server model
        #pool.servers.map(model_extractor)
        metrics = pool.servers.map(evaluate_global_model)
        #pool.servers.map(inferencer)
        #metricsB = pool.servers.map(evaluate_global_model_bad_data)
        # Optional: clean-up unused memory
        selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
        #lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        #print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(1, clients_per_round = 1)