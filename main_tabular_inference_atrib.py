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
device

#flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=2)
flex_data, server_id = load_and_preprocess_horizontal_tabular(dataname="nursery", trasnform=False, nodes=2)
count=0
d=np.array(flex_data[0].y_data)
for i in range(len(d)):#Esto es para elminar la clase que es minoría, antes del entrenamiento, antes de convertir a tensor
    if d[i] == 2:
        print(i)
        count+=1
                  
print(set(np.array(flex_data[0].y_data)))
#print(len(flex_data[1].X_data))
#print(len(flex_data[server_id].X_data))

net_config = ExecutionNetwork()

@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    #criterion, model, optimizer = net_config.for_fd_server_model_config()
    model = build_model_for_tabular(in_put = 8, inner_put = 15,out_put = 4, dataname='nursery')

    server_flex_model["model"] = model.to(device)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = optim.Adam(model.parameters(), lr=0.001)
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model

def train(client_flex_model: FlexModel, client_data: Dataset):

    #train_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())#mnist_transform()
    #datas_x = np.array(client_data.X_data)
    #columns = datas_x.shape
    #min_value_for_attr=[np.min(datas_x, axis = ni) for ni in range (columns[0])]
    #max_value_for_attr=[np.max(datas_x, axis = ni) for ni in range (columns[0])]

    #train_dataset = transform_tabular(dataname="nursery", dataset = client_data)# Esto es para el entrenamiento sin utilizar la herramienta
    x_train, y_train = transform_tabular(dataname="nursery", dataset = client_data)
    #print("Dimensiones de x", x_train.shape,"Dimensiones de y", y_train.shape)
    columns = x_train[0].shape
    min_value_for_attr=np.min(x_train, axis = 0)
    max_value_for_attr=np.max(x_train, axis = 0)
    train_dataset = transform_numpy_to_tensor_dataset(x_train, y_train)
    client_dataloader = DataLoader(train_dataset, batch_size = 64)

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
    #a partir de aquí empieza la transformación a la herramienta
    optimizer = client_flex_model["optimizer_func"]
    criterion = client_flex_model["criterion"]
 
    #Esto es más para el ataque, pero bueno aquí puedo entrenar y después tomar el modelo, probarlo
    classifier = PyTorchClassifier(
                model=model,
                clip_values=(min_value_for_attr, max_value_for_attr),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optimizer,
                input_shape=columns,
                nb_classes=4,
            )
    #print(x_train.dtype)
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=50)
    client_flex_model["model"] = classifier.model


    #optimizer.param_groups[0]['params']= model.parameters() Esto hay que ver si trabaja con los parametros que agrega o si no
    #criterion = client_flex_model["criterion"]

    

    #net_config.trainNetwork(local_epochs = 1, criterion = criterion, optimizer = optimizer,
    #                        momentum = 0.9, lr = 0.005, trainloader = client_dataloader, testloader= None, 
    #                        model=model)


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

    x_train, y_train = transform_tabular(dataname="nursery", dataset = test_data)
    test_dataset = transform_numpy_to_tensor_dataset(x_train, y_train)
    #client_dataloader = DataLoader(train_dataset, batch_size = 64)
    #test_dataset = test_data.to_torchvision_dataset(transform = mnist_transform())
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=True, pin_memory=False
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
    print(i)


    state_dict_save = deepcopy(server_flex_model["model"].state_dict())
    grad_save = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in server_flex_model["model"].named_parameters()}
    new_model = deepcopy(server_flex_model)
    new_model["model"].load_state_dict(state_dict_save)
    for name, param in new_model["model"].named_parameters():
        if name in grad_save:
            param.grad = grad_save[name]
    
    
    return new_model

"""
@model_inference
def inferencer(model: FlexModel, data: Dataset):

    x_train, y_train = transform_tabular(dataname="nursery", dataset = data)
    min_value_for_attr=np.min(x_train, axis = 0)
    max_value_for_attr=np.max(x_train, axis = 0)
    columns = x_train[0].shape
    classes = len(set(y_train))

    optimizer = model["optimizer_func"]
    criterion = model["criterion"]

    classifier = PyTorchClassifier(
                model= model["model"],
                clip_values=(min_value_for_attr, max_value_for_attr),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optimizer,
                input_shape=columns,
                nb_classes = classes,
            )

    inf_att_tab.inf_atribute_attck(x_train, y_train, classifier)
    return model

@model_inference
def inferencer(model: FlexModel, data: Dataset):

    x_train, y_train = transform_tabular(dataname="nursery", dataset = data)
    min_value_for_attr=np.min(x_train, axis = 0)
    max_value_for_attr=np.max(x_train, axis = 0)
    columns = x_train[0].shape
    classes = len(set(y_train))

    optimizer = model["optimizer_func"]
    criterion = model["criterion"]

    classifier = PyTorchClassifier(
                model= model["model"],
                clip_values=(min_value_for_attr, max_value_for_attr),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optimizer,
                input_shape=columns,
                nb_classes = classes,
            )

    mem_inf.memberships_inference_atck(x_train, y_train, classifier)
    return model
"""

@model_inference
def inferencer(model: FlexModel, data: Dataset):

    x_train, y_train = transform_tabular(dataname="nursery", dataset = data)
    min_value_for_attr=np.min(x_train, axis = 0)
    max_value_for_attr=np.max(x_train, axis = 0)
    columns = x_train[0].shape
    classes = len(set(y_train))

    optimizer = model["optimizer_func"]
    criterion = model["criterion"]

    classifier = PyTorchClassifier(
                model= model["model"],
                clip_values=(min_value_for_attr, max_value_for_attr),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optimizer,
                input_shape=columns,
                nb_classes = classes,
            )

    mem_inf.inf_atribute_attck_based_on_membership(x_train, y_train, classifier)
    return model

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
        pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_diff_weights_pt, pool.servers)
        # Optional: evaluate the server model
        #pool.servers.map(model_extractor)
        metrics = pool.servers.map(evaluate_global_model)
        pool.servers.map(inferencer)
        #metricsB = pool.servers.map(evaluate_global_model_bad_data)
        # Optional: clean-up unused memory
        selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
        #lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        #print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(1, clients_per_round = 1)