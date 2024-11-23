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
from flex.data import FedDataset
from flex.model.model import FlexModel
from flex.actors.actors import FlexActors, FlexRoleManager

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)
from my_models_attacks import free_riding as fr
from my_models_attacks import param_manipulated_attacks as param_maniplt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

rounds = 600

free_riding_ids = []

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=20)#Carga de datos sin transform

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




@model_poisoner
def fr_attack(client_model: FlexModel = None, round = None):
    rondas = rounds
    prev = deepcopy(client_model["model"])

    fr_model_client = fr.free_riding_atack(type_noise = "disguised", flex_model = client_model,
                                            std_0 = 0, power = 1, decay = 1, f_round = rounds, multiplicator = 1)
    client_model["model"] = fr_model_client
    client_model["previous_model"] = prev

    return client_model

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

def select_criteria(actor_id, actor, list_not_selection = []):
    selection=False
    print("ids",actor_id)
    if actor_id not in list_not_selection:
        print("id",actor_id, "No esta")
        selection = True
    return selection

@param_maniplt.manipulated_weights_attack
def agregator_attack(aggregated_weights_as_list: list):
    f=8#Número de clientes maliciosos
    print("Antes de la modif",torch.sum(aggregated_weights_as_list[0][0]))
    modif_list_n = param_maniplt.adecuate_params(aggregated_weights_as_list)
    #modif_list = modif_list_n
    #modif_list = param_maniplt.scaling_attack_scale(modif_list_n, f, device)
    #modif_list = param_maniplt.trim_attack(modif_list_n, f, device)
    #modif_list = param_maniplt.krum_attack(modif_list_n, f, device)
    #modif_list = param_maniplt.min_max_attack(modif_list_n, f, device)
    modif_list = param_maniplt.min_sum_attack(modif_list_n, f, device)
      

    #i=0
    #for m in modif_list:
    #    print("Modificado",m.shape)
    #   print("Para", i+1, "Cliente")
    #    i+=1
    #i=0

    modif_to_otiginal = param_maniplt.adecuate_params_reverse(modif_list, aggregated_weights_as_list[0])
    print("Después de la modif", torch.sum(modif_to_otiginal[0][0]))
    #for c in modif_to_otiginal:
    #    print("Para cliente:", i+1)
    #    for p in c:
    #        print("Para parámetro:", p.shape)
    #    i+=1
    return modif_to_otiginal


def train_n_rounds(n_rounds, clients_per_round=20):
    pool = FlexPool.client_server_pool(
        fed_dataset = flex_data, server_id = server_id, init_func = build_server_model
    )
    n_free_riding = 1
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")

        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients

        global rounds
        rounds = i
        print(rounds)

        fr_clients_pool = selected_clients.select(n_free_riding)
        selected_fr_clients = fr_clients_pool.clients

        #test_clients_pool = selected_clients.clients.select(select_criteria(actor_id = None, actor = None, list_not_selection = selected_fr_clients.actor_ids))
        #test_clients = test_clients_pool.clients

        test_actors = FlexActors()
        test_models= {}
        test_new_data = FedDataset()

        for actor_ids in selected_clients.actor_ids:
            if actor_ids not in selected_fr_clients:
                test_actors[actor_ids] = selected_clients._actors[actor_ids]
                test_models[actor_ids] = selected_clients._models[actor_ids]
                test_new_data[actor_ids] = selected_clients._data[actor_ids]
        test_clients_pool = FlexPool(flex_actors = test_actors, flex_data = test_new_data, flex_models = test_models)
        test_clients = test_clients_pool.clients


        print(f"Selected initial clients for this round: {len(selected_clients)}")
        print(f"Selected initial clients for this round: {selected_clients.actor_ids}")
        print(f"Selected free rinding clients for this round: {len(selected_fr_clients)}")
        print(f"Selected free rinding clients for this round: {selected_fr_clients.actor_ids}")
        print(f"Selected not free rinding clients for this round: {test_clients.actor_ids}")
        

        # Deploy the server model to the selected clients
        pool.servers.map(deploy_serv, selected_clients)
        # Each selected client trains her model
        test_clients.map(train)
        selected_fr_clients.map(train)
        #selected_fr_clients.map(fr_attack)

        #Juntar los clientes tanto fr como normales
        actors = FlexActors()
        models= {}
        new_data = FedDataset()

        for actor_ids in selected_clients.actor_ids:
            print(actor_ids)
            if actor_ids in selected_fr_clients.actor_ids:
                actors[actor_ids] = selected_fr_clients._actors[actor_ids]
                models[actor_ids] = selected_fr_clients._models[actor_ids]
                new_data[actor_ids] = selected_fr_clients._data[actor_ids]
            elif actor_ids in test_clients.actor_ids:
                actors[actor_ids] = test_clients._actors[actor_ids]
                models[actor_ids] = test_clients._models[actor_ids]
                new_data[actor_ids] = test_clients._data[actor_ids]
        new_pool = FlexPool(flex_actors = actors, flex_data = new_data, flex_models = models)
        clients_to_agregate = new_pool.clients


        # print(f"Selected not free rinding clients for this round: {clients_to_agregate.actor_ids}")


        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_client_diff_weights_pt, clients_to_agregate)

        #pool.aggregators.map(ataque_de_manipulacion_de_parametros) a esto agregar que hay que pasar los parámetros como una lista ([param_modif]) porque sino tengo que estar cambiando la dimensión de procesamiento de los tensores y es mucha picha x ahora
        #al ataque hay que pasarle todos los datos de todos los clientes, es decir el listadito donde se guarda to eso, por tanto hay que convetir el listado entero y solo actualizar los valores de los que se van a modificar
        #hacer deepcopy cuando se pasen y a partir de ahí meterle el cat a todos y después el cat de nuevo para juntar todas las actualizaciones

        pool.aggregators.map(agregator_attack)
        pool.aggregators.map(fed_avg)

        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_diff_weights_pt, pool.servers)
        # Optional: evaluate the server model
        metrics = pool.servers.map(evaluate_global_model)
#        metricsB = pool.servers.map(evaluate_global_model_bad_data)
         #Optional: clean-up unused memory
        selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
#        lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
#        print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(8, clients_per_round = 10)
