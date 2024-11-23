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

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)

from my_poison_attacks import backdoorss, clean_label, dirty_label
from my_models_attacks.inversefed_for_gradient_attacks import *
from my_models_attacks import inversed_gradient

x_clients = 2
x_instances = 0.9
source_label = 2

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes = 2)#Carga de datos sin transform

num_labels_datas=10


d_l = dirty_label.dirty_label_decorator_class(x_clients, x_instances, flex_data, num_labels_datas, None)

@d_l.decorator_dirty
def dirty_labels(img = None, label = None, label_one = None, label_two = None, width = None, height = None):
    #print(label) 
    label_new = dirty_label.label_flipping_attack(num_labels = label_one, worker_label = label)
    #print(label_new)
    return img, label_new

dirty_labels()

flex_data = d_l.flex_data

#backdoor =  backdoorss.backdoor_decorator_class(x_clients, x_instances, flex_data, source_label)

#@backdoor.decorator_back
#def backdoor_func(img = None, label = None,source_label = None, width = None, height = None):
#    img, label = backdoorss.add_backdoor(img, label, source_label, width, height)
#    #print(img[0])
#    return img, label

#backdoor_func()#Ejecucion del ataque backdoor

#flex_data = backdoor.flex_data

#@backdoor.decorator_scalin_back
#def backdoor_func(imgs = None, labels = None,source_label = None, device = device):
#    
#    new_datas, new_labels = backdoorss.scaling_attack_insert_backdoor(imgs, labels, obj_label = source_label,  device = device)
#    
#    return new_datas, new_labels

net_config = ExecutionNetwork()

#backdoor_func()

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


def modificate_server_data():
    server_data = flex_data[server_id]
    x = np.array(server_data.X_data)
    y = np.array(server_data.y_data)

    cant_modif= 1
    cant_bad, w, h = backdoorss.util_for_backdoor_sniper(x, epsilon = cant_modif)
    cant = 0

    bad_imgs = []
    bad_labels = []
    while(cant_bad > cant):
        new_img, new_label = backdoorss.add_backdoor(x[cant], y[cant], source_label, w, h)
        bad_imgs.append(new_img)
        bad_labels.append(new_label)

        cant+=1

    print(len(bad_imgs))
    new_imgs = LazyIndexable(bad_imgs, length=len(bad_imgs))
    new_labels = LazyIndexable(bad_labels, length=len(bad_labels))

    new_bad_data = Dataset(X_data = new_imgs, y_data =new_labels)
    return new_bad_data

bad_test = modificate_server_data()

def evaluate_global_model_bad_data(server_flex_model: FlexModel, test_data: Dataset):#falta poner esto
    test_data = bad_test
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
    print(i)


    state_dict_save = deepcopy(server_flex_model["model"].state_dict())
    grad_save = {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param) for name, param in server_flex_model["model"].named_parameters()}
    new_model = deepcopy(server_flex_model)
    new_model["model"].load_state_dict(state_dict_save)
    for name, param in new_model["model"].named_parameters():
        if name in grad_save:
            param.grad = grad_save[name]
    
    
    return new_model


def train_n_rounds(n_rounds, clients_per_round=20):
    pool = FlexPool.client_server_pool(
        fed_dataset= flex_data, server_id=server_id, init_func=build_server_model
    )
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients
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
        metrics = pool.servers.map(evaluate_global_model)
        #metricsB = pool.servers.map(evaluate_global_model_bad_data)
        # Optional: clean-up unused memory
        selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
        #lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        #print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(1, clients_per_round=2)