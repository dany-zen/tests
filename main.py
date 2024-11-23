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

from my_poison_attacks import backdoorss, clean_label
from my_models_attacks.inversefed_for_gradient_attacks import *
from my_models_attacks import inversed_gradient


#import my_poison_attacks as pa

""""""""""
#Ataque clean label

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=5)#Carga de datos sin transform

#ataque
source_label1 = 2
source_label2 = 1
epsilon = 0.1
x_clients = 2
x_instances = 0.1 #Valor porcentual

c_l =  clean_label.clean_label_decorator_class(x_clients, x_instances, flex_data, label_one = source_label1, label_two = source_label2)

@c_l.decorator_cl
def clean_label_func(y_data = None, x_for_poison = None, label_one = None, label_two = None, width = None, height = None):# Quitar el none que solo es para probar
    all_img = clean_label.one_shot_kill(y_data, x_for_poison, thresholdp = 3.5, diffp = 100, maxTriesForOptimizing = 2, target_label_one = label_one, target_label_two = label_two, 
                         MaxIter = 10, coeffSimInp = 0.2, saveInterim = False, objThreshold = 2.9)
    #print(img[0])
    return all_img

clean_label_func()#Ejecucion del ataque clean label

flex_data = c_l.flex_data

#Fin del ataque



"""""""""





device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=5)#Carga de datos sin transform

#ataque
source_label = 2
#epsilon = 0.1
#x_clients = 2
#x_instances = 0.1 #Valor porcentual

#backdoor =  backdoorss.backdoor_decorator_class(x_clients, x_instances, flex_data, source_label)

#@backdoor.decorator_back
#def backdoor_func(img = None, label = None,source_label = None, width = None, height = None):
#    img, label = backdoorss.add_modification_sniper_backdoor(img, label, source_label, width, height)
#    #print(img[0])
#    return img, label

#backdoor_func()#Ejecucion del ataque backdoor

#flex_data = backdoor.flex_data

#Fin del ataque










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

    #gradients = []
    #i=0
    #for param in model.parameters():
    #    if param.grad is None:
    #        i+=1
    #print(i)


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
    gradients = []
    i=0
    model2 = client_flex_model["model"]
    state = model2.state_dict()
    #for param in model2.parameters():
    #    if param.grad is not None:
    #       print(param.grad.max())
    #print(i)

    return client_flex_model

def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset): #falta poner esto
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
        new_img, new_label = backdoorss.add_modification_sniper_backdoor(x[cant], y[cant], source_label, w, h)
        bad_imgs.append(new_img)
        bad_labels.append(new_label)

        cant+=1

    print(len(bad_imgs))
    new_imgs = LazyIndexable(bad_imgs, length=len(bad_imgs))
    new_labels = LazyIndexable(bad_labels, length=len(bad_labels))

    new_bad_data = Dataset(X_data = new_imgs, y_data =new_labels)
    return new_bad_data
    #flex_data[server_id] = new_bad_data

#bad_test = modificate_server_data()
#Para el test con los datos que están en el servidor

def evaluate_global_model_bad_data(server_flex_model: FlexModel, test_data: Dataset):#falta poner esto
    #test_data = bad_test
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


#Ataque de inferencia
@model_inference
def inferencer(client_model: FlexModel, client_data: Dataset):#Ver lo de la perdida que está dando Nan, puede ser que estoy solo viendo pocas iteraciones de todo y nada se a optimizado
                                                             #Ver bien lo de la media y el std, ver que se escojan los clientes para infestar y que sea a partir de un número de iteraciones

    for param in client_model["model"].parameters():
        if param.grad is None:
            print("No")
    train_dataset = client_data.to_torchvision_dataset(transform = mnist_transform())
    mean, std = inversed_gradient.get_meanstd(train_dataset)
    mean = torch.as_tensor(mean,)[:, None, None]
    std = torch.as_tensor(std,)[:, None, None]
    labels = [torch.as_tensor((label,)) for label in np.array(train_dataset.data.y_data)]
    num_images = len(labels)
    labels = torch.cat(labels)
    labels=labels.long()
    #print(mean.shape)#Covertir a tensor y aplicar el metodo cat
    #Ver lo de pasar de tensor a valor con el std y la media
    
    output, stats = inversed_gradient.reconstruction_gradient_attack(client_model, train_dataset[0][0].shape, mean, std, 
                                                                     num_images = num_images, labels=labels)
    return client_model


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

        #Aplica el ataque
        if i == 0:
            n_client_for_attack = 1
            client_for_attack = selected_clients.select(n_client_for_attack)
            client_for_attack.map(inferencer)#En este caso los mismos clientes selccionados son los qu estpy sacando las cosas, debería definir un número para cada uno

        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(collect_client_diff_weights_pt, selected_clients)
        pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_aggregated_diff_weights_pt, pool.servers)
        # Optional: evaluate the server model
        metrics = pool.servers.map(evaluate_global_model)
#        metricsB = pool.servers.map(evaluate_global_model_bad_data)
        # Optional: clean-up unused memory
        #selected_clients.map(clean_up_models)
        loss, acc = metrics[0]
#        lossb, accb = metricsB [0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
#        print(f"Server: Test accB: {accb:.4f}, test lossB: {lossb:.4f}")

train_n_rounds(5, clients_per_round=2)






"""""""""

#Ataque clean label

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

flex_data, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=5)#Carga de datos sin transform

#ataque
source_label1 = 2
source_label2 = 1
epsilon = 0.1
x_clients = 2
x_instances = 0.1 #Valor porcentual

c_l =  clean_label.clean_label_decorator_class(x_clients, x_instances, flex_data, label_one = source_label1, label_two = source_label2)

@c_l.decorator_cl
def clean_label_func(y_data = None, x_for_poison = None, label_one = None, label_two = None, width = None, height = None):# Quitar el none que solo es para probar
    all_img = clean_label.one_shot_kill(y_data, x_for_poison, thresholdp = 3.5, diffp = 100, maxTriesForOptimizing = 2, target_label_one = label_one, target_label_two = label_two, 
                         MaxIter = 10, coeffSimInp = 0.2, saveInterim = False, objThreshold = 2.9)
    #print(img[0])
    return all_img

clean_label_func()#Ejecucion del ataque clean label

flex_data = c_l.flex_data

#Fin del ataque

"""""""""
#Todo esto va antes del ejecutar xq son ataques de envenenamiento de datos




#"""""""""

""""""""""

1- tomar la BD
2- Poner la puerta trasera con un limite de instancias a modificar y la clase que modificar, aqui modificar de lo que implementé, solo trabajar con el trigger
3- Modificar el decorador o mejor crear uno nuevo para que tome el id del cliente donde esta el datase para establecer así el límite de instancias a modificar
4- Modificar los valores de los atributos y  después el de la clase si coincide con la etiqueta a modificar y si no se alcanza el limite de variables
5- Del data test, tomar un pedazo y hacer la modificación y dividir el dataset en 2, los modificados para la prueba y los normales

"""""""""