import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, vgg16, vgg19

from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import FeatureCollisionAttack


#Esto va siendo el prototipo para el decorador principal de los ataques de envenenamiento


def define_model(data_name, define_pretrain):
    model = None
    criterion = None
    optimizer = None
    if data_name == "cifar":
        if define_pretrain == "resnet-34":
            model = resnet34(pretrained=True)
        elif define_pretrain == "resnet-50":
            model = resnet50(pretrained=True)
        elif define_pretrain == "vgg-16":
            model = vgg16(pretrained=True)
        elif define_pretrain == "vgg-19":
            model = vgg19(pretrained=True)
        else:
            model = resnet18(pretrained=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters, lr = 0.001)

    elif data_name == "mnist":
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer, criterion

class SimpleCNN(nn.Module): #Para datasets de mnist o derivados
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Entrada 28x28, salida 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Entrada 14x14, salida 14x14
        self.fc1 = nn.Linear(64*7*7, 128)                        # Aplanado, 7x7 después de max pooling
        self.fc2 = nn.Linear(128, 10)                            # 10 clases de salida
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)               # Reduce las dimensiones espaciales a la mitad

    def forward(self, x):
        #print(x.ndimension())
        if x.ndimension() != 4:
            x = x.unsqueeze(1)
        #print(x.dtype)
        x = self.conv1(x)        # Conv1, salida 28x28x32
        x = self.relu(x)         
        x = self.maxpool(x)      # MaxPool, salida 14x14x32
        x = self.conv2(x)        # Conv2, salida 14x14x64
        x = self.relu(x)
        x = self.maxpool(x)      # MaxPool, salida 7x7x64
        x = x.view(x.size(0), -1)  # Aplanado, salida 64*7*7
        x = self.fc1(x)          # FC1
        x = self.relu(x)
        x = self.fc2(x)          # FC2, salida 10 clases
        return x

def feat_collision_attack(x_data, y_data, model, number_to_poison, target_class, base_class, # La definidicón del modelo la debo modificar, acá debo definir el modelo de pytorch pero utilizando lo que da la herramienta
                          iter = 10, sim_coeff = 256, wmark = 0.3, lr = 1):

    #Choose a target image
    #x = x_data[y_data == target_class]
    #print(x.shape)
    def inspect(to_target, class_objt, limit = 1):
        target_instance = None
        count = 0
        arr = []
        for x in to_target:
            if len(x.shape)==2:
                x = np.expand_dims(x, axis = 0).astype(np.float32)
            #print( x_expand.dtype )
            pred = model.predict(x)
            if np.argmax(pred) == class_objt:
                target_instance = x
                arr.append(target_instance)
                count+=1
                if count == limit:
                    break
        return target_instance, np.array(arr)
    
    to_target = x_data[y_data == target_class]
    #Funcion de arriba
    target_instance, _ = inspect(to_target, target_class)
    #Funcion de arriba
    target_instance = np.expand_dims(target_instance, axis = 0)
    feat_layer = model.layer_names[-2]

    #Choose an images for posion
    bases = x_data[y_data == base_class]
    _ , base_instances = inspect(bases, base_class, number_to_poison)
    base_instances = np.copy(base_instances)

    attack = FeatureCollisionAttack(model, target_instance, feat_layer, max_iter = iter,
                                     similarity_coeff = sim_coeff, watermark = wmark,
                                     learning_rate = lr)
    
    poison, poison_labels = attack.poison(base_instances.astype(np.float32))

    return poison, poison_labels

def run_attack_caract_colision(x_data, y_data, dataset, pret, number_to_poison, 
                               target_class, base_class, n_class,
                               iter, sim_coeff, wmark, lr): # acá x_data y y_data debe venir ya preprocesado, así que debo meterle un dataloader
    
    model, optimizer, criterion = define_model(data_name = dataset, define_pretrain = pret)
    porcent_to_train = 0.5
    min = np.min(x_data)
    max = np.max(x_data)
    sizes = x_data[0].shape
    classifier = PyTorchClassifier(
                model=model,
                clip_values=(min, max),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optimizer,
                input_shape = sizes,
                nb_classes = n_class,
            )
    classifier.fit(x_data, y_data, batch_size=64, nb_epochs = 2)

    poison_data, poison_label = feat_collision_attack(x_data.astype(np.float32), y_data, classifier, number_to_poison, target_class, base_class,
                          iter, sim_coeff, wmark, lr)
    
    print("Cantidad de poisons", len(poison_data), np.squeeze(poison_data, axis = 1).shape)
    labels_p=[]
    for label in poison_label:
        labels_p.append(np.argmax(label))
    print("Data envenenada:", poison_data.shape)

    print("Label envenenada:", poison_label.shape)
    
    return poison_data, poison_label