import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree

def inf_atribute_attck(x_data, y_data, model):#Model debe tener la estructura quelllos definen para los modelos
    attack_train_ratio = 0.9
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    attack_feature = 1
    attack_x_test_predictions = np.array([np.argmax(arr) for arr in model.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen
    print("Dim del test attck:", attack_x_test.shape)
    print("Dim del prediction attck:", attack_x_test_predictions.shape)



    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)
    bb_attack = AttributeInferenceBlackBox(model, attack_feature=attack_feature)
    attack_x_test = np.delete(attack_x_test, attack_feature, 1)
    print("Tamaño de las x de ataque",len(attack_x_test))
    print("Tamaño de las x de ataque",len(attack_x_test_predictions))

    bb_attack.fit(attack_x_train)
    #values = [-0.70718864, 1.41404987]
    values= None
    #values = [-1.424395723148083, -0.7130212117030289, -0.0016467002579747094, 0.7097278111870795, 1.4211023226321338]
    inferred_train_bb = bb_attack.infer(attack_x_test, pred = attack_x_test_predictions, values = values)
    print(np.max(inferred_train_bb), np.max(attack_x_test_feature))

    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train_bb, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_bb)
    print(train_acc)
    return inferred_train_bb

def inf_atribute_attck_wb_one(x_data, y_data, model):#El segundo de white box de inferir atributos en estos no se entrena un modelo, se utiliza la información del estimador, osea conocida la estructura del modelo del cliente + los parámetros que se tienene
    attack_train_ratio = 0.9
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    attack_feature = 1
    attack_x_test_predictions = np.array([np.argmax(arr) for arr in model.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen
    print("Dim del test attck:", attack_x_test.shape)
    print("Dim del prediction attck:", attack_x_test_predictions.shape)

    priors = [3465 / 5183, 1718 / 5183]

    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(model, attack_feature=attack_feature)

    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    print("Tamaño de las x de ataque",len(attack_x_test))
    print("Tamaño de las x de ataque",len(attack_x_test_predictions))



    #Aqui define el entrenamiento
    #values = [-0.70718864, 1.41404987]
    values= None
    #values = [-1.424395723148083, -0.7130212117030289, -0.0016467002579747094, 0.7097278111870795, 1.4211023226321338]
    inferred_train_wb1 = wb_attack.infer(attack_x_test, attack_x_test_predictions, values=values, priors=priors)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train_wb1, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_wb1)
    print(train_acc)
    return inferred_train_wb1

def inf_atribute_attck_wb_two(x_data, y_data, model):#El segundo de white box de inferir atributos
    attack_train_ratio = 0.9
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    attack_feature = 1
    attack_x_test_predictions = np.array([np.argmax(arr) for arr in model.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen
    print("Dim del test attck:", attack_x_test.shape)
    print("Dim del prediction attck:", attack_x_test_predictions.shape)

    priors = [3465 / 5183, 1718 / 5183]

    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    wb_attack = AttributeInferenceWhiteBoxDecisionTree(model, attack_feature=attack_feature)

    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    print("Tamaño de las x de ataque",len(attack_x_test))
    print("Tamaño de las x de ataque",len(attack_x_test_predictions))



    #Aqui define el entrenamiento
    #values = [-0.70718864, 1.41404987]
    values= None
    #values = [-1.424395723148083, -0.7130212117030289, -0.0016467002579747094, 0.7097278111870795, 1.4211023226321338]
    inferred_train_wb = wb_attack.infer(attack_x_test, attack_x_test_predictions, values = values, priors = priors)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train_wb, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_wb)
    print(train_acc)
    return inferred_train_wb