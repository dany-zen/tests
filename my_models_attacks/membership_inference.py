import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceMembership

#Esto usa el ataque de inferencia de mimebros para predecir la pertenencia de los valores de un atributo, a partir de los datos
def inf_atribute_attck_based_on_membership(x_data, y_data, model):#Model debe tener la estructura quelllos definen para los modelos



    attack_train_ratio = 0.75
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    factor_train_membership = 0.5
    limit_train = int(len(attack_x_test) * factor_train_membership)
    #limit_test = int(len(attack_x_test) * factor_train_membership)

    only_X_test_to_train = attack_x_test[:limit_train]
    only_y_test_to_train = attack_y_test[:limit_train]

    only_X_test_to_test = attack_x_test[limit_train:]
    only_y_test_to_test = attack_y_test[limit_train:]


    attack_feature = 1
    #atrb_all_values = np.delete(attack_x_train, attack_feature, 1)
    atrb_all_values = x_data [:, attack_feature]
    values = util_count_unique_atr_value(atrb_all_values)



    attack_x_test_feature = only_X_test_to_test[:, attack_feature].copy().reshape(-1, 1)
    mem_attack = MembershipInferenceBlackBox(model)
    attack_x_test = np.delete(only_X_test_to_test, attack_feature, 1)


    mem_attack.fit(attack_x_train, attack_y_train, only_X_test_to_train, only_y_test_to_train)#Probar pasando todas las de pruebas y solo limitar las de inferencia
    attack = AttributeInferenceMembership(model, mem_attack, attack_feature=attack_feature)


    #values= None

    inferred_train = attack.infer(attack_x_test, only_y_test_to_test, values=values)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train)
    print(train_acc)
    return train_acc


def memberships_inference_atck(x_data, y_data, model):# Lo que se debería pasar aquí son un conjunto de datos para determinar si pertenecen o no al cliente, a partir de los parámetros, la arquitectura del modelo y un conjunto de datos que asume el atacante que analiza el cliente o el modelo en si
    
    attack_train_ratio = 0.5
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    factor_train_membership = 0.7
    #limit_test = int(len(attack_x_test) * factor_train_membership)

#Para los miembros que pertenezcan
    limit_train = int(len(attack_x_train) * factor_train_membership)
    X_train_to_members = attack_x_train[:limit_train]
    y_train_to_members = attack_y_train[:limit_train]

    X_test_to_member = attack_x_train[limit_train:]
    y_test_to_member = attack_y_train[limit_train:]

#Para los miembros que no pertenezcan
    limit_train = int(len(attack_x_test) * factor_train_membership)
    X_train_to_non_members = attack_x_test[:limit_train]
    y_train_to_non_members = attack_y_test[:limit_train]

    X_test_to_non_member = attack_x_test[limit_train:]
    y_test_to_non_member = attack_y_test[limit_train:]

    mem_attack = MembershipInferenceBlackBox(model)

    mem_attack.fit(X_train_to_members, y_train_to_members, X_train_to_non_members, y_train_to_non_members)

    inferred_train_bb = mem_attack.infer(X_test_to_member, y_test_to_member)
    inferred_test_bb = mem_attack.infer(X_test_to_non_member, y_test_to_non_member)

    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy {test_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")

    return inferred_train_bb


def util_count_unique_atr_value(x):
    values = []

    values = np.unique(x).tolist()
    print(values)

    return values