import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import os

# Función para cargar el conjunto de datos
def load_dataset(dataset_loader):
    data = dataset_loader()
    return np.array(data.data), np.array(data.target)

# Función para normalizar los datos entre [0, 1]
def normalize_data(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Función para dividir los datos en entrenamiento y prueba
def split_data(data_X, data_y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(data_X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(data_X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return (data_X[train_indices], data_y[train_indices],
            data_X[test_indices], data_y[test_indices])

# Codificación simple con picos
def single_spike_encoding(data, num_neurons=4):
    
    encoded_data = []
    step = 1 / num_neurons
    for features in data:
        spike_train = []
        for value in features:
            neuron_idx = min(int(value / step), num_neurons - 1)
            spike = [0] * num_neurons
            spike[neuron_idx] = 1
            spike_train.append(spike)  # Cada característica es un arreglo separado
        encoded_data.append(spike_train)
    return np.array(encoded_data, dtype=object)  # Arreglo de arreglos

# Codificación temporal
def temporal_encoding(general_data, num_neurons=4, max_time=10):
    general_encoded_data = []
    for data in general_data:
        encoded_data = []
        step = 1 / num_neurons
        neurona = 0
        for value in data:
            for g in range(num_neurons):
                encoded_data.append([0]*max_time)
                print(g)
                
            neuron_idx = min(int(value / step), num_neurons - 1)

            norm_val = (value - (neuron_idx * step) )/ step

            spike_time = int(norm_val * max_time)
            spike_train = [0] * max_time
            if spike_time < max_time:
                spike_train[spike_time] = 1

            encoded_data[neurona + neuron_idx] = spike_train
            neurona += num_neurons
        general_encoded_data.append(encoded_data)
    return np.array(general_encoded_data, dtype=object)  # Convertir a arreglo de arreglos

# Función para guardar los datos en formato .npy
def save_to_npy(file_name, data_X, data_y):
    np.save(file_name, {"data_X": data_X, "data_y": data_y})

# Función para cargar los datos desde un archivo .npy
def load_from_npy(file_name):
    data = np.load(file_name, allow_pickle=True).item()
    return data["data_X"], data["data_y"]

# Directorios para guardar los datos
['iris', 'wine', 'bc']
for dataset in ['iris', 'wine', 'bc']:
    output_dir_ss = f'{dataset}_data_ss'
    output_dir_t = f'{dataset}_data_t'
    os.makedirs(output_dir_ss, exist_ok=True)
    os.makedirs(output_dir_t, exist_ok=True)

    # Cargar y normalizar el conjunto de datos
    if dataset == 'iris':
        data_X, data_y = load_dataset(load_iris)
    elif dataset == 'bc':
        data_X, data_y = load_dataset(load_breast_cancer)
    elif dataset == 'wine':
        data_X, data_y = load_dataset(load_wine)
    
    # Normalizar el conjunto de datos
    data_X_normalized = normalize_data(data_X)

    # Dividir en entrenamiento y prueba
    train_X, train_y, test_X, test_y = split_data(data_X_normalized, data_y)

    # Codificar los datos con Single Spike Encoding
    train_X_single_spike = single_spike_encoding(train_X)
    test_X_single_spike = single_spike_encoding(test_X)

    # Codificar los datos con Temporal Encoding
    train_X_temporal = temporal_encoding(train_X)
    test_X_temporal = temporal_encoding(test_X)

    # Guardar los conjuntos en formato .npy
    save_to_npy(os.path.join(output_dir_ss, "train_ss.npy"), train_X_single_spike, train_y)
    save_to_npy(os.path.join(output_dir_ss, "test_ss.npy"), test_X_single_spike, test_y)

    save_to_npy(os.path.join(output_dir_t, "train_t.npy"), train_X_temporal, train_y)
    save_to_npy(os.path.join(output_dir_t, "test_t.npy"), test_X_temporal, test_y)

    # Verificar los datos guardados
    print(f'Datos guardados en: {output_dir_ss} y {output_dir_t}')

# Ejemplo de carga de datos
#loaded_X, loaded_y = load_from_npy(os.path.join(output_dir, "train_t.npy"))
#print("Primer dato cargado:", loaded_X[0], "Etiqueta:", loaded_y[0])
