from ANNarchy import *
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import random as rd
import time

LIF = Neuron(  #I = 75
    parameters = """
    tau = 50.0 : population
    I = 0.0
    tau_I = 10.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + (I-65) : init=0
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v >= -40.0",
    reset = "v = -65"
)

IZHIKEVICH = Neuron(  #I = 20
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
        du/dt = a*(b*v - u) : init=-14.0
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)

def bootstrap_data(data_x, data_y,n_bootstrap, n_samples):
    # Lista para guardar los conjuntos de datos generados
    bootstrap_datasets = []
    # Generar los conjuntos de datos usando bootstrapping
    for i in range(n_bootstrap):
        X_bootstrap, y_bootstrap = resample(data_x, data_y, replace=True, n_samples=n_samples)
        bootstrap_datasets.append((X_bootstrap, y_bootstrap))
    return bootstrap_datasets
# DATOS
# Función para cargar los datos desde un archivo
def load_from_file(file_name):
    data = np.load(file_name, allow_pickle=True).item()
    return data["data_X"], data["data_y"]

def _load_eval_config():
    params = {}
    try:
        with open('config/neuron.cfg') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    params[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return params

_EVAL_PARAMS = _load_eval_config()
_dataset  = _EVAL_PARAMS.get('dataset', 'bc')
_encoder  = _EVAL_PARAMS.get('encoder', 'single_spike')
_enc_tag  = 'ss' if _encoder == 'single_spike' else 't'
_train_file = f"data/{_dataset}_data_{_enc_tag}/train_{_enc_tag}.npy"
data_X, data_y = load_from_file(_train_file)
largo = len(data_y)
subsets = bootstrap_data(data_X, data_y, 50, largo)

# DECODING
# Decodificación por votación
def vote_decoding(spikes, n_input, n_output):
    #rm  = spikes[0][0]
    #for key in spikes:
    #    spikes[key] = [x for x in spikes[key] if x != rm]
    votes = []
    total = n_input + n_output
    for i in range(n_input,total):
        votes.append(len(spikes[i]))
    max_spikes = max(votes)
    indices_maximos = [i for i, x in enumerate(votes) if x == max_spikes]
    
    if len(indices_maximos) == 1:
        index = indices_maximos[0]
    else:
        index = rd.choice(indices_maximos)

    return index
# Decodificación por primera espiga
def first_spike_decoding(spikes, n_input, n_output):
    #rm  = spikes[0][0]
    #for key in spikes:
    #    spikes[key] = [x for x in spikes[key] if x != rm]
    menor = float('inf')
    index = -1
    flag = False
    flag2 = False
    for i in range(n_input, n_input + n_output):
        #print("i: ", i)
        #print("spikes[i]: ", spikes[i])
        if len(spikes[i]) > 0:
            #print("len(spikes[i]): ", len(spikes[i]))
            flag = True
            if i == 0:
                menor  = spikes[i][0]
                index = 0
                #print("menor: ", menor, " index: ", index)
            else:
                if spikes[i][0] < menor:
                    flag2 = False
                    menor  = spikes[i][0]
                    index = i - n_input
                    #print("menor: ", menor, " index: ", index)
                elif spikes[i][0] == menor:
                    index = rd.choice([index, i-n_input])
                    flag2 = True
                    #print("rd - menor: ", menor, " index: ", index)
    #print("index: ", index)
    #if flag2: print("rd")
    if flag: return index
    else: return rd.randint(0, n_output-1)

# SIMULATION
# Simulación con espíga única
def simulate_single_spike(sample, n_input, neurons_per_input, pop, time=10.0, flag = False):
    simulate(1.0)
    if flag: print("sample: ",sample)
    # Para cada muestra
    if neurons_per_input != 1:
        n = neurons_per_input
        for k in range(n_input):
            if sample[k//n][k%n] == 1:
                pop[k].v = 30.0
            if flag: print("pop[k].v: ",pop[k].v)
    else:
        for k in range(n_input):
            pop[k].v = 30.0 if sample[k] == 1 else 0.0  # 15.0 es un ejemplo de intensidad de corriente
    simulate(time)

# Simulación con temporal encoding
def simulate_temporal(sample, n_input, pop, neurons_per_input=4, time=10):
    simulate(1.0)
    for t in range(int(time)):
        for k in range(n_input):
            if sample[k][t] == 1:
                pop[k].v = 30.0
        simulate(1.0)  # Simular cada paso temporal  
    simulate(10.0)


def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial):
    try:
        clear()
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        pop.a = float(_EVAL_PARAMS.get('a', 0.02))
        pop.b = float(_EVAL_PARAMS.get('b', 0.2))
        pop.c = float(_EVAL_PARAMS.get('c', -65.0))
        pop.d = float(_EVAL_PARAMS.get('d', 8.0))
        proj = Projection(pre=pop, post=pop, target='exc')

        if matrix.size == 0:
            raise ValueError("matrix is empty")

        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        nombre = 'annarchy/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))

        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])

        fit = fitness(pop,M,int(n_entrada),int(n_salida))
        #guardar_red(pop,proj,nombre+'/red')

        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)
        return -1

def fitness(pop, M, n_input, n_output, flag=False):
    encoder = _EVAL_PARAMS.get('encoder', 'single_spike')
    decoder = _EVAL_PARAMS.get('decoder', 'first_spike')
    neuronas_por_input = 4
    total = 0
    n = len(subsets)
    for i in range(n):
        x = subsets[i][0]
        y = subsets[i][1]
        sum = 0
        samples = len(x)
        for j in range(samples):
            target = y[j]
            if encoder == 'single_spike':
                simulate_single_spike(x[j], n_input, neuronas_por_input, pop, 10.0, flag)
            else:
                simulate_temporal(x[j], n_input, pop, neuronas_por_input, 10)
            spikes = M.get('spike')
            if decoder == 'first_spike':
                index = first_spike_decoding(spikes, n_input, n_output)
            else:
                index = vote_decoding(spikes, n_input, n_output)
            if index == target:
                sum += 1.0
            pop.reset()
            M.reset()
        total += (sum / samples)
    fitness = round(total / n, 2)
    return fitness
