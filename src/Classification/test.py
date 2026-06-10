from ANNarchy import *
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import random as rd

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


# DECODING
# Decodificación por votación
def vote_decoding(spikes, n_input, n_output):
    rm  = spikes[0][0]
    for key in spikes:
        spikes[key] = [x for x in spikes[key] if x != rm]
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
    rm  = spikes[0][0]
    for key in spikes:
        spikes[key] = [x for x in spikes[key] if x != rm]
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

def snn(n_entrada, n_salida, n, i, matrix, trial, x, y):
    try:
        clear()
        pop = Population(geometry=n, neuron=LIF) #MODIFICAR según neurona
        proj = Projection(pre=pop, post=pop, target='exc')

        if matrix.size == 0:
            raise ValueError("matrix is empty")

        """lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)"""

        proj.connect_from_sparse(matrix)
        nombre = 'annarchy-test/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))

        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])

        fit = fitness(pop,M,int(n_entrada),int(n_salida),x,y)
        #guardar_red(pop,proj,nombre+'/red')

        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)
        return -1

def fitness(pop, M, n_input, n_output, x, y, flag=False):

    sum = 0
    samples = len(x)
    for j in range(samples):
        neuronas_por_input = 4
        target = y[j]
        
        simulate_single_spike(x[j], n_input, neuronas_por_input, pop, 10.0, flag)
        spikes = M.get('spike')
        index = first_spike_decoding(spikes, n_input, n_output)
        if index == target:
                sum += 1.0

        pop.reset()
        M.reset()

    fitness = round((sum/samples),2)
    return fitness

def read_matrix(file_name,n):
    matrix = scipy.sparse.lil_matrix((int(n), int(n)))
    with open(file_name, 'r') as file:
        flag = False
        for line in file:
            line = line.strip().split(';')
            row = int(line[0])-1
            col = int(line[1])-1
            weight = float(line[2])
            matrix[row, col] = weight

    return matrix, 0, 0


if __name__ == "__main__":

    argv = sys.argv
    trial  = argv[1]

    n_input = 120 #MODIFICAR para base de datos (cantidad de características X neuronas por caracteristica)
    n_output = 2 #MODIFICAR para base de datos (numero de salidas)
    
    matrix, id, idAnnarchy = read_matrix("results/trial-"+str(trial)+"/best0.txt",300)

    train_file = "data/bc_data_ss/test_ss.npy" #MODIFICAR para la base de datos y codificación a utilizar
    # Leer los datos desde los archivos para verificar
    data_X, data_y = load_from_file(train_file)

    largo = len(data_y)
    subsets = bootstrap_data(data_X, data_y,50,largo)
    test_fitness = 0
    for i in range(len(subsets)):
        data_X = subsets[i][0]
        data_y = subsets[i][1]
        fit = snn(n_input, n_output, 300, idAnnarchy, matrix, trial, data_X, data_y)
        test_fitness += fit
    fitness = test_fitness/len(subsets)
    print("test_fitness: ",round(fitness,2))
