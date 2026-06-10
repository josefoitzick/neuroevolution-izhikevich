from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd

import threading

# Declarar un mutex global para sincronizar el acceso a compile()
compile_mutex = threading.Lock()

LIF = Neuron(
    parameters = """
    tau = 30.0 : population
    I = 15.0
    tau_I = 3.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + I : init=13.5
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v >= 15.0",
    reset = "v = 13.5",
    refractory = 3.0
)

def snn(n_entrada, n_salida, n, i, matrix): 
    clear()
    pop = Population(geometry=n, neuron=LIF)
    proj = Projection(pre=pop, post=pop, target='exc')
    #print(matrix,"\n")
    #Matrix to numpy array
    matrix = np.array(matrix)

    #Reemplazar los valores 0 por None
    matrix[matrix == 0] = None

    proj.connect_from_matrix(matrix)
    #print('nombre')
    nombre = 'annarchy/annarchy-'+str(int(i))
    #print(nombre)
    compile(directory=nombre)
    M = Monitor(pop, ['spike'])
    input_index = []
    output_index = []
    n_entrada = int(n_entrada)
    n_salida = int(n_salida)
    for i in range(n_entrada):
        input_index.append(i)
    for i in range(n_entrada,n_salida+n_entrada):
        output_index.append(i)
    fit = fitness(pop,M,input_index,output_index,xor)
    return fit

def fitness(pop,Monitor,input_index,output_index,funcion):
    fit = 0
    fit = funcion(pop,Monitor,input_index,output_index)
    return fit
     

def xor(pop,Monitor,input_index,output_index):
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fitness = 0
    for entrada in entradas:
        for i, val in zip(input_index, entrada):
            if val == 1:
                pop[int(i)].I = 20
            else:
                pop[int(i)].I = 0
        simulate(10000.0)
        spikes = Monitor.get('spike')
        print("spikes: ",spikes) 
        #print("entradas: ",entrada)
        #Get the output
        output = 0
        for i in output_index:
            output += len(spikes[i])
        #print("spike output: ",output)
        #Get the average spikes of all neurons
        average = 0
        for i in range(len(pop)):
            average += len(spikes[i])
        average = average/len(pop)
        #print("average spikes: ",average)
        decode_output = -2
        if output > average:
            decode_output = 1
        if output <= average:
            decode_output = 0
        #t, n = Monitor.raster_plot(spikes)
        #plt.plot(t, n, 'b.')
        #plt.title('Raster plot')
        #plt.show()



        #comparar las entradas y la salida esperada con el output
        if entrada[0] ^ entrada[1] == decode_output:
            fitness += 1
        reset(pop)
    #fitness = rd.randint(1,4)
    return fitness
        
def exampleIzhikevich(): 
    pop = Population(geometry=1000, neuron=Izhikevich)
    excSize = int(800)
    Exc = pop[:800]
    Inh = pop[800:]
    print("2")

    re = np.random.random(800)      ; ri = np.random.random(200)
    Exc.noise = 5.0                 ; Inh.noise = 2.0
    Exc.a = 0.02                    ; Inh.a = 0.02 + 0.08 * ri
    Exc.b = 0.2                     ; Inh.b = 0.25 - 0.05 * ri
    Exc.c = -65.0 + 15.0 * re**2    ; Inh.c = -65.0
    Exc.d = 8.0 - 6.0 * re**2       ; Inh.d = 2.0
    Exc.v = -65.0                   ; Inh.v = -65.0
    Exc.u = Exc.v * Exc.b           ; Inh.u = Inh.v * Inh.b
    exc_proj = Projection(pre=Exc, post=pop, target='exc')
    exc_proj.connect_all_to_all(weights=Uniform(0.0, 0.5))

    inh_proj = Projection(pre=Inh, post=pop, target='inh')
    inh_proj.connect_all_to_all(weights=Uniform(0.0, 1.0))

    compile()
    M = Monitor(pop, ['spike', 'v'])

    simulate(3000.0, measure_time=True)
    spikes = M.get('spike')
    v = M.get('v')
    t, n = M.raster_plot(spikes)
    fr = M.histogram(spikes)
    print(spikes[0])
    fig = plt.figure(figsize=(12, 12))

    # First plot: raster plot
    plt.subplot(311)
    plt.plot(t, n, 'b.')
    plt.title('Raster plot')

    # Second plot: membrane potential of a single excitatory cell
    plt.subplot(312)
    plt.plot(v[:, 15]) # for example
    plt.title('Membrane potential')

    # Third plot: number of spikes per step in the population.
    plt.subplot(313)
    plt.plot(fr)
    plt.title('Number of spikes')
    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()
    return 0


def snn2(n_entrada, n_salida, n, i, matrix): 
    clear()
    pop = Population(geometry=n, neuron=LIF)
    proj = Projection(pre=pop, post=pop, target='exc')
    #print(matrix,"\n")
    #Matrix to numpy array
    matrix = np.array(matrix)

    proj.connect_from_matrix(matrix)
    #print('nombre')
    nombre = 'annarchy/annarchy-'+str(int(i))
    #print(nombre)
    compile(directory=nombre)
    M = Monitor(pop, ['spike'])

    pop[0:n_entrada].I = 20.
    simulate(1000.0, measure_time=True)

    spikes = M.get('spike')
    t, n = M.raster_plot(spikes)
    plt.plot(t, n, 'b.')
    plt.title('Raster plot')
    print(spikes)
    plt.show()

    return 0

#main
p = 10.0
#matriz = [[0,0,p],[0,0,p],[0,0,0]]
matriz = [[0,0,0],[0,0,0],[p,p,0]]

snn2(2,1,3,0,matriz)