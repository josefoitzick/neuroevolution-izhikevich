from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.sparse
import gymnasium as gym
from scipy.special import erf

_NEURON_CFG_CACHE = None

def _get_neuron_cfg():
    global _NEURON_CFG_CACHE
    if _NEURON_CFG_CACHE is not None:
        return _NEURON_CFG_CACHE
    params = {}
    try:
        with open('config/neuron.cfg') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    params[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    _NEURON_CFG_CACHE = params
    return params



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

def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial):
    try:
        ncfg = _get_neuron_cfg()
        clear()
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        pop.a = float(ncfg.get('a', 0.02))
        pop.b = float(ncfg.get('b', 0.2))
        pop.c = float(ncfg.get('c', -65.0))
        pop.d = float(ncfg.get('d', 8.0))
        proj = Projection(pre=pop, post=pop, target='exc')
        #Matrix to numpy array
         # Verificar el tamaño de la matrix
        if matrix.size == 0:
            raise ValueError("matrix is empty")
        #lil_matrix scipy nxn with values of matrix
        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        nombre = 'annarchy/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))
        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])
        input_index = []
        output_index = []
        n_entrada = int(n_entrada)
        n_salida = int(n_salida)
        for i in range(n_entrada):
            input_index.append(i)
        for i in range(n_entrada,n_salida+n_entrada):
            output_index.append(i)
        # Verificar el tamaño de inputWeights
        if inputWeights.size == 0:
            raise ValueError("inputWeights is empty")

        funcion = get_function('results/trial-'+ str(int(trial)))
        fit = fitness(pop,M,input_index,output_index, funcion, inputWeights)
        #return fit
        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)

def fitness(pop, Monitor, input_index, output_index, funcion, inputWeights):
    if funcion == "xor":
        return xor(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole":
        return cartpole(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "lunar_lander":
        return lunar_lander(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole2":
        return cartpole2(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole3":
        return cartpole3(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "acrobot":
        return acrobot(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "mountaincar":
        return mountaincar(pop, Monitor, input_index, output_index, inputWeights)
    else:
        raise ValueError(f"Unknown function: {funcion}")


def get_function(folder):
    # Open config file and get the parameter "function"
    config_path = folder + '/config.cfg'
    with open(config_path) as f:
        lines = f.readlines()
        for line in lines:
            if "function" in line:
                return line.split('=')[1].strip()
    return None
     

def xor(pop,Monitor,input_index,output_index,inputWeights):
    Monitor.reset()
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fitness = 0
    for entrada in entradas:
        for i, val in zip(input_index, entrada):
            if val == 1:
                pop[int(i)].I = 15.1*inputWeights[i]
            else:
                pop[int(i)].I = 0
        simulate(10.0)
        spikes = Monitor.get('spike')
        #print("spikes: ",spikes) 
        #Get the output
        output = 0
        for i in output_index:
            output += np.size(spikes[i])
        #print("spike output: ",output)

        decode_output = 0
        if output > 1:
            decode_output = 1

        pop.reset()
        Monitor.reset()
        #comparar las entradas y la salida esperada con el output
        if entrada[0] ^ entrada[1] == decode_output:
            fitness += 1
    return fitness



def cartpole(pop,Monitor,input_index,output_index,inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 500
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    #Generar 4 input weights para cada input
    inputWeights = np.random.uniform(minInput,maxInput,4)
    #Number of episodes
    episodes = 100
    h=0
    #Final fitness 
    final_fitness = 0
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    pop[int(input_index[i])].I = -val*inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*inputWeights[k]
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 2 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2:
                action = 0
            elif output1 < output2:
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        env.reset()
        #print("Episode: ",h," Fitness: ",np.sum(returns))
        final_fitness += np.sum(returns)
        h += 1
    final_fitness = final_fitness/episodes
    #print("Final mean fitness: ",final_fitness,"\n")
    env.close()
    #print("Returns: ",returns)
    #print("Actions: ",actions_done)
    return final_fitness


def cartpole2(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness 
    final_fitness = 0
    
    # Definir límites para cada variable de observación
    limites = [
        (-4.8, 4.8),  # Posición del carro
        (-10.0, 10.0),  # Velocidad del carro (estimado)
        (-0.418, 0.418),  # Ángulo del poste en radianes
        (-10.0, 10.0)  # Velocidad angular del poste (estimado)
    ]
    
    num_neuronas_por_variable = 20
    intervals = []

    for low, high in limites:
        # Generar valores centrados en 0 siguiendo una distribución normal
        values = np.random.normal(loc=0, scale=1, size=1000)
        z = np.linspace(low, high, num_neuronas_por_variable + 1)
        interval_limits = np.percentile(values, (0.5 * (1 + erf(z / np.sqrt(2)))) * 100)
        # Dividir los valores en intervalos
        intervals = [values[(values >= interval_limits[i]) & (values < interval_limits[i+1])] for i in range(num_neuronas_por_variable)]
        intervals[-1] = np.append(intervals[-1], values[-1])  # Asegurar que el último intervalo incluye el valor máximo

    
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            # Codificar observación
            for i, obs in enumerate(observation):  # Primer ciclo: Itera sobre cada observación
                for j in range(num_neuronas_por_variable):
                    if obs >= interval_limits[j] and obs < interval_limits[j + 1]:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 20 # Activa la neurona correspondiente
                        break
            simulate(50.0)

            # Decodificar la acción basada en la cual neurona de salida tuvo la primera spike
            spikes = Monitor.get('spike')
            min_left = np.inf
            for i in output_index[:20]:
                if len(spikes[i]) > 0:
                    if min(spikes[i]) < min_left:
                        min_left = min(spikes[i])
            min_right = np.inf
            for i in output_index[20:]:
                if len(spikes[i]) > 0:
                    if min(spikes[i]) < min_right:
                        min_right = min(spikes[i])


            action = env.action_space.sample()
            if min_left < min_right:
                action = 0
            elif min_right < min_left:
                action = 1
        

            
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        env.reset()
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness


def lunar_lander(pop, Monitor, input_index, output_index, inputWeights):
    #funcion similar a cartpole, solo que con el entorno de lunar lander 16 entradas y 4 salidas
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    #Generar 8 input weights para cada input
    inputWeights = np.random.uniform(minInput,maxInput,8)
    #Number of episodes
    episodes = 100
    h=0
    #Final fitness
    final_fitness = 0
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        while j < max_steps and not terminated and not truncated:
            #encode observation, 8 values split in 16 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    pop[int(input_index[i])].I = -val*inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*inputWeights[k]
                i += 2
                k += 1
            simulate(100.0)
            spikes = Monitor.get('spike')
            #Output from 4 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            output4 = np.size(spikes[output_index[3]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3 and output1 > output4:
                action = 0
            elif output2 > output1 and output2 > output3 and output2 > output4:
                action = 1
            elif output3 > output1 and output3 > output2 and output3 > output4:
                action = 2
            elif output4 > output1 and output4 > output2 and output4 > output3:
                action = 3
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            Monitor.reset()
            j += 1
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness/episodes
    env.close()
    return final_fitness

def cartpole3(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness 
    final_fitness = 0
    
    # Definir límites para cada variable de observación
    limites = [
        (-4.8, 4.8),  # Posición del carro
        (-10.0, 10.0),  # Velocidad del carro (estimado)
        (-0.418, 0.418),  # Ángulo del poste en radianes
        (-10.0, 10.0)  # Velocidad angular del poste (estimado)
    ]
    
    num_neuronas_por_variable = 20
    intervals = []

    for low, high in limites:
        # Generar valores centrados en 0 siguiendo una distribución normal
        values = np.random.normal(loc=0, scale=1, size=1000)
        z = np.linspace(low, high, num_neuronas_por_variable + 1)
        interval_limits = np.percentile(values, (0.5 * (1 + erf(z / np.sqrt(2)))) * 100)
        # Dividir los valores en intervalos
        intervals = [values[(values >= interval_limits[i]) & (values < interval_limits[i+1])] for i in range(num_neuronas_por_variable)]
        intervals[-1] = np.append(intervals[-1], values[-1])  # Asegurar que el último intervalo incluye el valor máximo

    flag=True
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            # Codificar observación
            for i, obs in enumerate(observation):  # Primer ciclo: Itera sobre cada observación
                for j in range(num_neuronas_por_variable):
                    if obs >= interval_limits[j] and obs < interval_limits[j + 1]:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 75 # Activa la neurona correspondiente
                        break
            simulate(50.0)
            spikes = Monitor.get('spike')
            # Decodificar la acción basada en el número de picos en las neuronas de salida
            left_spikes = sum(np.size(spikes[idx]) for idx in output_index[:20])  # Neuronas que controlan el movimiento a la izquierda
            right_spikes = sum(np.size(spikes[idx]) for idx in output_index[20:])  # Neuronas que controlan el movimiento a la derecha
            
            action = env.action_space.sample()
            if left_spikes > right_spikes:
                action = 0  # Mover a la izquierda
            elif left_spikes < right_spikes:
                action = 1  # Mover a la derecha

            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            #resetear I=0, resetear a -65 (Iz valor de descanso)
            j += 1
        env.reset()
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness
        
def acrobot(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("Acrobot-v1")
    observation, info = env.reset(seed=42)
    max_steps = 500
    episodes = 100
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    inputWeights_rand = np.random.uniform(minInput, maxInput, 6)
    final_fitness = 0
    for h in range(episodes):
        terminated = False
        truncated = False
        observation, _ = env.reset()
        returns = []
        j = 0
        while j < max_steps and not terminated and not truncated:
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    pop[int(input_index[i])].I = -val * inputWeights_rand[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val * inputWeights_rand[k]
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            counts = [np.size(spikes[output_index[a]]) for a in range(3)]
            action = int(np.argmax(counts)) if max(counts) > 0 else env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            pop.reset()
            Monitor.reset()
            j += 1
        env.reset()
        final_fitness += np.sum(returns)
    final_fitness /= episodes
    env.close()
    return final_fitness


def normalize(value, min_val, max_val):
    return abs((value - min_val) / (max_val - min_val))


def mountaincar(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("MountainCar-v0")
    observation, info = env.reset()
    terminated = False
    truncated = False
    episodes = 100
    h = 0
    final_fitness = 0

    limits = [
        (-1.2, 0.6),    # Car position
        (-0.07, 0.07),  # Car velocity
    ]

    minInput = inputWeights[0]
    maxInput = inputWeights[1]
    inputWeights = np.random.uniform(minInput, maxInput, 4)

    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        observation, info = env.reset()
        while not terminated and not truncated:
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    normval = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = normval * inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    normval = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = normval * inputWeights[k]
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            action = env.action_space.sample()
            if output1 > output2:
                if output1 > output3:
                    action = 0
                else:
                    action = 2
            else:
                if output2 > output3:
                    action = 1
                else:
                    action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        final_fitness += np.sum(returns)
        h += 1
        pop.reset()
        Monitor.reset()
    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness


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




 #Example SNN for cartpole2
#snn(80,40,200,0,np.random.rand(200,200),np.random.rand(80),0)
