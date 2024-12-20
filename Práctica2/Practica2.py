from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
from fontTools.merge.util import first


def move_nodes(service_start_times, actualTime, service_end_times, estadoNodos, service_times, probs):
    for i in range(len(estadoNodos)):
        if estadoNodos[i] != -1:
            usuarioActual = estadoNodos[i]  # índice del nodo en el que estamos
            #Tiempo de salida = Tiempo en el que empiza + tiempo que dura el servicio
            service_end_times[i][usuarioActual] = service_start_times[0][usuarioActual] + service_times[i][usuarioActual]
            nodo_seleccionado = i  # Se inicializa con el nodo actual ya que ese si o si tiene algo
            while estadoNodos[nodo_seleccionado] != -1:  # Mientras el nodo siguiente no este vacio se sigue ejecutando
                # Si devuelve el mismo nodo en el que esta se considera que se sale del sistema
                nodo_seleccionado = random.choices(range(len(estadoNodos)), weights=probs[i], k=1)[0]
            estadoNodos[i] = -1  # vacias ese nodo
            if nodo_seleccionado != i:  # si el resultado es ir a otro nodo
                # El tiempo de llegada es el momento en el que sale del anterior
                service_start_times[nodo_seleccionado][usuarioActual] = service_end_times[i][usuarioActual]
                estadoNodos[nodo_seleccionado] = usuarioActual


def simulate_priority_queue_M51(lambda_rate, mu_rate, num_customers, priority_levels, probs):
    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)

    service_times = [[], [], [], [], []]
    for i in range(5):
        service_times[i] = np.random.exponential(1 / mu_rate[i], num_customers)

    priorities = np.random.choice(range(priority_levels), size=num_customers, p=[0.3, 0.7])

    # Calcular tiempos de llegada acumulativos
    arrival_times = np.cumsum(inter_arrival_times)

    # Inicializar matrices para tiempos de servicio
    service_start_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                           np.zeros(num_customers), np.zeros(num_customers)]
    service_end_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                         np.zeros(num_customers), np.zeros(num_customers)]
    wait_times = np.zeros(num_customers)

    instancias_cola = np.zeros(num_customers)
    people_in_queue = deque()

    # Inicializar colas para cada nivel de prioridad
    queues = [deque() for _ in range(priority_levels)]

    # Inicializar tiempo de disponibilidad del servidor
    actualTime = 0

    # Estado de los nodos para ir metiendo
    estadoNodos = [-1, -1, -1, -1, -1]

    # Inicializar índice de llegada
    i = 0
    while i < num_customers or any(queues[p] for p in range(priority_levels)):
        if i < num_customers and arrival_times[i] <= actualTime:
            # Añadir cliente a la cola correspondiente
            queues[priorities[i]].append(i)
            i += 1
        else:
            if any(queues[p] for p in range(priority_levels)):
                # Encontrar la cola con la más alta prioridad disponible
                for p in range(priority_levels):
                    if queues[p]:
                        next_customer = queues[p].popleft()
                        break

                # Asignar tiempos de servicio
                service_start_times[0][next_customer] = max(arrival_times[next_customer], actualTime)
                wait_times[next_customer] = service_start_times[0][next_customer] - arrival_times[next_customer]
                estadoNodos[0] = next_customer

                move_nodes(service_start_times, actualTime, service_end_times, estadoNodos, service_times, probs)

                # rellenar instancias de cola para hacer la media para el número promedio en cola
                # para ello metemos a la gente en la cola si su hora de llegada es menor que el actual

                #if arrival_times[next_customer] <= actualTime < service_start_times[0][next_customer]:
                    #people_in_queue.append(next_customer)
                    #instancias_cola[next_customer] = len(people_in_queue)

                # se quitan de la cola de la tienda si ya ha sido atendido
                #while first(people_in_queue) >= actualTime:
                    #people_in_queue.popleft()

                # Actualizar tiempo libre del servidor
                actualTime = service_end_times[0][next_customer]
            else:
                if i < num_customers:
                    # Avanzar el tiempo al siguiente evento de llegada
                    actualTime = arrival_times[i]
                else:
                    break

    # Calcular tiempos en el sistema
    system_times = []
    for i in range(num_customers):
        final_servicio = service_end_times[0][i]
        for j in range(1,5):
            if final_servicio < service_end_times[j][i]:
                final_servicio = service_end_times[j][i]
        system_times.append(final_servicio - arrival_times[i])

    # Calcular métricas
    # average_wait = np.mean(wait_times)
    # average_system_time = np.mean(system_times)
    server_utilization = [np.sum(service_times[0]) / service_end_times[0][-1] if service_end_times[0][-1] > 0 else 0,
                          np.sum(service_times[1]) / service_end_times[1][-1] if service_end_times[1][-1] > 0 else 0,
                          np.sum(service_times[2]) / service_end_times[2][-1] if service_end_times[2][-1] > 0 else 0,
                          np.sum(service_times[3]) / service_end_times[3][-1] if service_end_times[3][-1] > 0 else 0,
                          np.sum(service_times[4]) / service_end_times[4][-1] if service_end_times[4][-1] > 0 else 0
                          ]

    return wait_times, system_times, priorities, server_utilization, instancias_cola


if __name__ == '__main__':

    num = 0  # para el bucle por si no introduce numero bien
    while num == 0:
        #print("Que modelo usar: 1. M/M/1     2. M/M/1/K")
        #entrada = int(input())
        # if (entrada == 1):
        with open('instancia1.txt', 'r') as file:
            lines = file.readlines()
            landa = int(lines[0])
            mu = list(map(int, lines[1].split()))
            probs = [list(map(float, lines[2].split())),
                     list(map(float, lines[3].split())),
                     list(map(float, lines[4].split())),
                     list(map(float, lines[5].split())),
                     list(map(float, lines[6].split()))
                     ]
        num = 1
        # resultado_mm1 = mm1_model(valores[0], valores[1])
        # print("Resultados M/M/1:")
        # for clave, valor in resultado_mm1.items():
        # print(f"{clave}: {valor:.4f}")
        # Parámetros
        num_customers = 10000
        priority_levels = 2  # 0 - Alta, 1 - Baja
        # Ejecutar la simulación
        wait_times, system_times, priorities, utilization, promedioCola = simulate_priority_queue_M51(landa,mu,num_customers,priority_levels, probs)

        # elif (entrada == 2):
        # with open('instancia2.txt', 'r') as file:
        # lines = file.readlines()
        # valores = list(map(int, lines[0].split()))  # 0 landa, 1 mu, 2 K
        # num = 1
        # results_mm1k, pn_dict = mm1k_model(valores[0], valores[0], valores[2])

        # print("\nResultados para el Modelo M/M/1/K:")
        # for key, value in results_mm1k.items():
        # if 'Tiempo' in key:
        # print(f"{key}: {value:.4f} horas ({value * 60:.2f} minutos)")
        # elif 'λe' in key:
        # print(f"{key}: {value:.4f} clientes/hora")
        # else:
        # print(f"{key}: {value:.4f}")

        # print("\nProbabilidades Pn (n = 0 a {0}):".format(valores[2]))
        # for n, pn in pn_dict.items():
        # print(f"P_{n}: {pn:.4f}")
        # Parámetros
        num_customers = 10000
        priority_levels = 2  # 0 - Alta, 1 - Baja
        # Ejecutar la simulación
        # wait_times, system_times, priorities, utilization, promedioCola = simulate_priority_queue_MM1K(valores[0], valores[1], num_customers,valores[2],priority_levels)

        # Separar métricas por prioridad
        high_priority_indices = np.where(priorities == 0)[0]
        low_priority_indices = np.where(priorities == 1)[0]

        system_times = np.array(system_times)

        wait_times_high = wait_times[high_priority_indices]
        system_times_high = system_times[high_priority_indices]
        people_inQueue_high = promedioCola[high_priority_indices]

        wait_times_low = wait_times[low_priority_indices]
        system_times_low = system_times[low_priority_indices]
        people_inQueue_low = promedioCola[low_priority_indices]

        # Imprimir métricas generales
        print(f"Tiempo de espera promedio en cola (General): {np.mean(wait_times):.2f} horas")
        print(f"Tiempo promedio en el sistema (General): {np.mean(system_times):.2f} horas")
        print(f"Numero promedio en la cola (General): {np.mean(promedioCola):} \n")
        for i in range(5):
            print(f"Utilización del servidor en el nodo {i}: {utilization[i]:.2%}\n", end= '')
        print()

        # Imprimir métricas por prioridad
        print(f"--- Métricas para Prioridad Alta (0) ---")
        print(f"Tiempo de espera promedio en cola (Alta): {np.mean(wait_times_high):.2f} horas")
        print(f"Tiempo promedio en el sistema (Alta): {np.mean(system_times_high):.2f} horas")
        print(f"Numero promedio en la cola (Alta): {np.mean(people_inQueue_high):} \n")

        print(f"--- Métricas para Prioridad Baja (1) ---")
        print(f"Tiempo de espera promedio en cola (Baja): {np.mean(wait_times_low):.2f} horas")
        print(f"Tiempo promedio en el sistema (Baja): {np.mean(system_times_low):.2f} horas")
        print(f"Numero promedio en la cola (Baja): {np.mean(people_inQueue_low):}\n")

        # Visualizar la distribución de tiempos de espera
        plt.figure(figsize=(12, 8))
        plt.hist(wait_times_high, bins=50, density=True, edgecolor='black', alpha=0.7, label='Prioridad Alta')
        plt.hist(wait_times_low, bins=50, density=True, edgecolor='black', alpha=0.5, label='Prioridad Baja')
        plt.title('Distribución de tiempos de espera en cola por prioridad')
        plt.xlabel('Tiempo de espera (horas)')
        plt.ylabel('Densidad de probabilidad')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Visualizar la distribución de tiempos en el sistema
        plt.figure(figsize=(12, 8))
        plt.hist(system_times_high, bins=50, density=True, edgecolor='black', alpha=0.7, label='Prioridad Alta')
        plt.hist(system_times_low, bins=50, density=True, edgecolor='black', alpha=0.5, label='Prioridad Baja')
        plt.title('Distribución de tiempos en el sistema por prioridad')
        plt.xlabel('Tiempo en el sistema (horas)')
        plt.ylabel('Densidad de probabilidad')
        plt.legend()
        plt.grid(True)
        plt.show()
