from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import random


# Funciones para calcular medidas en M/M/1
def mm1_measures(lam, mu):
    rho = lam / mu
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu - lam)
    Wq = rho / (mu - lam)
    return {
        'lambda': lam,
        'mu': mu,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq
    }

def mm1k_measures(lam, mu, K):
    rho = lam / mu
    # Verificar si rho es igual a 1
    if rho == 1:
        S = K + 1
    else:
        S = (1 - rho ** (K + 1)) / (1 - rho)
    # Calcular Pn
    pn_dict = {}
    for n in range(K + 1):
        Pn = (rho ** n) / S
        pn_dict[n] = Pn
    # Probabilidad de que el sistema esté lleno (PK)
    PK = pn_dict[K]
    # Tasa efectiva de llegada
    lam_e = lam * (1 - PK)
    # Número promedio de clientes en el sistema (L)
    L = sum(n * pn_dict[n] for n in range(K + 1))
    # Tiempo promedio en el sistema (W)
    W = L / lam_e if lam_e > 0 else 0
    # Tiempo promedio en cola (Wq)
    Wq = W - (1 / mu)
    # Número promedio de clientes en cola (Lq)
    Lq = L - (1 - pn_dict[0])
    return {
        'lambda': lam,
        'lambdae': lam_e,
        'mu': mu,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pk' : PK
    }

# Función para imprimir resultados
def print_results(department, measures):
    print(f"\nResultados para el Nodo {department}:")
    print(f"Tasa de llegada (lambda): {measures['lambda']:.4f} clientes/hora")
    print(f"Tasa de servicio (mu): {measures['mu']:.4f} clientes/hora")
    print(f"Utilización (rho): {measures['rho']:.4f}")
    print(f"Número promedio en el sistema (L): {measures['L']:.4f} clientes")
    print(f"Número promedio en cola (Lq): {measures['Lq']:.4f} clientes")
    print(f"Tiempo promedio en el sistema (W): {measures['W']:.4f} horas ({measures['W']*60:.2f} minutos)")
    print(f"Tiempo promedio en cola (Wq): {measures['Wq']:.4f} horas ({measures['Wq']*60:.2f} minutos)")

def print_resultsK(department, measures):
    print(f"\nResultados para el Nodo {department}:")
    print(f"Tasa de llegada (lambda): {measures['lambda']:.4f} clientes/hora")
    print(f"Tasa de efectiva de llegada (lambdae): {measures['lambdae']:.4f} clientes/hora")
    print(f"Tasa de servicio (mu): {measures['mu']:.4f} clientes/hora")
    print(f"Utilización (rho): {measures['rho']:.4f}")
    print(f"Número promedio en el sistema (L): {measures['L']:.4f} clientes")
    print(f"Número promedio en cola (Lq): {measures['Lq']:.4f} clientes")
    print(f"Tiempo promedio en el sistema (W): {measures['W']:.4f} horas ({measures['W']*60:.2f} minutos)")
    print(f"Tiempo promedio en cola (Wq): {measures['Wq']:.4f} horas ({measures['Wq']*60:.2f} minutos)")
    print(f"Probabilidad de que el sistema esté lleno (PK): {measures['Pk']:.4f}%")

def move_nodes(service_start_times, service_end_times, estadoNodos, service_times, probs):
    datos = []
    for i in range(len(estadoNodos)):
        if estadoNodos[i] != -1:
            datos.append((i,estadoNodos[i]))
            estadoNodos[i] = -1

    for nodo, usuarioActual in datos:
        #Tiempo de salida = Tiempo en el que empiza + tiempo que dura el servicio
        service_end_times[nodo][usuarioActual] = service_start_times[nodo][usuarioActual] + service_times[nodo][usuarioActual]

        nodo_seleccionado = -1  # Se inicializa con el nodo actual ya que ese si o si tiene algo
        while estadoNodos[nodo_seleccionado] != -1:  # Mientras el nodo siguiente no este vacio se sigue ejecutando
            # Si devuelve el mismo nodo en el que esta se considera que se sale del sistema
            nodo_seleccionado = random.choices(range(len(estadoNodos)), weights=probs[nodo], k=1)[0]
        estadoNodos[nodo] = -1  # vacías ese nodo
        if nodo_seleccionado != nodo:  # si el resultado es ir a otro nodo
            # El tiempo de llegada es el momento en el que sale del anterior
            service_start_times[nodo_seleccionado][usuarioActual] = service_end_times[nodo][usuarioActual]
            estadoNodos[nodo_seleccionado] = usuarioActual


def simulate_priority_queue_MM1(lambda_rate, mu_rate, num_customers, priority_levels, probs, maxTime):
    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)

    service_times = []
    for i in range(5):
        service_times.append(np.random.exponential(1 / mu_rate[i], num_customers))

    priorities = np.random.choice(range(priority_levels), size=num_customers, p=[0.7, 0.3])

    # Calcular tiempos de llegada acumulativos
    arrival_times = np.cumsum(inter_arrival_times)

    # Inicializar matrices para tiempos de servicio
    service_start_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                           np.zeros(num_customers), np.zeros(num_customers)]
    service_end_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                         np.zeros(num_customers), np.zeros(num_customers)]
    wait_times = np.zeros(num_customers)

    instancias_cola_baja = np.zeros(num_customers)
    instancias_cola_alta = np.zeros(num_customers)

    # Inicializar colas para cada nivel de prioridad
    queues = [deque() for _ in range(priority_levels)]

    # Inicializar tiempo de disponibilidad del servidor
    actualTime = 0
    # Estado de los nodos para ir metiendo
    estadoNodos = [-1, -1, -1, -1, -1]
    abandonados = 0
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
                ok = False
                for p in range(priority_levels):
                    if queues[p]:
                        while queues[p]:
                            next_customer = queues[p].popleft()
                            #Comprobar tiempo máximo
                            if actualTime - arrival_times[next_customer] > maxTime:
                                for j in range(5):
                                    service_start_times[j][next_customer] = -1
                                    service_end_times[j][next_customer] = -1
                                    service_times[j][next_customer] = -1
                                wait_times[next_customer] = -1
                                arrival_times[next_customer] = -1
                                priorities[next_customer] = -1
                                abandonados += 1
                            else:
                                ok = True
                                break
                        if not ok: #Si el next_customer no tiene valor
                            next_customer = None
                            continue
                        if p == 1: # baja
                            instancias_cola_baja[next_customer] = len(queues[p])
                            instancias_cola_alta[next_customer] = -1
                        else:
                            instancias_cola_alta[next_customer] = len(queues[p])
                            instancias_cola_baja[next_customer] = -1
                        break

                if not next_customer or next_customer == None:
                    continue

                # Asignar tiempos de servicio
                service_start_times[0][next_customer] = max(arrival_times[next_customer], actualTime)
                wait_times[next_customer] = service_start_times[0][next_customer] - arrival_times[next_customer]
                estadoNodos[0] = next_customer

                move_nodes(service_start_times, service_end_times, estadoNodos, service_times, probs)

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
        if service_times[0][i] != -1:
            final_servicio = service_end_times[0][i]
            for j in range(1,5):
                if final_servicio < service_end_times[j][i]:
                    final_servicio = service_end_times[j][i]
            system_times.append(final_servicio - arrival_times[i])
        else:
            system_times.append(-1)

    # Calcular métricas
    # Filtrar valores 0 de service_end_times
    filtered_service_end_times = [
        [t for t in times if t > 0] for times in service_end_times
    ]
    filtered_service_times = [
        [t for t in times if t > 0] for times in service_times
    ]
    server_utilization = [np.sum(filtered_service_times[i]) / filtered_service_end_times[i][-1] if len(filtered_service_end_times[i]) > 0 else 0 for i in range(5)]

    return wait_times, system_times, priorities, server_utilization, instancias_cola_baja, instancias_cola_alta, abandonados

def calcularClientes(arrival_times,service_end_times,actualTime,priorities,estadoNodos):
    # Separar métricas por prioridad
    high_priority_indices = np.where(priorities == 0)[0]
    low_priority_indices = np.where(priorities == 1)[0]

    service_end_times_high = service_end_times[0][high_priority_indices]
    arrival_times_high = arrival_times[high_priority_indices]

    service_end_times_low = service_end_times[0][low_priority_indices]
    arrival_times_low = arrival_times[low_priority_indices]

    filtered_service_end_times_high = [
        [t for t in service_end_times_high if t > 0]
    ]
    filtered_arrival_times_high = [
        [t for t in arrival_times_high if t > 0]
    ]
    filtered_service_end_times_low = [
        [t for t in service_end_times_low if t > 0]
    ]
    filtered_arrival_times_low = [
        [t for t in arrival_times_low if t > 0]
    ]

    sum = [0, 0, 0]
    i = 0 #El último cliente terminado
    while i < len(filtered_service_end_times_high) and service_end_times_high[i] != 0:
        i += 1
    if i < len(filtered_service_end_times_high):
        for j in range(i,len(filtered_arrival_times_high)):
            if 0 < filtered_arrival_times_high[j][0] <= actualTime:
                sum[0] += 1
            else:
                break
    i = 0  # El último cliente terminado
    while i < len(filtered_service_end_times_low) and filtered_service_end_times_low[i] != 0:
        i += 1
    if i < len(filtered_service_end_times_low):
        for j in range(i, len(filtered_arrival_times_low)):
            if 0 < filtered_arrival_times_low[j][0] <= actualTime:
                sum[1] += 1
            else:
                break
    for i in estadoNodos:
        if i != -1:
            sum[2] += 1
    return sum

def simulate_priority_queue_MM1K(lambda_rate, mu_rate, num_customers, priority_levels, probs, K, maxTime):
    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)

    service_times = []
    for i in range(5):
        service_times.append(np.random.exponential(1 / mu_rate[i], num_customers))

    priorities = np.random.choice(range(priority_levels), size=num_customers, p=[0.7, 0.3])

    # Calcular tiempos de llegada acumulativos
    arrival_times = np.cumsum(inter_arrival_times)

    # Inicializar matrices para tiempos de servicio
    service_start_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                           np.zeros(num_customers), np.zeros(num_customers)]
    service_end_times = [np.zeros(num_customers), np.zeros(num_customers), np.zeros(num_customers),
                         np.zeros(num_customers), np.zeros(num_customers)]
    wait_times = np.zeros(num_customers)

    instancias_cola_baja = np.zeros(num_customers)
    instancias_cola_alta = np.zeros(num_customers)

    # Inicializar colas para cada nivel de prioridad
    queues = [deque() for _ in range(priority_levels)]

    # Inicializar tiempo de disponibilidad del servidor
    actualTime = 0

    # Estado de los nodos para ir metiendo
    estadoNodos = [-1, -1, -1, -1, -1]

    #
    abandonadosPorTiempo = 0 #Abandonados por tiempo
    abandonadosPorBloqueo = 0 #Abandonados por bloqueo
    # Inicializar índice de llegada
    i = 0
    while i < num_customers or any(queues[p] for p in range(priority_levels)):
        if i < num_customers and arrival_times[i] <= actualTime:
            # Añadir cliente a la cola correspondiente
            sum = calcularClientes(arrival_times, service_end_times, actualTime, priorities, estadoNodos)
            if sum[0] + sum[1] + sum[2] <= K:  # Si hay espacio en el sistema
                queues[priorities[i]].append(i)
            else:
                # Asignar tiempos de servicio negativos, ya que no se atiende
                for j in range(5):
                    service_start_times[j][i] = -1
                    service_end_times[j][i] = -1
                    service_times[j][i] = -1
                wait_times[i] = -1
                arrival_times[i] = -1
                priorities[i] = -1
                abandonadosPorBloqueo += 1
            i += 1
        else:
            if any(queues[p] for p in range(priority_levels)):
                # Encontrar la cola con la más alta prioridad disponible
                ok = False
                for p in range(priority_levels):
                    if queues[p]:
                        while queues[p]:
                            next_customer = queues[p].popleft()
                            # Comprobar tiempo máximo
                            if actualTime - arrival_times[next_customer] > maxTime:
                                for j in range(5):
                                    service_start_times[j][next_customer] = -1
                                    service_end_times[j][next_customer] = -1
                                    service_times[j][next_customer] = -1
                                wait_times[next_customer] = -1
                                arrival_times[next_customer] = -1
                                priorities[next_customer] = -1
                                abandonadosPorTiempo += 1
                            else:
                                ok = True
                                break
                        if not ok:  # Si el next_customer no tiene valor
                            next_customer = None
                            continue
                        if p == 1:  # baja
                            instancias_cola_baja[next_customer] = len(queues[p])
                            instancias_cola_alta[next_customer] = -1
                        else:
                            instancias_cola_alta[next_customer] = len(queues[p])
                            instancias_cola_baja[next_customer] = -1
                        break

                if not next_customer or next_customer == None:
                    continue

                # Asignar tiempos de servicio
                service_start_times[0][next_customer] = max(arrival_times[next_customer], actualTime)
                wait_times[next_customer] = service_start_times[0][next_customer] - arrival_times[next_customer]
                estadoNodos[0] = next_customer

                move_nodes(service_start_times, service_end_times, estadoNodos, service_times, probs)

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
        if service_times[0][i] != -1:
            final_servicio = service_end_times[0][i]
            for j in range(1, 5):
                if final_servicio < service_end_times[j][i]:
                    final_servicio = service_end_times[j][i]
            system_times.append(final_servicio - arrival_times[i])
        else:
            system_times.append(-1)

    # Calcular métricas
    # Filtrar valores 0 de service_end_times
    filtered_service_end_times = [
        [t for t in times if t > 0] for times in service_end_times
    ]
    filtered_service_times = [
        [t for t in times if t > 0] for times in service_times
    ]
    server_utilization = [np.sum(service_times[i]) / filtered_service_end_times[i][-1] if filtered_service_end_times[i][-1] > 0 else 0 for i in range(5)]

    return wait_times, system_times, priorities, server_utilization, instancias_cola_baja, instancias_cola_alta, abandonadosPorBloqueo, abandonadosPorTiempo

if __name__ == '__main__':

    num = 0  # para el bucle por si no introduce numero bien
    while num == 0:
        print("Que modelo usar: 1. M/M/1     2. M/M/1/K")
        entrada = int(input())
        if (entrada == 1):
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
                maxTime = float(lines[7])
            num = 1
            #Datos teóricos
            # Datos del sistema
            # Cálculo de las utilizaciones
            rho = [landa/mu[i] for i in range(5)]
            # Cálculo de medidas para cada departamento
            measures = [mm1_measures(landa, mu[i]) for i in range(5)]
            # Imprimir resultados
            print("\nValores matemáticos: \n")
            for i in range(5):
                print_results(str(i), measures[i])

            #Simulación
            # Parámetros
            num_customers = 10000
            priority_levels = 2  # 0 - Alta, 1 - Baja
            # Ejecutar la simulación
            wait_times, system_times, priorities, utilization, promedioColaBaja, promedioColaAlta, abandonados = simulate_priority_queue_MM1(landa,mu,num_customers,priority_levels, probs, maxTime)

        elif (entrada == 2):
            with open('instancia2.txt', 'r') as file:
                lines = file.readlines()
                landa = int(lines[0])
                mu = list(map(int, lines[1].split()))
                k = int(lines[2])
                probs = [list(map(float, lines[3].split())),
                         list(map(float, lines[4].split())),
                         list(map(float, lines[5].split())),
                         list(map(float, lines[6].split())),
                         list(map(float, lines[7].split()))
                         ]
                maxTime = float(lines[8])

            # Datos teóricos
            # Datos del sistema
            # Cálculo de las utilizaciones
            rho = [landa / mu[i] for i in range(5)]
            # Cálculo de medidas para cada departamento
            measures = [mm1k_measures(landa, mu[i],k) for i in range(5)]
            # Imprimir resultados
            print("\nValores matemáticos: \n")
            for i in range(5):
                print_resultsK(str(i), measures[i])

            #Simulación
            # Parámetros
            num_customers = 10000
            priority_levels = 2  # 0 - Alta, 1 - Baja
            # Ejecutar la simulación
            wait_times, system_times, priorities, utilization, promedioColaBaja, promedioColaAlta, abandonadosPorBloqueo, abandonados = simulate_priority_queue_MM1K(
                landa, mu, num_customers, priority_levels, probs, k, maxTime)
            abandonados += abandonadosPorBloqueo
        # Separar métricas por prioridad

        high_priority_indices = np.where(priorities == 0)[0]
        low_priority_indices = np.where(priorities == 1)[0]

        system_times = np.array(system_times)

        wait_times_high = wait_times[high_priority_indices]
        system_times_high = system_times[high_priority_indices]
        #people_inQueue_high = promedioCola[high_priority_indices]

        wait_times_low = wait_times[low_priority_indices]
        system_times_low = system_times[low_priority_indices]
        #people_inQueue_low = promedioCola[low_priority_indices]

        filtered_wait_times = [
            [t for t in wait_times if t > 0]
        ]
        filtered_system_times = [
            [t for t in system_times if t > 0]
        ]
        filtered_wait_times_high = [
            [t for t in wait_times_high if t > 0]
        ]
        filtered_system_times_high = [
            [t for t in system_times_high if t > 0]
        ]
        filtered_wait_times_low = [
            [t for t in wait_times_low if t > 0]
        ]
        filtered_system_times_low = [
            [t for t in system_times_low if t > 0]
        ]

        # Filtrar valores 0 de promedioColaBaja y promedioColaAlta
        filtered_T_cola_alta = [t for t in promedioColaAlta if t >= 0]
        filtered_T_cola_baja = [t for t in promedioColaBaja if t >= 0]

        # Imprimir métricas generales
        print("\nValores generales: \n")
        print(f"Tiempo de espera promedio en cola (General): {np.mean(filtered_wait_times):.2f} horas")
        print(f"Tiempo promedio en el sistema (General): {np.mean(filtered_system_times):.2f} horas")
        print(f"Número promedio en la cola (General): {(np.mean(filtered_T_cola_baja)+np.mean(filtered_T_cola_alta))/2:.0f} \n")
        for i in range(5):
            print(f"Utilización del servidor en el nodo {i}: {utilization[i]:.2%}\n", end= '')
        print(f"Tasa de abandono: {abandonados / num_customers:.4f}")
        if entrada == 2:
            print(f"Probabilidad de bloqueo: {abandonadosPorBloqueo / num_customers * 100:.2f}%")
        print()

        # Imprimir métricas por prioridad
        print(f"--- Métricas para Prioridad Alta (0) ---")
        print(f"Tiempo de espera promedio en cola (Alta): {np.mean(filtered_wait_times_high):.2f} horas")
        print(f"Tiempo promedio en el sistema (Alta): {np.mean(filtered_system_times_high):.2f} horas")
        print(f"Número promedio en la cola (Alta): {np.mean(filtered_T_cola_alta):.0f} \n")

        print(f"--- Métricas para Prioridad Baja (1) ---")
        print(f"Tiempo de espera promedio en cola (Baja): {np.mean(filtered_wait_times_low):.2f} horas")
        print(f"Tiempo promedio en el sistema (Baja): {np.mean(filtered_system_times_low):.2f} horas")
        print(f"Número promedio en la cola (Baja): {np.mean(filtered_T_cola_baja):.0f}\n")

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

        break
