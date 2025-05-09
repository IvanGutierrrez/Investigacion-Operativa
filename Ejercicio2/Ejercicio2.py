from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def mm1_model(lam, mu):
    rho = lam / mu
    if rho >= 1:
        return "El sistema es inestable (rho >= 1)."
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu - lam)
    Wq = rho / (mu - lam)
    results = {
        'Utilización (rho)': rho,
        'Número promedio en el sistema (L)': L,
        'Número promedio en cola (Lq)': Lq,
        'Tiempo promedio en el sistema (W)': W,
        'Tiempo promedio en cola (Wq)': Wq
    }
    return results

def mm1k_model(lam, mu, K):

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

    results = {
        'Factor de utilización (rho)': rho,
        'Probabilidad de que el sistema esté lleno (PK)': PK,
        'Tasa efectiva de llegada (λe)': lam_e,
        'Número promedio de clientes en el sistema (L)': L,
        'Número promedio de clientes en cola (Lq)': Lq,
        'Tiempo promedio en el sistema (W)': W,
        'Tiempo promedio en cola (Wq)': Wq
    }
    return results, pn_dict

def calcularClientes(arrival_times,service_end_times,actualTime,priorities):
    # Separar métricas por prioridad
    high_priority_indices = np.where(priorities == 0)[0]
    low_priority_indices = np.where(priorities == 1)[0]

    service_end_times_high = service_end_times[high_priority_indices]
    arrival_times_high = arrival_times[high_priority_indices]

    service_end_times_low = service_end_times[low_priority_indices]
    arrival_times_low = arrival_times[low_priority_indices]

    sum = [0, 0]
    i = 0 #El último cliente terminado
    while i < len(service_end_times_high) and service_end_times_high[i] != 0:
        i += 1
    if i < len(service_end_times_high):
        for j in range(i,len(arrival_times_high)):
            if 0 < arrival_times_high[j] <= actualTime:
                sum[0] += 1
            else:
                break
    i = 0  # El último cliente terminado
    while i < len(service_end_times_low) and service_end_times_low[i] != 0:
        i += 1
    if i < len(service_end_times_low):
        for j in range(i, len(arrival_times_low)):
            if 0 < arrival_times_low[j] <= actualTime:
                sum[1] += 1
            else:
                break
    return sum

def simulate_priority_queue_MM1K(lambda_rate, mu_rate, num_customers, K, priority_levels):

    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)
    service_times = np.random.exponential(1 / mu_rate, num_customers)
    priorities = np.random.choice(range(priority_levels), size=num_customers, p=[0.7, 0.3])

    # Calcular tiempos de llegada acumulativos
    arrival_times = np.cumsum(inter_arrival_times)

    # Inicializar matrices para tiempos de servicio
    service_start_times = np.zeros(num_customers)
    service_end_times = np.zeros(num_customers)
    wait_times = np.zeros(num_customers)

    # Para sacar número promedio en cada cola.
    instancias_cola_baja = np.zeros(num_customers)
    instancias_cola_alta = np.zeros(num_customers)

    # Inicializar colas para cada nivel de prioridad
    queues = [deque() for _ in range(priority_levels)]

    # Inicializar tiempo de disponibilidad del servidor
    actualTime = 0

    # Inicializar índice de llegada
    i = 0

    while i < num_customers or any(queues[p] for p in range(priority_levels)):

        #RELLENAR COLA
        if i < num_customers and arrival_times[i] <= actualTime:
            # Añadir cliente a la cola correspondiente
            sum = calcularClientes(arrival_times,service_end_times,actualTime,priorities)
            if sum[0] + sum[1] <= K: #Si hay espacio en el sistema
                queues[priorities[i]].append(i)
                i += 1
            else:
                # Asignar tiempos de servicio negativos, ya que no se atiende
                service_start_times = np.delete(service_start_times, i)
                wait_times = np.delete(wait_times, i)
                service_end_times = np.delete(service_end_times, i)
                arrival_times = np.delete(arrival_times, i)
                service_times = np.delete(service_times, i)
                priorities = np.delete(priorities, i)
                num_customers -= 1

        #VACIAR COLA
        else:
            if any(queues[p] for p in range(priority_levels)):
                # Encontrar la cola con la más alta prioridad disponible
                for p in range(priority_levels):
                    if queues[p]:
                        next_customer = queues[p].popleft()
                        sum = calcularClientes(arrival_times,service_end_times,actualTime,priorities)
                        instancias_cola_baja[next_customer] = sum[0]
                        instancias_cola_alta[next_customer] = sum[1]
                        break

                # Asignar tiempos de servicio
                service_start_times[next_customer] = max(arrival_times[next_customer], actualTime)
                wait_times[next_customer] = service_start_times[next_customer] - arrival_times[next_customer]
                service_end_times[next_customer] = service_start_times[next_customer] + service_times[next_customer]

                # Actualizar tiempo libre del servidor
                actualTime = service_end_times[next_customer]
            else:
                if i < num_customers:
                    # Avanzar el tiempo al siguiente evento de llegada
                    actualTime = arrival_times[i]
                else:
                    break

    # Filtrar valores -1
    filtered_service_end = [t for t in service_end_times if t >= 0]
    filtered_arrival = [t for t in arrival_times if t >= 0]
    filtered_service = [t for t in service_times if t >= 0]

    # Calcular tiempos en el sistema
    system_times = np.array(filtered_service_end) - np.array(filtered_arrival)

    # Calcular métricas
    server_utilization = np.sum(filtered_service) / filtered_service_end[-1] if len(filtered_service_end) > 0 else 0

    return wait_times, system_times, priorities, server_utilization, instancias_cola_baja, instancias_cola_alta

def simulate_priority_queue_MM1(lambda_rate, mu_rate, num_customers, priority_levels):

    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)
    service_times = np.random.exponential(1 / mu_rate, num_customers)
    priorities = np.random.choice(range(priority_levels), size=num_customers, p=[0.3, 0.7])

    # Calcular tiempos de llegada acumulativos
    arrival_times = np.cumsum(inter_arrival_times)

    # Inicializar matrices para tiempos de servicio
    service_start_times = np.zeros(num_customers)
    service_end_times = np.zeros(num_customers)
    wait_times = np.zeros(num_customers)

    #Para sacar número promedio en cada cola.
    instancias_cola_baja = np.zeros(num_customers)
    instancias_cola_alta = np.zeros(num_customers)

    # Inicializar colas para cada nivel de prioridad
    queues = [deque() for _ in range(priority_levels)]

    # Inicializar tiempo de disponibilidad del servidor
    actualTime = 0

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
                        sum = calcularClientes(arrival_times, service_end_times, actualTime, priorities)
                        instancias_cola_baja[next_customer] = sum[0]
                        instancias_cola_alta[next_customer] = sum[1]
                        break

                # Asignar tiempos de servicio
                service_start_times[next_customer] = max(arrival_times[next_customer], actualTime)
                wait_times[next_customer] = service_start_times[next_customer] - arrival_times[next_customer]
                service_end_times[next_customer] = service_start_times[next_customer] + service_times[next_customer]

                # Actualizar tiempo libre del servidor
                actualTime = service_end_times[next_customer]
            else:
                if i < num_customers:
                    # Avanzar el tiempo al siguiente evento de llegada
                    actualTime = arrival_times[i]
                else:
                    break

    # Calcular tiempos en el sistema
    system_times = service_end_times - arrival_times

    # Calcular métricas
    server_utilization = np.sum(service_times) / service_end_times[-1] if service_end_times[-1] > 0 else 0

    return wait_times, system_times, priorities, server_utilization, instancias_cola_baja, instancias_cola_alta

if __name__ == '__main__':

    num = 0 # para el bucle por si no introduce numero bien
    while num == 0:

        print("Que modelo usar: 1. M/M/1     2. M/M/1/K")
        entrada = int(input())
        if (entrada == 1):
            with open('instancia1.txt', 'r') as file:
                lines = file.readlines()
                valores = list(map(int, lines[0].split())) # 0 landa, 1 mu, 2 num_customers, 3 priority_levels # 0 - Alta, 1 - Baja
            num = 1
            resultado_mm1 = mm1_model(valores[0], valores[1])
            print("Resultados M/M/1:")
            for clave, valor in resultado_mm1.items():
                print(f"{clave}: {valor:.4f}")
            # Ejecutar la simulación
            wait_times, system_times, priorities, utilization, promedioColaBaja, promedioColaAlta = simulate_priority_queue_MM1(valores[0], valores[1], valores[2], valores[3])

        elif (entrada == 2):
            with open('instancia2.txt', 'r') as file:
                lines = file.readlines()
                valores = list(map(int, lines[0].split())) # 0 landa, 1 mu, 2 num_customers, 3 priority_levels, 4 k
            num = 1
            results_mm1k, pn_dict = mm1k_model(valores[0], valores[1], valores[4])

            print("\nResultados para el Modelo M/M/1/K:")
            for key, value in results_mm1k.items():
                if 'Tiempo' in key:
                    print(f"{key}: {value:.4f} horas ({value * 60:.2f} minutos)")
                elif 'λe' in key:
                    print(f"{key}: {value:.4f} clientes/hora")
                else:
                    print(f"{key}: {value:.4f}")

            print("\nProbabilidades Pn (n = 0 a {0}):".format(valores[4]))
            for n, pn in pn_dict.items():
               print(f"P_{n}: {pn:.4f}")
            # Ejecutar la simulación
            wait_times, system_times, priorities, utilization, promedioColaBaja, promedioColaAlta = simulate_priority_queue_MM1K(valores[0], valores[1], valores[2], valores[4], valores[3])

# Separar métricas por prioridad
high_priority_indices = np.where(priorities == 0)[0]
low_priority_indices = np.where(priorities == 1)[0]

wait_times_high = wait_times[high_priority_indices]
system_times_high = system_times[high_priority_indices]

wait_times_low = wait_times[low_priority_indices]
system_times_low = system_times[low_priority_indices]

promedioCola = np.array(promedioColaAlta) + np.array(promedioColaBaja)

# Imprimir métricas generales
print("\nValores experimentales\n")
print(f"Tiempo de espera promedio en cola (General): {np.mean(wait_times):.2f} horas")
print(f"Tiempo promedio en el sistema (General): {np.mean(system_times):.2f} horas")
print(f"Número promedio en la cola (General): {np.mean(promedioCola):.2f}")
print(f"Utilización del servidor: {utilization:.2%}\n")

# Imprimir métricas por prioridad
print(f"--- Métricas para Prioridad Alta (0) ---")
print(f"Tiempo de espera promedio en cola (Alta): {np.mean(wait_times_high):.2f} horas")
print(f"Tiempo promedio en el sistema (Alta): {np.mean(system_times_high):.2f} horas")
print(f"Número promedio en la cola (Alta): {np.mean(promedioColaAlta):.2f} \n")

print(f"--- Métricas para Prioridad Baja (1) ---")
print(f"Tiempo de espera promedio en cola (Baja): {np.mean(wait_times_low):.2f} horas")
print(f"Tiempo promedio en el sistema (Baja): {np.mean(system_times_low):.2f} horas")
print(f"Número promedio en la cola (Baja): {np.mean(promedioColaBaja):.2f}\n")

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
