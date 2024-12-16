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

def simulate_fifo_queue_MM1K(lambda_rate, mu_rate, num_customers, k):
    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)
    service_times = np.random.exponential(1 / mu_rate, num_customers)

    # Inicializar variables
    arrival_times = np.cumsum(inter_arrival_times) # array con tiempos acumulativos de tiempos de llegada

    service_start_times = np.zeros(num_customers) # array de 0 para el inicio de tiempos de servicio
    service_end_times = np.zeros(num_customers) # array de 0 para el fin de tiempos de servicio

    people_queued = np.zeros(num_customers) # array de 0 para el numero de gente en cola
    people_inSys = np.zeros(num_customers) # array de 0 para el numero de gente siendo atendida
    peopleInQueue = list # guardar tiempos de fin de servicio para saber que gente esta dentro
    peopleInShop = list

    wait_times = np.zeros(num_customers) # array de 0 para los tiempos de espera de cada cliente

    # Simulación del sistema de colas
    for i in range(num_customers):
        if i == 0:
            service_start_times[i] = arrival_times[i]
        else:
            service_start_times[i] = max(arrival_times[i], service_end_times[i - 1])

        wait_times[i] = service_start_times[i] - arrival_times[i]
        service_end_times[i] = service_start_times[i] + service_times[i]

    # Calcular tiempos en el sistema
    system_times = service_end_times - arrival_times

    # Métricas
    #average_wait = np.mean(wait_times)
    #average_system_time = np.mean(system_times)
    server_utilization = np.sum(service_times) / service_end_times[-1]

    return wait_times, system_times, server_utilization

def simulate_fifo_queue_MM1(lambda_rate, mu_rate, num_customers):
    # Generar tiempos entre llegadas y tiempos de servicio
    inter_arrival_times = np.random.exponential(1 / lambda_rate, num_customers)
    service_times = np.random.exponential(1 / mu_rate, num_customers)

    # Inicializar variables
    arrival_times = np.cumsum(inter_arrival_times)
    service_start_times = np.zeros(num_customers)
    service_end_times = np.zeros(num_customers)
    wait_times = np.zeros(num_customers)

    # Simulación del sistema de colas
    for i in range(num_customers):
        if i == 0:
            service_start_times[i] = arrival_times[i]
        else:
            service_start_times[i] = max(arrival_times[i], service_end_times[i - 1])
        wait_times[i] = service_start_times[i] - arrival_times[i]
        service_end_times[i] = service_start_times[i] + service_times[i]

    # Calcular tiempos en el sistema
    system_times = service_end_times - arrival_times

    # Métricas
    # average_wait = np.mean(wait_times)
    # average_system_time = np.mean(system_times)
    server_utilization = np.sum(service_times) / service_end_times[-1]

    return wait_times, system_times, server_utilization

if __name__ == '__main__':

    lambda_rate = 4  # tasa de llegada
    mu_rate = 5  # tasa de servicio
    num = 0 # para el bucle por si no introduce numero bien

    while num == 0:

        print("Que modelo usar: 1. M/M/1     2. M/M/1/K")
        entrada = int(input())
        if (entrada == 1):
            num = 1
            resultado_mm1 = mm1_model(lambda_rate, mu_rate)
            print("Resultados M/M/1:")
            for clave, valor in resultado_mm1.items():
                print(f"{clave}: {valor:.4f}")
            # Parámetros
            num_customers = 10000

            # Ejecutar la simulación
            wait_times, system_times, utilization = simulate_fifo_queue_MM1(lambda_rate, mu_rate, num_customers)

        elif (entrada == 2):
            num = 1
            K = 10000
            results_mm1k, pn_dict = mm1k_model(lambda_rate, mu_rate, K)

            print("\nResultados para el Modelo M/M/1/K:")
            for key, value in results_mm1k.items():
                if 'Tiempo' in key:
                    print(f"{key}: {value:.4f} horas ({value * 60:.2f} minutos)")
                elif 'λe' in key:
                    print(f"{key}: {value:.4f} clientes/hora")
                else:
                    print(f"{key}: {value:.4f}")

            print("\nProbabilidades Pn (n = 0 a {0}):".format(K))
            for n, pn in pn_dict.items():
               print(f"P_{n}: {pn:.4f}")
            # Parámetros
            num_customers = 10000

            # Ejecutar la simulación
            wait_times, system_times, utilization = simulate_fifo_queue_MM1K(lambda_rate, mu_rate, num_customers, K)

print(f"Tiempo de espera promedio en cola (FIFO): {np.mean(wait_times):.2f} horas")
print(f"Tiempo promedio en el sistema (FIFO): {np.mean(system_times):.2f} horas")
print(f"Utilización del servidor (FIFO): {utilization:.2%}")

# Visualizar la distribución de tiempos de espera
plt.figure(figsize=(10, 6))
plt.hist(wait_times, bins=50, density=True, edgecolor='black', alpha=0.7)
plt.title('Distribución de tiempos de espera en cola (FIFO)')
plt.xlabel('Tiempo de espera (horas)')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()

# Visualizar la distribución de tiempos en el sistema
plt.figure(figsize=(10, 6))
plt.hist(system_times, bins=50, density=True, edgecolor='black', alpha=0.7)
plt.title('Distribución de tiempos en el sistema (FIFO)')
plt.xlabel('Tiempo en el sistema (horas)')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()
