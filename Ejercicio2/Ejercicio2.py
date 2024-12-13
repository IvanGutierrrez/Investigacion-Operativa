import numpy as np

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

if __name__ == '__main__':

    lam = 5  # tasa de llegada
    mu = 7  # tasa de servicio

    print("Que modelo usar: 1. M/M/1     2. M/M/1/K")

    if (input() == 1):
        resultado_mm1 = mm1_model(lam, mu)
        print("Resultados M/M/1:")
        for clave, valor in resultado_mm1.items():
            print(f"{clave}: {valor:.4f}")

    else:
        K = 3
        results_mm1k, pn_dict = mm1k_model(lam, mu, K)

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

def generar_llegadas_y_servicios(tasa_llegadas, tasa_servicios, n_eventos):
    """
    Genera tiempos de llegada y servicio para un sistema M/M/1.

    Parámetros:
    - tasa_llegadas: Tasa promedio de llegadas (λ).
    - tasa_servicios: Tasa promedio de servicios (μ).
    - n_eventos: Número de llegadas a simular.

    Retorna:
    - tiempos_llegada: Lista de tiempos de llegada acumulados.
    - tiempos_servicio: Lista de tiempos de servicio para cada llegada.
    """
    # Generar tiempos entre llegadas (exponencial con tasa λ)
    tiempos_entre_llegadas = np.random.exponential(1 / tasa_llegadas, n_eventos)
    tiempos_llegada = np.cumsum(tiempos_entre_llegadas)  # Acumular para tiempos de llegada

    # Generar tiempos de servicio (exponencial con tasa μ)
    tiempos_servicio = np.random.exponential(1 / tasa_servicios, n_eventos)

    return tiempos_llegada, tiempos_servicio


# Parámetros del sistema
lambda_llegadas = 5  # Tasa de llegadas (λ)
mu_servicios = 6  # Tasa de servicio (μ)
n_llegadas = 100  # Número de eventos a simular

# Generar tiempos de llegada y servicio
llegadas, servicios = generar_llegadas_y_servicios(lambda_llegadas, mu_servicios, n_llegadas)

# Mostrar los primeros 10 resultados
print("Tiempos de llegada (acumulados):", llegadas[:10])
print("Tiempos de servicio:", servicios[:10])
