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

# Modelo M/M/1

if __name__ == '__main__':

    lam = 5  # tasa de llegada
    mu = 7  # tasa de servicio
    resultado_mm1 = mm1_model(lam, mu)
    print("Resultados M/M/1:")
    for clave, valor in resultado_mm1.items():
        print(f"{clave}: {valor:.4f}")

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
