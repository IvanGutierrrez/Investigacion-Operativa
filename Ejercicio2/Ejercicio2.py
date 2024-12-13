import numpy as np


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
