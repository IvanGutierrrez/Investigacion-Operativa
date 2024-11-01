from gurobipy import Model, GRB, quicksum

def leer_instancia(archivo):
    with open(archivo, 'r') as file:
        # Leer todas las líneas y quitar los saltos de línea
        lines = [line.strip() for line in file.readlines()]

        # La primera línea es n
        n = int(lines[0])  # Número de ciudades

        # Las siguientes n líneas son los valores de c
        c = dict() # Distancias entre ciudades (matriz c_{ij})
        for i in range(n):
            valores = list(map(int, lines[i + 1].split()))  # Obtener valores de la línea i+1
            k = 0
            for j in range(n):
                if i != j:  # Excluir los casos donde i == j
                    c[(i + 1, j + 1)] = valores[k]
                    k += 1

        # La siguiente línea con n valores es w
        w = list(map(int, lines[n + 1].split()))# Beneficio de visitar cada ciudad

        # La penúltima línea es W
        W = int(lines[n + 2])# Beneficio mínimo que se debe acumular

        # La última línea con n valores es p
        p = list(map(int, lines[n + 3].split())) # Penalización por no visitar una ciudad
    return n, c, w, W, p

# Función para crear y resolver el modelo
def resolver_problema(archivo_instancia):
    print(f"\n\033[1;34mResolviendo instancia: {archivo_instancia}\033[0m")  # Título en azul

    # Datos
    n, c, w, W, p = leer_instancia(archivo_instancia)
    ciudades = range(1, n + 1)

    # Crear el modelo
    modelo = Model("Prize-collecting Travel-Salesman-Problem")

    # Variables de decisión: x[i,j] = 1 si se viaja de ciudad i a ciudad j
    x = modelo.addVars(ciudades, ciudades, vtype=GRB.BINARY, name="x")
    # Variables auxiliares para eliminar subtours (MTZ)
    u = modelo.addVars(ciudades, vtype=GRB.CONTINUOUS, lb=1, ub=n, name="u")
    #Variable de decisión: y[i] = 1 si i esta incluido en el recorrido.
    y = modelo.addVars(ciudades, vtype=GRB.BINARY, name="y")

    # Establecer la función objetivo
    modelo.setObjective(
        #FALTA POR DEFINIR
    )

    # Restricción 10, asegura que si un nodo j es visitado
    # entonces debe haber al menos un nodo i desde el cual
    # se llegue a j a través del arco (i,j) en el recorrido.
    for j in ciudades:
        modelo.addConstr(
            quicksum(x[i, j] for i in ciudades) == y[j],
            name=f"entrada_{j}"
        )
    # Restricción 11, es complementaria a la anterior,
    # si un nodo i es visitado debe haber al menos un
    # nodo j al que se salga desde i a través del arco
    # (i,j) en el recorrido.
    for i in ciudades:
        modelo.addConstr(
            quicksum(x[i, j] for j in ciudades) == y[i],
            name=f"salida_{i}"
        )

    # Optimizar el modelo
    modelo.optimize()

    # Imprimir la solución
    if modelo.status == GRB.OPTIMAL:
        print(f"\nCoste mínimo del recorrido: {modelo.ObjVal}")
        ruta = []
        ciudad_actual = 1
        while True:
            ruta.append(ciudad_actual)
            siguiente_ciudad = None
            for j in ciudades:
                if j != ciudad_actual and x[ciudad_actual, j].X > 0.5:
                    siguiente_ciudad = j
                    break
            if siguiente_ciudad == 1:
                ruta.append(1)
                break
            else:
                ciudad_actual = siguiente_ciudad
        print("Ruta óptima:")
        print(" -> ".join(map(str, ruta)))
    else:
        print("No se encontró una solución óptima.")


print("Seleccione que instancia desea resolver")
print("1.- Instancia 1 (problema basico)")
print("2.- Instancia 2 (segundo problema)")
segundo = int(input())
if (segundo == 1):
    resolver_problema("instancia1.txt")
else:
    resolver_problema("instancia2.txt")
