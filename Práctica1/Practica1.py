from gurobipy import Model, GRB, quicksum
# Datos
n = 5 # Número de ciudades
ciudades = range(1, n + 1)
# Distancias entre ciudades (matriz c_{ij})
c = {
 (1, 2): 10, (1, 3): 8, (1, 4): 9, (1, 5): 7,
 (2, 1): 10, (2, 3): 10, (2, 4): 5, (2, 5): 6,
 (3, 1): 8, (3, 2): 10, (3, 4): 8, (3, 5): 9,
 (4, 1): 9, (4, 2): 5, (4, 3): 8, (4, 5): 6,
 (5, 1): 7, (5, 2): 6, (5, 3): 9, (5, 4): 6
}
# Beneficio de visitar cada ciudad
w = {
    1: 15, 2: 10, 3: 20, 4: 25, 5: 18
}
# Beneficio mínimo que se debe acumular
W = 50
# Penalización por no visitar una ciudad
p = {
    1: 5, 2: 7, 3: 6, 4: 8, 5: 4
}
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
