from gurobipy import Model, GRB
# Función para leer la instancia desde un archivo .txt
def leer_instancia(archivo):
    with open(archivo, 'r') as file:
        lines = file.readlines()
        ganancias = list(map(float, lines[0].split()))
        harina_requerida = list(map(float, lines[1].split()))
        horas_requeridas = list(map(float, lines[2].split()))
        max_unidades = list(map(float, lines[3].split()))
        recursos = list(map(float, lines[4].split()))
    return ganancias, harina_requerida, horas_requeridas, max_unidades, recursos

# Función para crear y resolver el modelo
def resolver_problema(archivo_instancia):
    print(f"\n\033[1;34mResolviendo instancia: {archivo_instancia}\033[0m")  # Título en azul
    # Leer datos desde la instancia
    ganancias, harina_requerida, horas_requeridas, max_unidades, recursos = leer_instancia(archivo_instancia)
    model = Model("Galletas")

    # Definir las variables
    gc = model.addVar(name="Galletas_Chocolate", lb=0)
    ga = model.addVar(name="Galletas_Avena", lb=0)
    gm = model.addVar(name="Galletas_Mantequilla", lb=0)
    gj = model.addVar(name="Galletas_Jengibre", lb=0)
    m = model.addVar(name="Macarons", lb=0)

    # Definir la función objetivo
    model.setObjective(ganancias[0]*gc + ganancias[1]*ga + ganancias[2]*gm + ganancias[3]*gj + ganancias[4]*m, GRB.MAXIMIZE)

    # Definir las restricciones
    model.addConstr(harina_requerida[0]*gc + harina_requerida[1]*ga + harina_requerida[2]*gm + harina_requerida[3]*gj + harina_requerida[4]*m <= recursos[0], name="Disponibilidad_Harina")
    model.addConstr(horas_requeridas[0]*gc + horas_requeridas[1]*ga + horas_requeridas[2]*gm + horas_requeridas[3]*gj + horas_requeridas[4]*m <= recursos[1], name="Capacidad_Horas_Producción")
    model.addConstr(gc <= max_unidades[0], name="Demanda_Máxima_Chocolate")
    model.addConstr(ga <= max_unidades[1], name="Demanda_Máxima_Avena")
    model.addConstr(gm <= max_unidades[2], name="Demanda_Máxima_Mantequilla")
    model.addConstr(gj <= max_unidades[3], name="Demanda_Máxima_Jengibre")
    model.addConstr(m <= max_unidades[4], name="Demanda_Máxima_Macarons")
    model.addConstr(gc + m >= 0.3*(gc + ga + gm + gj + m), name="Galletas_Premium")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Imprimir los costes reducidos de las variables
        print("\n\033[1;32mCostes reducidos de las variables:\033[0m")
        for var in model.getVars():
            print(f"\033[1;32m{var.VarName}: {var.RC:.4f}\033[0m")

        # Imprimir los valores sombra de las restricciones
        print("\n\033[1;32mValores sombra de las restricciones:\033[0m")
        for constr in model.getConstrs():
            print(f"\033[1;32m{constr.ConstrName}: {constr.Pi:.4f}\033[0m")

        # Imprimir las holguras de las restricciones
        print("\n\033[1;32mHolguras de las restricciones:\033[0m")
        for constr in model.getConstrs():
            print(f"\033[1;32m{constr.ConstrName}: {constr.Slack:.4f}\033[0m")

        print("\n\033[1;32mResultados para la instancia " + archivo_instancia + ":\033[0m")  # Verde para los resultados

        # Imprimir los valores optimizados de las variables
        print(f"\033[1;32mUnidades de galletas de chocolate: {gc.X:.2f}\033[0m")
        print(f"\033[1;32mUnidades de galletas de avena: {ga.X:.2f}\033[0m")
        print(f"\033[1;32mUnidades de galletas de mantequilla: {gm.X:.2f}\033[0m")
        print(f"\033[1;32mUnidades de galletas de jengibre: {gj.X:.2f}\033[0m")
        print(f"\033[1;32mUnidades de macarons: {m.X:.2f}\033[0m")
        print(f"\033[1;32mBeneficio Máximo: {model.ObjVal:.2f} euros\033[0m")
    else:
        print("\033[1;31mNo se encontró un resultado óptimo para la instancia\033[0m")

# Resolver ambas instancias
resolver_problema("instancia1.txt")
resolver_problema("instancia2.txt")
