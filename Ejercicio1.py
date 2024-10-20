from gurobipy import Model, GRB

model = Model("Galletas")

gc = model.addVar(name="Galletas_Chocolate", lb=0)
ga = model.addVar(name="Galletas_Avena", lb=0)
gm = model.addVar(name="Galletas_Mantequilla", lb=0)
gj = model.addVar(name="Galletas_Jengibre", lb=0)
m = model.addVar(name="Macarons", lb=0)

model.setObjective(30*gc + 25*ga + 20*gm + 15*gj + 40*m, GRB.MAXIMIZE)

model.addConstr(3*gc + 2*ga + 4*gm + gj + 3*m <= 500, name="Disponibilidad_Harina")
model.addConstr(2*gc + ga + 0.6*gm + 0.5*gj + m <= 60, name="Capacidad_Horas_Producción")
model.addConstr(gc <= 30, name="Demanda_Máxima")
model.addConstr(ga <= 50, name="Demanda_Máxima")
model.addConstr(gm <= 40, name="Demanda_Máxima")
model.addConstr(gj <= 50, name="Demanda_Máxima")
model.addConstr(m <= 40, name="Demanda_Máxima")
model.addConstr(gc + m <= 0.3*(gc + ga + gm + gj + m), name="Galletas_Premium")

model.optimize()

if model.status == GRB.OPTIMAL:
    # Imprimir los costes reducidos de las variables
    print("\nCostes reducidos de las variables:")
    for var in model.getVars():
        print(f"{var.VarName}: {var.RC:.4f}")

    # Imprimir los valores sombra de las restricciones
    print("\nValores sombra de las restricciones:")
    for constr in model.getConstrs():
        print(f"{constr.ConstrName}: {constr.Pi:.4f}")

    # Imprimir las holguras de las restricciones
    print("\nHolguras de las restricciones:")
    for constr in model.getConstrs():
        print(f"{constr.ConstrName}: {constr.Slack:.4f}")

    print("\nValores resultantes:\n")

    print(f"Unidades de galletas de chocolate: {gc.X:.2f}")
    print(f"Unidades de galletas de avena: {ga.X:.2f}")
    print(f"Unidades de galletas de mantequilla: {gm.X:.2f}")
    print(f"Unidades de galletas de jengibre: {gj.X:.2f}")
    print(f"Unidades de macarong: {m.X:.2f}")
    print(f"Beneficio Maximo {model.ObjVal:.2f} euros")
else:
    print("No encontrado resultado optimo")
