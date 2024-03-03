import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import math
from numpy.ma import count

contenedor_de_la_matriz = []
contenedor_matrix_aa = []
contenedor_lista_matriz = []

contenedor_nodos = []
contenedor_x = []
xx = []
yy = []
contenedor_dxf = []
contenedor_df_concatenado = []
contenedor_valor = []
resultados = []
contendor_diccionario = []
contenedor_f = []
contenedor_sumaI = []
contenedor_rk = []
contenedor__sk = []
contenedor_skk = []
contenedor_rkk = []
fi = []
Ii = []
con_suma = []
contenedor_v = []

def calcular_similitud(base):
  
    contenedor_indice_similaridad = []
    contenedor_matriz = []
    matriz = np.zeros((len(base.columns), len(base.columns)))
    matriz_n = np.zeros((len(base.columns), len(base.columns)))
    for i in range(len(base.columns)):
        for j in range(i + 1, len(base.columns)):
            A = base.columns[i]
            B = base.columns[j]
            ai = base[A]
            aj = base[B]
            card = np.sum((ai * aj))
            n = len(base)
            n_ai =  np.sum((ai))
            n_aj = np.sum((aj))
            kc = (card - (n_ai * n_aj) / n) / np.sqrt((n_ai * n_aj) / n)
            sim = norm.cdf(kc)
            # CREACION DE LA MATRIZ SIMETRICA CON INDICES DE SIMILARIDAD
            contenedor_indice_similaridad.append((A, B, card, kc, sim))
            contenedor_lista_matriz.append((A, B, round(sim, 2)))

            matriz[i, j] = sim
            matriz[j, i] = sim

            segundo_maximo = np.amax(matriz)  # valor maximo valor de la matriz^2

    contenedor_matriz.append(matriz)
    column_labels = base.columns.tolist()  # encabezado automaticamente
    
    dfn = pd.DataFrame(matriz)
    dfn=(round(dfn,7))
    # Agregar encabezados a las columnas y filas de la matriz cero
    encabezados = column_labels
    dfn.columns = encabezados  # encabezados de la columna
    dfn.index = encabezados  # encabezados de la fila
    contenedor_de_la_matriz.append(dfn)  # contenedor de la matriz data frame

    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])

    resultado_text.delete(1.0, tk.END)  # Limpiar el cuadro de texto
    resultado_text.insert(tk.END, "VALORES DE COPRESENCIAS ESTANDARIZADAS E ÍNDICES DE SIMILARIDAD\n")
    resultado_text.insert(tk.END, str(data_sim) + "\n")

    
    resultado_text.insert(tk.END, str(dfn) + "\n")
    resultado_text.update_idletasks()

        
    
    
   
    # CREACION DEL DENDOGRAMA
    def dendograma_invertido(m):
        nueva_matriz1 = np.fill_diagonal(m, 1)
        # DENDOGRAMA INVERTIDO
        distancias = hierarchy.distance.pdist(matriz)
        enlaces = hierarchy.linkage(distancias, method='complete')
        dendrogram = hierarchy.dendrogram(enlaces, labels=column_labels, orientation='bottom')
        # Muestra los valores entre las parejas de variables
        for i, d, c in zip(dendrogram['icoord'], dendrogram['dcoord'], dendrogram['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y)
           
        # Configura los ejes y muestra el gráfico
        plt.title('Dendrograma')
        plt.xlabel('Índices de muestra')
        plt.ylabel('SIMILARIDAD')
        plt.savefig('Dendo.pdf')
        plt.show()
        data_sim.to_excel('values.xlsx', index=False)
        dfn.to_excel('similaridad.xlsx',index=False)

    dendograma_invertido(matriz)

def ordenar_nodos(datos):
    dfnuevo = pd.DataFrame(datos)
    v_1 = (dfnuevo[0] + "," + dfnuevo[1])
    v_2 = (dfnuevo[1] + "," + dfnuevo[0])
    a = np.array(v_1)
    b = np.array(v_2)
    c = np.array(dfnuevo[2])

    orden = np.argsort(c)[::-1]

    a_ordenada = a[orden]
    b_ordenada = b[orden]
    c_ordenada = c[orden]

    df_ordenados = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada, 'Valor': c_ordenada})
    df_ordenadoc = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada})
    df_ordenadov = pd.DataFrame({'Valor1': c_ordenada, 'Valor2': c_ordenada})

    df_combined1 = pd.DataFrame(df_ordenadoc.values.reshape(-1), columns=['Variables'])
    df_combined2 = pd.DataFrame(df_ordenadov.values.reshape(-1), columns=['Valor'])
    df_concatenado = pd.concat([df_combined1, df_combined2], axis=1)

    contenedor_valor.append(df_combined2)

    c_ordenado = df_concatenado['Valor'].values  # Obtener los valores de la columna 'Valor'

    c_orden = pd.Series(c_ordenado)

    grupos = c_orden.groupby(c_orden).groups
    num = 0
    for i, grupo in enumerate(grupos, 1):
        yy.append(grupo)
    xx.append(grupos)

    diccionario = xx[0]
    contenedor_valores = []
    contendor_diccionario.append(diccionario)

    dados = contendor_diccionario[0]
    w = 0
    for chave, valor in dados.items():
        w = w + 1
        lista = valor
        indice = pd.Index(valor)
        valores = indice.values.tolist()
        contenedor_valores.append(valores)

    variables = contenedor_valores

    def tabla_resolucion_nodos(variables, contenedor):
        v_alpha_k = 0
        s_k = len(contenedor)
        suma_total = 0
        mk = len(variables)
        for i in range(mk):
            contar = count(variables[i])
            resultados.append(contar)
            suma_total += contar
            f = (count(variables[i]) - 1)
            contenedor_sumaI.append(suma_total)
            contenedor_f.append(f)
        resultado_text.insert(tk.END, "matriz de nodos significativos\n")
        fila_invertida = contenedor_sumaI[::-1]
        columna_invertida = contenedor_f[::-1]
        s_k = (len((contenedor)) - 1)
        con_al = []
        for i in range(len(contenedor_f)):
            fi.append(columna_invertida[i])
            Ii.append(fila_invertida[i])
            rk = i + 1
            sk = s_k - i
            contenedor_rkk.append(rk)
            contenedor_skk.append(sk)
            card = (sum(Ii) - rk * ((rk + 1) / 2) - sum(fi))
            s_beta_k = round((card - (0.5 * sk * rk)) / math.sqrt((sk * rk * (sk + rk + 1)) / 12), 5)
            con_al.append(s_beta_k)
            valoresv = con_al
            resultados_v = []
            contenedor_vector = []
            for i in range(1, len(valoresv)):
                resta = valoresv[i] - valoresv[i - 1]
                contenedor_vector.append(resta)
            contenedor_vector.insert(0, valoresv[0])
            for j in range(len(contenedor_vector)):
                v = round(contenedor_vector[j], 3)
            contenedor_v.append(v)
            resultado_text.insert(tk.END, f"Card() [{i}] : {card}, S(Ω,k) [{i}]: {s_beta_k}, V(Ω,k)[{i}]: {v}\n")
        resultado_text.update_idletasks()

    tabla_resolucion_nodos(variables, contenedor_valor[0])

def grafica_omega(v):
    n = len(v)
    x = list(range(1, n + 1))
    y = v
    plt.plot(x, y, 'bo-', color='green', label='Valores de V(Ω,k)')
    for i in range(len(x)):
        plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom', color='black')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Índice de k')
    plt.ylabel('Valor de V(Ω,k)')
    plt.title('Gráfico de V(Ω,k)')
    plt.savefig('nodos.pdf')
    plt.show()

def seleccionar_archivo():
    archivo_excel = filedialog.askopenfilename(filetypes=[("Archivos Excel", "*.xlsx")])
    if archivo_excel:
        try:
            global base
            base = pd.read_excel(archivo_excel)
            # Limpia el cuadro de texto antes de cargar nuevos datos
            resultado_text.delete(1.0, tk.END)
            # Mostrar las variables en casillas de selección
            mostrar_variables()
        except Exception as e:
            resultado_text.delete(1.0, tk.END)
            resultado_text.insert(tk.END, f"Error al procesar: {str(e)}")

def mostrar_variables():
    if len(base.columns) > 30:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        frame = ttk.Frame(ventana_variables)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        var_selection = []

        for i, variable in enumerate(base.columns):
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(scrollable_frame, text=variable, variable=var_var)
            var_checkbutton.grid(row=i, column=0, sticky="w")
            var_selection.append((variable, var_var))

        # Botón para seleccionar todas las variables
        var_select_all_var = tk.BooleanVar()
        checkbutton_select_all = tk.Checkbutton(ventana_variables, text="Seleccionar Todas", variable=var_select_all_var, command=lambda: select_all_variables(var_selection, var_select_all_var))
        checkbutton_select_all.grid(row=i+1, column=0, sticky="w")

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.grid(row=i+2, column=0, pady=10)
    else:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        var_selection = []

        for variable in base.columns:
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(ventana_variables, text=variable, variable=var_var)
            var_checkbutton.pack()
            var_selection.append((variable, var_var))

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.pack()

def select_all_variables(var_selection, var_select_all_var):
    select_all_state = var_select_all_var.get()
    for _, var_var in var_selection:
        var_var.set(select_all_state)

def calcular_similaridad_desde_seleccion(var_selection):
    global variables_seleccionadas
    variables_seleccionadas = [variable for variable, var_var in var_selection if var_var.get()]

    if len(variables_seleccionadas) >= 2:
        # Calcular la similaridad solo con las variables seleccionadas
        base_seleccionada = base[variables_seleccionadas]
        calcular_similitud(base_seleccionada)
        ordenar_nodos(contenedor_lista_matriz)
        grafica_omega(contenedor_v)
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Seleccione al menos dos variables para calcular la similaridad.\n")

ventana = tk.Tk()
ventana.title("Calcular Similaridad y Nodos Significativos")

boton_seleccionar = tk.Button(ventana, text="Seleccionar Archivo Excel", command=seleccionar_archivo)
boton_seleccionar.pack(pady=10)

resultado_text = tk.Text(ventana, height=250, width=200)
resultado_text.pack(pady=100)

ventana.mainloop()
