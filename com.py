
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import math

def cargar_archivo(entrada):
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos Excel", "*.xlsx;*.xls")])
    entrada.delete(0, tk.END)
    entrada.insert(0, ruta_archivo)
    print(f"Archivo cargado: {ruta_archivo}")

def convertir_a_numero(valor):
    try:
        return float(valor)
    except ValueError:
        return valor

def comparar_archivos(archivo1, archivo2, resultado_text):
    try:
        df1 = pd.read_excel(archivo1, header=None)
        df2 = pd.read_excel(archivo2, header=None)

        num_elementos_iguales = 0
        tolerancia = 0.0001  # Puedes ajustar este valor según tu necesidad

        for i in range(len(df1.index)):
            for j in range(i+1, len(df1.columns)):
                valor1 = convertir_a_numero(df1.iloc[i, j])
                valor2 = convertir_a_numero(df2.iloc[i, j])

                if isinstance(valor1, (int, float)) and isinstance(valor2, (int, float)):
                    if valor1 == valor2 or math.isclose(valor1, valor2, rel_tol=tolerancia):
                        num_elementos_iguales += 1

        resultado_text.config(state=tk.NORMAL)
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, f"El número de elementos iguales por encima de la diagonal superior es: {num_elementos_iguales}")
        resultado_text.config(state=tk.DISABLED)

    except Exception as e:
        resultado_text.config(state=tk.NORMAL)
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, f"Error: {str(e)}")
        resultado_text.config(state=tk.DISABLED)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Comparador de Archivos Excel")

# Crear widgets
label_archivo1 = tk.Label(ventana, text="Archivo 1:")
entrada_archivo1 = tk.Entry(ventana, width=40)
boton_cargar1 = tk.Button(ventana, text="Cargar", command=lambda: cargar_archivo(entrada_archivo1))

label_archivo2 = tk.Label(ventana, text="Archivo 2:")
entrada_archivo2 = tk.Entry(ventana, width=40)
boton_cargar2 = tk.Button(ventana, text="Cargar", command=lambda: cargar_archivo(entrada_archivo2))

boton_comparar = tk.Button(ventana, text="Comparar", command=lambda: comparar_archivos(entrada_archivo1.get(), entrada_archivo2.get(), resultado_text))

resultado_label = tk.Label(ventana, text="Resultado:")
resultado_text = tk.Text(ventana, height=4, width=40, wrap=tk.WORD)
resultado_text.config(state=tk.DISABLED)

# Ubicar widgets en la ventana
label_archivo1.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
entrada_archivo1.grid(row=0, column=1, padx=5, pady=5)
boton_cargar1.grid(row=0, column=2, padx=5, pady=5)

label_archivo2.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
entrada_archivo2.grid(row=1, column=1, padx=5, pady=5)
boton_cargar2.grid(row=1, column=2, padx=5, pady=5)

boton_comparar.grid(row=2, column=0, columnspan=3, pady=10)

resultado_label.grid(row=3, column=0, columnspan=3, pady=5)
resultado_text.grid(row=4, column=0, columnspan=3, pady=5)

# Iniciar el bucle de la ventana
ventana.mainloop()
