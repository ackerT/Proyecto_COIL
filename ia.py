import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import threading
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variables globales para el dataset, modelo y conteo de métricas
dataset = None
fila_actual = 0
modelo_ia = None
VP = 0  # Verdaderos Positivos
FP = 0  # Falsos Positivos
VN = 0  # Verdaderos Negativos
FN = 0  # Falsos Negativos

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Asistente de Corrección de Postura con IA")
ventana.geometry("800x700")

# Configuración inicial para gráficos
fig, ax = plt.subplots(figsize=(8, 5))
canvas = FigureCanvasTkAgg(fig, master=ventana)
canvas.get_tk_widget().pack(pady=20)

# Función para cargar el dataset
def cargar_dataset():
    global dataset, fila_actual, modelo_ia, VP, FP, VN, FN
    archivo_path = filedialog.askopenfilename(
        title="Seleccionar Dataset", 
        filetypes=[("CSV files", "*.csv")]
    )
    
    if archivo_path:
        try:
            dataset = pd.read_csv(archivo_path)
            fila_actual = 0  # Reiniciamos al cargar nuevo dataset
            VP = FP = VN = FN = 0  # Reiniciar métricas
            lbl_status.config(text="Dataset cargado", bg="lightblue")
            entrenar_modelo_ia()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el dataset: {e}")

# Función para entrenar el modelo de IA
def entrenar_modelo_ia():
    global modelo_ia, dataset
    if dataset is None:
        messagebox.showwarning("Advertencia", "Cargue un dataset primero.")
        return
    
    # Preparar los datos para entrenar el modelo
    features = dataset.drop(columns=['filename', 'class_name', 'class_no'])
    labels = dataset['class_name'].apply(lambda x: 1 if x == "no_pose" else 0)
    
    # Dividir datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Entrenar un clasificador RandomForest
    modelo_ia = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_ia.fit(X_train, y_train)
    
    # Evaluar precisión
    y_pred = modelo_ia.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    lbl_status.config(text=f"Modelo IA entrenado: {precision*100:.2f}% precisión", bg="lightgreen")

# Función para predecir postura usando el modelo IA
def predecir_postura(fila):
    global modelo_ia
    if modelo_ia is None:
        messagebox.showwarning("Advertencia", "Entrene el modelo de IA primero.")
        return None
    
    # Seleccionar las características de la fila actual
    caracteristicas = fila.drop(labels=['filename', 'class_name', 'class_no']).values.reshape(1, -1)
    
    # Hacer la predicción
    prediccion = modelo_ia.predict(caracteristicas)
    return prediccion[0]

# Función para verificar la postura basado en el dataset y el modelo IA
def verificar_postura():
    global fila_actual, VP, FP, VN, FN
    if dataset is None:
        messagebox.showwarning("Advertencia", "Cargue un dataset primero.")
        return
    
    while fila_actual < len(dataset):
        fila = dataset.iloc[fila_actual]
        postura_real = 1 if fila['class_name'] == 'no_pose' else 0
        postura_predicha = predecir_postura(fila)
        
        # Actualizar métricas y mostrar estado
        if postura_predicha == 1 and postura_real == 1:
            VP += 1
            lbl_resultado.config(text="Postura Incorrecta", bg="red")
        elif postura_predicha == 1 and postura_real == 0:
            FP += 1
            lbl_resultado.config(text="Postura Incorrecta (Falso Positivo)", bg="orange")
        elif postura_predicha == 0 and postura_real == 0:
            VN += 1
            lbl_resultado.config(text="Postura Correcta", bg="green")
        elif postura_predicha == 0 and postura_real == 1:
            FN += 1
            lbl_resultado.config(text="Postura Correcta (Falso Negativo)", bg="yellow")
        
        actualizar_status()
        fila_actual += 1
        actualizar_reporte()
        time.sleep(5)

# Actualizar el estado en la interfaz
def actualizar_status():
    lbl_status.config(text=f"VP: {VP} | FP: {FP} | VN: {VN} | FN: {FN}", bg="lightyellow")

# Función para iniciar el monitoreo en un hilo separado
def iniciar_monitoreo():
    if dataset is None:
        messagebox.showwarning("Advertencia", "Cargue un dataset primero.")
        return
    
    hilo_monitoreo = threading.Thread(target=verificar_postura, daemon=True)
    hilo_monitoreo.start()

# Función para actualizar el gráfico en tiempo real
def actualizar_reporte():
    ax.clear()
    categorias = ['Verdaderos Positivos', 'Falsos Positivos', 'Verdaderos Negativos', 'Falsos Negativos']
    valores = [VP, FP, VN, FN]
    colores = ['green', 'red', 'blue', 'orange']
    
    ax.bar(categorias, valores, color=colores)
    ax.set_xlabel('Tipo de Métrica')
    ax.set_ylabel('Cantidad')
    ax.set_title('Reporte de Predicciones del Modelo')
    ax.set_ylim(0, max(1, max(valores) + 5))
    
    canvas.draw()

# Etiqueta de estado de postura
lbl_status = tk.Label(ventana, text="Esperando...", font=("Arial", 16), width=50, height=2)
lbl_status.pack(pady=10)

# Etiqueta para mostrar resultados de postura
lbl_resultado = tk.Label(ventana, text="", font=("Arial", 18), width=50, height=2)
lbl_resultado.pack(pady=10)

# Botón para cargar el dataset
btn_cargar = tk.Button(ventana, text="Cargar Dataset", font=("Arial", 14), command=cargar_dataset)
btn_cargar.pack(pady=10)

# Botón para iniciar el monitoreo
btn_iniciar = tk.Button(ventana, text="Iniciar Monitoreo", font=("Arial", 14), command=iniciar_monitoreo)
btn_iniciar.pack(pady=10)

# Iniciar la ventana principal
ventana.mainloop()
