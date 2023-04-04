import cv2
import dlib
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

# Inicializar el detector de rostros y el predictor de puntos de referencia
detector_rostro = dlib.get_frontal_face_detector()
predictor_puntos = dlib.shape_predictor("Proyecto Final IA\shape_predictor_68_face_landmarks.dat")

# Función que toma una foto y la guarda con el nombre y cargo ingresados
def tomar_foto():
    nombre = entry_nombre.get()
    cargo = entry_cargo.get()
    camara = cv2.VideoCapture(0)
    _, imagen_bgr = camara.read()
    ruta = "Proyecto Final IA\Personal\ "
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    rostros_detectados = detector_rostro(imagen_rgb)
    
    # Si no se detectan rostros, mostrar un mensaje de advertencia
    if len(rostros_detectados) == 0:
        messagebox.showwarning("No se detectaron rostros", "No se detectó ningún rostro en la imagen. Intente de nuevo.")
        return

    # Si se detecta más de un rostro, mostrar un mensaje de advertencia
    elif len(rostros_detectados) > 1:
        messagebox.showwarning("Demasiados rostros detectados", "Se detectaron varios rostros en la imagen. Intente de nuevo con una sola cara en la imagen.")
        return
    
    # Si se detecta un solo rostro, obtener los puntos de referencia del rostro
    puntos = predictor_puntos(imagen_rgb, rostros_detectados[0])
    puntos = np.array([(p.x, p.y) for p in puntos.parts()])
    
    # Previsualización de la imagen tomada
    img = Image.fromarray(imagen_rgb)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(image=img)
    label_imagen.config(image=img_tk)
    label_imagen.image = img_tk
    
    # Preguntar si desea guardar la imagen
    respuesta = messagebox.askyesno("Guardar Imagen", "¿Desea guardar la imagen?")
    if respuesta:
        cv2.imwrite(ruta + nombre + '_' + cargo + ".jpg", imagen_bgr)
        messagebox.showinfo("Imagen Guardada", "La imagen se ha guardado correctamente.")
        limpiar_campos()
    else:
        if os.path.exists(ruta + nombre + '_' + cargo + ".jpg"):
            os.remove(ruta + nombre + '_' + cargo + ".jpg")
            messagebox.showinfo("Imagen No Guardada", "La imagen no se ha guardado.")
        else:
            messagebox.showinfo("Imagen No Guardada", "La imagen no se ha guardado.")

        limpiar_foto()
    
    camara.release()

#limpiamos
def limpiar_campos():
    entry_nombre.delete(0, tk.END)
    entry_cargo.delete(0, tk.END)
    label_imagen.config(image=None)
    label_imagen.image = None
#limpiamos solo img
def limpiar_foto():
    
    label_imagen.config(image=None)
    label_imagen.image = None
# Creación de la ventana
ventana = tk.Tk()
ventana.title("Registro")
# Obtener las dimensiones de la pantalla
screen_width = ventana.winfo_screenwidth()
screen_height = ventana.winfo_screenheight()

# Calcular la posición de la ventana en el centro de la pantalla
x = int((screen_width/2) - (500/2))
y = int((screen_height/2) - (400/2))

# Configurar la geometría de la ventana
ventana.geometry(f"500x400+{x}+{y}")

# Labels para los campos de nombre y cargo
label_nombre = tk.Label(ventana, text="Nombre:")
label_nombre.pack()
entry_nombre = tk.Entry(ventana)
entry_nombre.pack()
label_cargo = tk.Label(ventana, text="Cargo:")
label_cargo.pack()
entry_cargo = tk.Entry(ventana)
entry_cargo.pack()

# Botón para tomar la foto
boton_foto = tk.Button(ventana, text="Tomar Foto" , command=tomar_foto)
boton_foto.pack(pady=10)

# Label para mostrar la imagen tomada
label_imagen = tk.Label(ventana)
label_imagen.pack()

ventana.mainloop()
