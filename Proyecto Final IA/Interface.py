import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess

def abrir_registro():
    subprocess.Popen(['python', 'Proyecto Final IA\Registra.py'])
    
def abrir_reconocimiento():
    subprocess.Popen(['python', 'Proyecto Final IA\Reconocimiento.py'])

# Función que carga y repite el video de fondo
def load_video(filename):
    fondo = cv2.VideoCapture(filename)
    while True:
        ret, frame = fondo.read()
        if not ret:
            fondo = cv2.VideoCapture(filename)
            ret, frame = fondo.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

# Crear la ventana
root = tk.Tk()
# Obtener las dimensiones de la pantalla
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calcular la posición de la ventana en el centro de la pantalla
x = int((screen_width/2) - (800/2))
y = int((screen_height/2) - (600/2))

# Configurar la geometría de la ventana
root.geometry(f"800x600+{x}+{y}")

root.title("Reconocimento facial ")
root.configure(bg="black")

# Agregar el video de fondo
video_canvas = tk.Canvas(root, width=800, height=600, highlightthickness=0)
video_canvas.pack()
video_frames = load_video("Proyecto Final IA\Fondo\Video2.mp4")
current_frame = next(video_frames)
video_image = ImageTk.PhotoImage(Image.fromarray(current_frame))
video_canvas.create_image(0, 0, anchor=tk.NW, image=video_image)

# Función que actualiza el video en el fondo
def update_video():
    global current_frame, video_image
    current_frame = next(video_frames)
    video_image = ImageTk.PhotoImage(Image.fromarray(current_frame))
    video_canvas.create_image(0, 0, anchor=tk.NW, image=video_image)
    
    root.after(10, update_video)

# Configurar estilo para el botón black
style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", background="black", foreground="white", borderwidth=0)

# Configurar estilo para el botón cuando se pone el puntero sobre él
style.map("TButton", foreground=[("active", "white")], background=[("active", "blue")])

# Agregar los botones con el nuevo estilo y tamaño ajustado
button1 = ttk.Button(root, text="Iniciar", style="TButton", command=abrir_reconocimiento, padding=(10, 10, 10, 10))
button1.place(relx=0.4, rely=0.9, anchor=tk.CENTER)

button2 = ttk.Button(root, text="Registrar", style="TButton", command=abrir_registro, padding=(10, 10, 10, 10))
button2.place(relx=0.6, rely=0.9, anchor=tk.CENTER)

# Iniciar la actualización del video en el fondo
root.after(0, update_video)

# Ejecutar la aplicación
root.mainloop()
