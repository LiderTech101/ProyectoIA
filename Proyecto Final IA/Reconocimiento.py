#importamos las librerias
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

#accedemos a la carpeta 
path ='Personal'
imagenes = []
clases = []
listas = os.listdir(path)
#print (listas)
#variables
comp1=100
#leemos los rostros
for lis in listas:
    # leemos las imagenes
    imgdb=cv2.imread(f'{path}/{lis}')
    #almacenamos la imagen
    imagenes.append(imgdb)
    #almacenamos el nombre
    clases.append(os.path.splitext(lis)[0])
print(clases)

#funcion para codificar los rostros
def codrostros(imagenes):
    listacod=[]
    
    for img in imagenes:
        #corregmos color
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #codificamos la imageb
        cod=fr.face_encodings(img)[0]
        
        #almacenamos
        listacod.append(cod)
        
    return listacod
# hora de ingreso
#hora de ingreso
def horario(nombre):
    #abrimos eñ arcivo en modo adición
    with open ('Proyecto Final IA\horario.csv','a') as h:
        #extraemos info actual
        info=datetime.now()
        #extraemos fecha
        fecha=info.strftime('%Y:%M:%D:')
        #extraemos hora
        hora=info.strftime('%H:%M:%S')
        #guardamos info
        h.writelines(f'\n{nombre},{fecha},{hora}')
        print(hora)


#leemos la funcion
rostrocod=codrostros(imagenes)
#rostro_detectado = False

#Tamaño de ventana
#creamos la ventana de visualización con tamaño de 800x600
cv2.namedWindow("Reconocimiento Facial Con OpenCV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconocimiento Facial Con OpenCV", 800, 600)

#realizamos la captura del video
cap=cv2.VideoCapture(0)

#inicio
while True:
    #leemos los fotogramas
    ret, frame=cap.read()
    #reducimos las imagenes para mejorar el procesado
    frame2=cv2.resize(frame,(0,0),None,0.25,0.25)
    #realizamos una convercion de color
    rgb=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    #buscamos los rostros
   
    faces=fr.face_locations(rgb)
    facescod=fr.face_encodings(rgb,faces)
    facescod = np.array(facescod)
    #integramos
    for facecod,faceloc in zip(facescod,faces):
        #comparamos rostros
        comparacion=fr.compare_faces(rostrocod,facescod)
        #calculamos la similitud
        simi=fr.face_distance(np.array(rostrocod), np.array(facescod))
        #imprimimos simi
        #buscamos el alor mas bajo
        min=np.argmin(simi)
        if comparacion[min]:
            nombre=clases[min].upper()
            #extraemos cordenadas
            yi,xf,yf,xi=faceloc
            #escalamos
            yi,xf,yf,xi=yi*4,xf*4,yf*4,xi*4
            indice=comparacion.index(True)
            #comparamos
            if comp1 !=indice:
                #para dibujar lo que hacemos es cambiar los colores
                r=random.randrange(0,225,50)
                g=random.randrange(0,225,50)
                b=random.randrange(0,225,50)
                comp1=indice
                
            if comp1==indice:
                #dibujamos
                cv2.rectangle(frame,(xi,yi),(xf,yf),(r,g,b),3)
                cv2.rectangle(frame,(xi,yf-35),(xf,yf),(r,g,b),cv2.FILLED)
                cv2.putText(frame,nombre,(xi+6,yf-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                horario(nombre)
               # if not rostro_detectado:
                  #  print(nombre)
                   # rostro_detectado = True
                   # horario(nombre)
        #else:
           # rostro_detectado = False
    #mostramos frames
    cv2.imshow("Reconocimiento Facial Con OpenCV",frame)
    #leemos el teclado
    t=cv2.waitKey(5)
    if t==27:
        break

cv2.destroyAllWindows()
cap.release()
                
                

    

