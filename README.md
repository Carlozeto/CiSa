# CiSa
Proyecto CiSa Convocatoria Datos a la U - Modulo Anilisis de Datos y Prediccion de Delitos - Bucaramanga DB

Para la correcta instalacion del software CiSa es necesario instalar python y el codigo de las librerias que el codigo utiliza para esto podemos usar nuestro equipo u otras aplicaciones como google colabs

# Instalación en equipo (Recomendada)
## Instalación de python
Para instalar python es necesario dirigirnos a la pagina oficial de python e instalar la version 3.12.7 o superior esta se encuentra disponible en el siguiente link
https://www.python.org/downloads/release/python-3127/

## Clonación de repositorio
Una vez instalado python, clonamos este repositorio con el comando
git clone https://github.com/Carlozeto/CiSa

## Instalación de librerias
Ya clonado el repositorio nos dirigimos a la carpeta en la que lo instalamos y ejecutamos los siguientes comandos para instalar las librerias que este utiliza
```shell
pip install streamlit
pip install pandas
pip install numpy
pip install plotly
pip install folium
pip install streamlit_folium
pip install pydeck
pip install scikit-learn
pip install joblib
pip install imblearn
```
## Ejecución código
Una vez instaladas las librerias podemos ejecutar el proyecto con el comando
```shell
streamlit run app.py
```
Esto deberia abrirte el sitio web desarrollado, si este no es el caso abre la página
http://localhost:8501/

## Generación de modelos de predicción
Dado el tamaño de los modelos generados estos no pudieron ser agregados al repositorio, sin embargo, estos se pueden generar mediante los botones de la interfaz "Entrenar modelo" y "Generar datos futuros"


# Google Colabs
Para la correcta ejecucion de este codigo en google Colabs es necesario instalar las librerias, para esto podemos hacer una linea de código con la instalación de las mismas: 
```shell
!pip install streamlit
!pip install pandas
!pip install numpy
!pip install plotly
!pip install folium
!pip install streamlit_folium
!pip install pydeck
!pip install scikit-learn
!pip install joblib
!pip install imblearn
```

Agregamos la siguiente linea de código al final y Ejecutamos el programa
```shell
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
Esto deberia abrirte el sitio web desarrollado, si este no es el caso abre la página
http://localhost:8501/

## Generación de modelos de predicción
Dado el tamaño de los modelos generados estos no pudieron ser agregados al repositorio, sin embargo, estos se pueden generar mediante los botones de la interfaz "Entrenar modelo" y "Generar datos futuros"
