#---------------------------------Importación de Librerías --------->
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import joblib
import datetime
from imblearn.over_sampling import SMOTE
import streamlit_folium

#---------------------------------Agregar logo en la parte superior de la barra lateral --------->
logo_path = "cisa.png"
banner_path = "CiSaBanner.jpeg"

st.sidebar.image(logo_path, use_container_width=True)
st.image(banner_path, use_container_width=True)

#---------------------------------Function to load data --------->
@st.cache_data
def load_data(file_path):
    # Limpieza de datos
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['ANO', 'MES', 'DIA'])
    df['year'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['month'] = pd.to_numeric(df['MES'], errors='coerce')
    df['day'] = pd.to_numeric(df['DIA'], errors='coerce')
    df = df.dropna(subset=['ANO', 'MES', 'DIA'])
    df['ANO'] = df['ANO'].astype(int)
    df['MES'] = df['MES'].clip(1, 12).astype(int)
    df['DIA'] = df['DIA'].clip(1, 31).astype(int)
    
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['date']) 
    return df

# Cargar datos (Actualmente de manera local)
file_path = "delitosBucaramanga.csv"
df = load_data(file_path)

# Verificar si los datos se cargaron correctamente
if df is not None and not df.empty:
    #--------------------------------- Pestañas para el resumen de datos --------->
    st.header('Análisis Exploratorio de Datos')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top 5 Filas", "Ubicaciones de Delitos", "Delitos por Día de la Semana", "Principales Tipos de Delitos", "Delitos a lo Largo del Tiempo:"])

    with tab1:
        st.write("Top 5 filas del conjunto de datos:")
        st.write(df.head())

    with tab2:
        st.write("Top 5 Ubicaciones con Mayor Incidencia de Delitos:")
        top_locations = df['BARRIOS_HECHO'].value_counts().head(5).reset_index()
        top_locations.columns = ['Barrio', 'Conteo']
        fig = px.bar(top_locations, x='Barrio', y='Conteo', title='Top 5 Ubicaciones de Delitos')
        st.plotly_chart(fig, key='crime_locations_plot')

    with tab3:
        st.write("Delitos por Día de la Semana:")
        weekday_counts = df['DIA_SEMANA'].value_counts().reset_index()
        weekday_counts.columns = ['Día de la Semana', 'Conteo']
        fig = px.bar(weekday_counts, x='Día de la Semana', y='Conteo', title='Incidentes de Delitos por Día de la Semana')
        st.plotly_chart(fig, key='crime_weekday_plot')

    with tab4:
        st.write("Principales Tipos de Delitos:")
        top_crimes = df['CONDUCTA'].value_counts().head(5).reset_index()
        top_crimes.columns = ['Conducta', 'Conteo']
        fig = px.bar(top_crimes, x='Conducta', y='Conteo', title='Top 5 Tipos de Delitos')
        st.plotly_chart(fig, key='crime_types_plot')
    
    with tab5:
        st.write("Delitos a lo Largo del Tiempo:")
        crime_over_time = df['date'].value_counts().reset_index()
        crime_over_time.columns = ['Fecha', 'Cantidad de Delitos']
        crime_over_time = crime_over_time.sort_values('Fecha')
        fig = px.line(crime_over_time, x='Fecha', y='Cantidad de Delitos', title='Cantidad de Delitos a lo Largo del Tiempo')
        st.plotly_chart(fig, key='crime_over_time_plot')

    #---------------------------------Agregar Visualización Geoespacial --------->
    # Opciones de Visualización
    st.sidebar.header('Visualización Geoespacial')
    map_type = st.sidebar.selectbox('Seleccionar Tipo de Mapa', ['Dispersión', 'Mapa de Calor'], key='map_type')
    year_filter = st.sidebar.slider('Seleccionar Año', min_value=int(df['ANO'].min()), max_value=2021, value=(int(df['ANO'].min()), int(df['ANO'].max())), key='year_filter')
    month_filter = st.sidebar.slider('Seleccionar Mes', min_value=1, max_value=12, value=(1, 12), key='month_filter')

    # Filtrar datos según el año y mes seleccionados
    filtered_df = df[(df['ANO'] >= year_filter[0]) & (df['ANO'] <= year_filter[1]) & (df['MES'] >= month_filter[0]) & (df['MES'] <= month_filter[1])]

    if not filtered_df.empty and st.sidebar.button('Mostrar Mapa', key='show_map_button'):
        m = folium.Map(location=[filtered_df['LATITUD'].mean(), filtered_df['LONGITUD'].mean()], zoom_start=12)
        if map_type == 'Dispersión':
            for _, row in filtered_df.iterrows():
                folium.CircleMarker(location=[row['LATITUD'], row['LONGITUD']],
                                    radius=5,
                                    popup=row['CONDUCTA'],
                                    color='red',
                                    fill=True,
                                    fill_color='red').add_to(m)
        elif map_type == 'Mapa de Calor':
            heat_data = [[row['LATITUD'], row['LONGITUD']] for index, row in filtered_df.iterrows() if not pd.isna(row['LATITUD']) and not pd.isna(row['LONGITUD'])]
            HeatMap(heat_data).add_to(m)
        folium_static(m)

    #---------------------------------Opciones de Filtrado Avanzado --------->
    # Filtrado Avanzado
    st.sidebar.header('Filtrar Datos Avanzado')
    crime_type = st.sidebar.multiselect('Seleccionar Tipo de Delito', df['CONDUCTA'].unique(), default=None, key='crime_type')
    selected_barrio = st.sidebar.multiselect('Seleccionar Barrio', df['BARRIOS_HECHO'].unique(), default=None, key='selected_barrio')
    selected_ano = st.sidebar.multiselect('Seleccionar Año', df['ANO'].unique(), default=None, key='selected_ano')
    selected_mes = st.sidebar.multiselect('Seleccionar Mes', df['MES'].unique(), default=None, key='selected_mes')

    if crime_type or selected_barrio or selected_ano or selected_mes:
        filtered_df = filtered_df.copy()
        if crime_type:
            filtered_df = filtered_df[filtered_df['CONDUCTA'].isin(crime_type)]
        if selected_barrio:
            filtered_df = filtered_df[filtered_df['BARRIOS_HECHO'].isin(selected_barrio)]
        if selected_ano:
            filtered_df = filtered_df[filtered_df['ANO'].isin(selected_ano)]
        if selected_mes:
            filtered_df = filtered_df[filtered_df['MES'].isin(selected_mes)]
        st.write("Datos Filtrados:")
        st.dataframe(filtered_df)

        # Opcionalmente, graficar los datos filtrados en el mapa
        if not filtered_df.empty and st.sidebar.button('Mostrar Datos Filtrados en el Mapa', key='show_filtered_map_button'):
            m = folium.Map(location=[filtered_df['LATITUD'].mean(), filtered_df['LONGITUD'].mean()], zoom_start=12)
            for _, row in filtered_df.iterrows():
                folium.CircleMarker(location=[row['LATITUD'], row['LONGITUD']],
                                    radius=5,
                                    popup=row['CONDUCTA'],
                                    color='blue',
                                    fill=True,
                                    fill_color='blue').add_to(m)
            folium_static(m)

    #---------------------------------Modelo de Predicción --------->
    # Codificación de variables categóricas
    st.sidebar.header('Construir Modelo de Predicción')
    le_conducta = LabelEncoder()
    df['CONDUCTA_ENCODED'] = le_conducta.fit_transform(df['CONDUCTA'])

    X = df[['LATITUD', 'LONGITUD', 'ANO', 'MES', 'DIA']]
    y = df['CONDUCTA_ENCODED']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Balanceo de clases con SMOTE si es posible
    apply_smote = True
    for class_count in y_train.value_counts():
        if class_count < 6:  # Verificar si hay suficientes muestras para aplicar SMOTE
            apply_smote = False
            break

    if apply_smote:
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Entrenamiento modelo
    if st.sidebar.button('Entrenar Modelo', key='train_model_button'):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'crime_prediction_model.pkl')
        st.success('¡Modelo entrenado exitosamente con hiperparámetros optimizados!')

        # Mostrar el rendimiento del modelo
        y_pred = best_model.predict(X_test)
        st.write("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))
        st.write("Matriz de Confusión:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicción Baja', 'Predicción Alta'],
            y=['Real Baja', 'Real Alta'],
            colorscale='Blues'
        ))
        fig.update_layout(title='Matriz de Confusión', xaxis_title='Predicción', yaxis_title='Real')
        st.plotly_chart(fig, key='conf_matrix_plot')

    # Hacer predicciones con entrada del usuario
    st.sidebar.header('Hacer una Predicción')
    if 'input_lat' not in st.session_state:
        st.session_state.input_lat = float(df['LATITUD'].mean())
    if 'input_long' not in st.session_state:
        st.session_state.input_long = float(df['LONGITUD'].mean())
    if 'input_ano' not in st.session_state:
        st.session_state.input_ano = 2022  
    if 'input_mes' not in st.session_state:
        st.session_state.input_mes = 1
    if 'input_dia' not in st.session_state:
        st.session_state.input_dia = 1

    # Agregar mapa
    st.write("Haz clic en el mapa para seleccionar la Latitud y Longitud para la predicción:")
    def update_location(lat, lon):
        st.session_state.input_lat = lat
        st.session_state.input_long = lon

    if 'marker' not in st.session_state:
        st.session_state.marker = [float(df['LATITUD'].mean()), float(df['LONGITUD'].mean())]


    def update_location(lat, lon):
        st.session_state.marker = [lat, lon]

    m = folium.Map(location=st.session_state.marker, zoom_start=12)
    marker = folium.Marker(
        location=st.session_state.marker,
        draggable=True
    )
    marker.add_child(folium.Popup('Mover el marcador para actualizar la ubicación'))
    m.add_child(marker)

    folium_map = streamlit_folium.st_folium(m, key="map", width=700, height=400)

    # Obtener la ubicación del clic desde el mapa de streamlit_folium
    if folium_map['last_object_clicked'] is not None:
        last_clicked = folium_map['last_object_clicked']
        update_location(last_clicked['lat'], last_clicked['lng'])
        print(f"Nueva ubicación seleccionada: Latitud {last_clicked['lat']}, Longitud {last_clicked['lng']}")

    # Actualizar los valores de latitud y longitud con la ubicación seleccionada en el mapa
    input_lat = st.session_state.marker[0]
    input_long = st.session_state.marker[1]
    input_ano = st.sidebar.number_input('Año', min_value=2022, max_value=2030, value=st.session_state.input_ano, key='input_ano')
    input_mes = st.sidebar.number_input('Mes', min_value=1, max_value=12, value=st.session_state.input_mes, key='input_mes')
    input_dia = st.sidebar.number_input('Día', min_value=1, max_value=31, value=st.session_state.input_dia, key='input_dia')

    # Cargar modelo entrenado y hacer predicción
    if st.sidebar.button('Hacer Predicción', key='make_prediction_button'):
        best_model = joblib.load('crime_prediction_model.pkl')
        input_data = np.array([[input_lat, input_long, input_ano, input_mes, input_dia]])
        prediction = best_model.predict(input_data)
        predicted_conducta = le_conducta.inverse_transform(prediction)
        st.write(f'Tipo de Delito Predicho: {predicted_conducta[0]}')
        print(f'Tipo de Delito Predicho: {predicted_conducta[0]}')

    #---------------------------------Generar Datos Futuros de Delitos para Mapa de Calor --------->
    st.sidebar.header('Generar Datos de Delitos Futuros')
    start_date = st.sidebar.date_input("Fecha de inicio", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("Fecha de fin", datetime.date(2030, 12, 31))

    if st.sidebar.button('Generar Datos Futuros', key='generate_future_data_button'):
        best_model = joblib.load('crime_prediction_model.pkl')  
        future_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        future_data = []
        for date in future_dates:
            # Seleccionar aleatoriamente latitud, longitud y tipo de delito del historial de datos para la predicción futura
            sample = df.sample(1).iloc[0]
            lat, long = sample['LATITUD'], sample['LONGITUD']
            ano, mes, dia = date.year, date.month, np.random.randint(1, 29)
            input_data = np.array([[lat, long, ano, mes, dia]])
            predicted_crime = best_model.predict(input_data)[0]
            future_data.append([lat, long, ano, mes, dia, predicted_crime])
        future_df = pd.DataFrame(future_data, columns=['LATITUD', 'LONGITUD', 'ANO', 'MES', 'DIA', 'CONDUCTA_ENCODED'])
        future_df['CONDUCTA'] = le_conducta.inverse_transform(future_df['CONDUCTA_ENCODED'])

        # Mostrar mapa de calor de delitos futuros predichos
        m = folium.Map(location=[future_df['LATITUD'].mean(), future_df['LONGITUD'].mean()], zoom_start=12)
        heat_data = [[row['LATITUD'], row['LONGITUD']] for index, row in future_df.iterrows() if not pd.isna(row['LATITUD']) and not pd.isna(row['LONGITUD'])]
        HeatMap(heat_data).add_to(m)
        st.write("Mapa de Calor de Delitos Futuros Predichos:")
        folium_static(m)

else:
    st.error("Error al cargar los datos. Por favor, revise el archivo y asegúrese de que contiene las columnas 'ANO', 'MES' y 'DIA' válidas.")
