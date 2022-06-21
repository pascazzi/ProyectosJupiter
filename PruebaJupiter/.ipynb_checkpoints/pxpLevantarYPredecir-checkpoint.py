import os
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import scipy
import pxpLibreriaDataFrame as pd_pxp
from keras.models import model_from_json
from pickle import load

##################################################################
# Parametrizo
archivoGuardaModelo = "Diabetes_Modelo.json"
archivoGuardaPesos = "Diabetes_Peso.h5"
archivoGuardaTransformaciones = "Diabetes_Transformaciones.pkl"
archivoAPredecir = "DiabetesColor_ParaPredecir.csv"
nombreArchivoPredicciones = "Diabetes_predicciones.csv"
nombreArchivoPrediccionesTransformadas = "Diabetes_prediccionesTransformadas.csv"
columnaObjetivo = 'aparicion'

logPrograma = [pd_pxp.registroMomento("*) INICIO DEL PROCESO ", True)]
##################################################
# Cargo modelo y pesos
logPrograma += [pd_pxp.registroMomento("*) Carga de modelos y pesos ", True)]
json_file = open(archivoGuardaModelo, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights(archivoGuardaPesos)
print("Cargado " + archivoGuardaModelo + " y " + archivoGuardaPesos)


################# COMPILANDO modelo
logPrograma += [pd_pxp.registroMomento("*) Compilando modelo ", True)]
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

################# LEVANTO datos a predecir
logPrograma += [pd_pxp.registroMomento("*) Levantando datos a predecir de " + archivoAPredecir, True)]
df_leida = pd.read_csv(archivoAPredecir, sep=';')

################# PREPARO COLUMNAS
logPrograma += [pd_pxp.registroMomento("*) Preparando Columnas ", True)]
df_leida.rename(columns={columnaObjetivo:'objetivo'}, inplace=True)
df_leida['objetivo'] = 0  # la necesito pra la transformacion, luego la borro

################# TRANSFORMO COLUMNAS
logPrograma += [pd_pxp.registroMomento("*) Transformando Columnas ", True)]
coltraLeida = load(open(archivoGuardaTransformaciones, "rb"))
transformacionLeida = coltraLeida.transform(df_leida)
df_transformada = pd.DataFrame(transformacionLeida, columns=coltraLeida.get_feature_names())
prediccion_transformada = df_transformada.drop(['objetivo'], axis=1)
prediccion = df_leida.drop(['objetivo'], axis=1)

####################################### PREDICCIONES
logPrograma += [pd_pxp.registroMomento("*) Iniciando prediccion ", True)]
pp = loaded_model.predict(prediccion_transformada)
logPrograma += [pd_pxp.registroMomento("*) Iniciando vuelco de predicciones ", True)]
prediccion_transformada["PROBABILIDAD"] = pp
prediccion_transformada["PRONOSTICO"] = prediccion_transformada["PROBABILIDAD"].apply(round)
prediccion["PROBABILIDAD"] = pp
prediccion["PRONOSTICO"] = prediccion_transformada["PROBABILIDAD"].apply(round)

####################################### GUARDO RESULTADOS
logPrograma += [pd_pxp.registroMomento("*) Guardando datos en " + nombreArchivoPredicciones+" y en "+nombreArchivoPrediccionesTransformadas, True)]
print('#' * 10, 'RESULTADOS PREDECIDOS', '#' * 10)
print(prediccion.sample(7))
prediccion.to_csv(nombreArchivoPredicciones, ';')
prediccion_transformada.to_csv(nombreArchivoPrediccionesTransformadas, ';')

logPrograma += [pd_pxp.registroMomento("*) FIN DEL PROCESO ", True)]
for mom in logPrograma:
        print(mom)

####################################### SALIDA FINAL
print("PREDICCIONES FINALIZADAS!")
print("Archivos guardados: " + nombreArchivoPredicciones + " - " + nombreArchivoPrediccionesTransformadas)
print("Valores predecidos: ")
print(prediccion['PRONOSTICO'].value_counts().to_string() )