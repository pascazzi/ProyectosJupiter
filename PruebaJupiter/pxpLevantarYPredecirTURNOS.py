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
import pxpLibreriaTURNOS as pxpLibreriaLocal
from keras.models import model_from_json
from pickle import load

##################################################################
# Parametrizo
archivoGuardaModelo = "Turnos_Modelo.json"
archivoGuardaPesos = "Turnos_Peso.h5"
archivoGuardaTransformaciones = "Turnos_Transformaciones.pkl"

archivoAPredecir = "Turnos_ParaPredecir.csv"
ArchivoPredicciones = "Turnos_predicciones.csv"
ArchivoPrediccionesTransformadas = "Turnos_prediccionesTransformadas.csv"
GenerarArchivoPrediccionesTransformadas = False

columnaObjetivo = 'AUSENTE'

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

## Genero el Dataframe de salida identico a la entrada (luego agregare la prediccion detras
prediccion = df_leida.drop([columnaObjetivo], axis=1)
# Elimino columnas absurdas (en este caso la creada por plsql)
prediccion = prediccion.drop(prediccion.columns[[0]], axis='columns')

df_leida = pxpLibreriaLocal.PreProcesoDataFrame(logPrograma, df_leida)

################ PREPARO COLUMNAS
logPrograma += [pd_pxp.registroMomento("*) Preparando Columnas ", True)]
df_leida[columnaObjetivo] = 0  # la necesito pra la transformacion, luego la borro

################# TRANSFORMO COLUMNAS
logPrograma += [pd_pxp.registroMomento("*) Transformando Columnas ", True)]
coltra = load(open(archivoGuardaTransformaciones, "rb"))
transformacion = coltra.transform(df_leida)
# 2604 df_transformada = pd.DataFrame(transformacionLeida, columns=coltra.get_feature_names())
# La transformacion puede devolver distintas cosas... no uso pd.DataFrame.sparse.from_spmatrix por que funciona pero adentro queda sparce igual
if type(transformacion) == scipy.sparse.csr.csr_matrix:
    transformacion = transformacion.toarray()
df_transformada = pd.DataFrame(transformacion, columns=coltra.get_feature_names())

prediccion_transformada = df_transformada.drop([columnaObjetivo], axis=1)
# prediccion = df_leida.drop([columnaObjetivo], axis=1)


####################################### PREDICCIONES
logPrograma += [pd_pxp.registroMomento("*) Iniciando prediccion ", True)]
pp = loaded_model.predict(prediccion_transformada)
logPrograma += [pd_pxp.registroMomento("*) Iniciando vuelco de predicciones ", True)]
prediccion_transformada["PROBABILIDAD"] = pp
prediccion_transformada["PRONOSTICO"] = prediccion_transformada["PROBABILIDAD"].apply(round)
prediccion["PROBABILIDAD"] = pp
prediccion["PRONOSTICO"] = prediccion_transformada["PROBABILIDAD"].apply(round)

####################################### GUARDO RESULTADOS
# print('#' * 10, 'RESULTADOS PREDECIDOS', '#' * 10)
# print(prediccion.sample(7))
logPrograma += [pd_pxp.registroMomento("*) Guardando datos en " + ArchivoPredicciones, True)]
prediccion.to_csv(ArchivoPredicciones, ';')
if GenerarArchivoPrediccionesTransformadas:
    logPrograma += [pd_pxp.registroMomento("*) Guardando datos en " + ArchivoPrediccionesTransformadas, True)]
    prediccion_transformada.to_csv(ArchivoPrediccionesTransformadas, ';')

logPrograma += [pd_pxp.registroMomento("*) FIN DEL PROCESO ", True)]
for mom in logPrograma:
    print(mom)

####################################### SALIDA FINAL
print("PREDICCIONES FINALIZADAS!")
print("Archivos guardados: ")
print("       - " + ArchivoPredicciones)
if GenerarArchivoPrediccionesTransformadas:
    print("       - " + ArchivoPrediccionesTransformadas)
print("Valores predecidos: ")
print(prediccion['PRONOSTICO'].value_counts().to_string())
