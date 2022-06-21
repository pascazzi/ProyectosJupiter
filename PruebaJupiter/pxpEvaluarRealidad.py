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
from sklearn.metrics import confusion_matrix

##################################################################
# Parametrizo
mostrarGraficos = False
nombreArchivoPredicciones = "Diabetes_predicciones.csv"
nombreArchivoRealidad = "Diabetes_predicciones.csv" #"Diabetes_realidad.csv"
archivoPrefijoPlotReal = "Diabetes_graph_real_"

################# LEVANTO
df_expectativa = pd.read_csv(nombreArchivoPredicciones, sep=';')
df_realidad    = pd.read_csv(nombreArchivoPredicciones, sep=';')

################# PREPARO COLUMNAS
df_realidad.rename(columns={'PRONOSTICO':'objetivo'}, inplace=True)
df_expectativa.rename(columns={'PRONOSTICO':'objetivo'}, inplace=True)

################# PROCESO
df_expectativa_tx = pd.DataFrame(df_expectativa['objetivo'])
df_realidad_tx = pd.DataFrame(df_realidad['objetivo'])

################# COMPARO
confundida = confusion_matrix(df_realidad_tx, df_expectativa_tx.round())
confundidaPxp = pd_pxp.matrizConfusionPxp(confundida)

################# MUESTRO
print(confundida)
print(confundidaPxp.obtenerResultadoEvaluacion())
pd_pxp.mostrarMatriz(confundida, archivoPrefijoPlotReal, mostrarGraficos)
