import os

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
'''
print('backend :', keras.backend.backend())
print('keras version :', keras.__version__)
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import pandas as pd
import numpy
#import datetime
import scipy
#import shutil
import pxpLibreriaDataFrame as pd_pxp
import pxpLibreriaTURNOS as pxpLibreriaLocal
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#from sklearn.utils import class_weight
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
from contextlib import redirect_stdout
from pickle import dump
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

logPrograma = []
logPrograma += [pd_pxp.registroMomento("*) Iniciando programa normalizacion "+__file__, True)]

##################################################################
# Parametrizo
archivoOriginal = "Turnos_Tomados_Nov2020_mar2021_8.csv"
archivoBalanceado = 'Turnos_Tomados_Balanceados.csv'
archivoGuardaTransformaciones = "Turnos_Transformaciones.pkl"
archivoInfoBalanceo = "Turnos_Info_Balanceo.txt"

columnaObjetivo = 'AUSENTE'
mostrarGraficos = True
mostrarGraficoMejoresCols = False

metodoBalanceo = 'SMOTE-TOMEK'  # 'RUS' 'NEARMISS'  'ROS'  'SMOTE' 'SMOTE-TOMEK' 'NONE'

##antes
##################################################################
# Parametrizo columnas (['pclass', 'sex', 'embarked'], OneHotEncoder())
column_trans = ColumnTransformer(
    [
        ('diasem', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['DIASEMANA_ID'])
        #  , ('Acme', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['ACME_ID'])
        , ('Sexo', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['SEXO_ID'])
        , ('tipo_acme', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['TIPO_ACME_ID'])
        , ('ubicacion', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['UBICACION_ID'])
        , ('agenda', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['AGENDA_ID'])
        #  ('rangoant', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['RANGO_ANTIGUEDAD'])
        , ('servicio', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['SERVICIO_ID'])
        , ('sectorgroup', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['GRUPO_SECTOR_ID'])
        , ('sector', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['SECTOR_ID'])
        , ('rangohora', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['RANGO_HORARIO_ID'])
        , ('area', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['AREA_ID'])
        , ('rango4hs', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['RANGO_4HS_ID'])
        # , ('institu', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['INSTITUCION_ID'])
        #  , ('decadas', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['EDAD_DECADAS'])
    ],
    remainder='passthrough')
# DIASEMANA_ID	ACME_ID	SEXO_ID	aparicion	TIPO_ACME_ID	UBICACION_ID	EDAD_PACIENTE_ID	RANGO_ANTIGUEDAD

##despues

##################################################################
# Levanto los datos
###################################################################
logPrograma += [pd_pxp.registroMomento("*) Levantando datos de " + archivoOriginal, True)]
df_leida = pd.read_csv(archivoOriginal, sep=';')

if mostrarGraficoMejoresCols:
    classes = df_leida[columnaObjetivo].values
    unique, counts = numpy.unique(classes, return_counts=True)

    plt.bar(unique, counts)
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

if mostrarGraficoMejoresCols:
    pd_pxp.ExplorarDatosPrevios(df_leida)

# [infocol] Guardo informacion de columnas para registro en el log de texto
df_Informacion = pd.DataFrame()
df_Informacion["Columnas originales"] = df_leida.columns.tolist();

####################################################################
# Procesamiento básico del Dataframe
df_leida = pxpLibreriaLocal.PreProcesoDataFrame(logPrograma, df_leida)


# [infocol] Guardo informacion de columnas para registro en el log de texto
ColumnasResultantes = []
for i in df_Informacion.index:
    if df_Informacion["Columnas originales"][i] in df_leida.columns:
        ColumnasResultantes += [df_Informacion["Columnas originales"][i]]
    else:
        ColumnasResultantes += ['-----']
df_Informacion["Columnas Resultantes"] = ColumnasResultantes

################ TRANSFORMACION
coltra = column_trans.fit(df_leida)
transformacion = column_trans.transform(df_leida)
# La transformacion puede devolver distintas cosas... no uso pd.DataFrame.sparse.from_spmatrix por que funciona pero adentro queda sparce igual
if type(transformacion) == scipy.sparse.csr.csr_matrix:
    transformacion = transformacion.toarray()
df_transformada = pd.DataFrame(transformacion, columns=coltra.get_feature_names())

################# CONTROL DE NORMALIZACION
for unacol in df_transformada.columns:
    if (df_transformada[unacol].max() > 1) or (df_transformada[unacol].min() < 0):
        print('*** ATENCION ***\nColumna con valores no normalizados: ' + unacol)
        mensaje = 'Minimo {minimo}  Maximo {maximo} '
        mensaje = mensaje.format(minimo=df_transformada[unacol].min(), maximo=df_transformada[unacol].max())
        print(mensaje)
        input("Esperando ....")

logPrograma += [pd_pxp.registroMomento("*) Fin PreProcesoDataFrame ", True)]

############################ //// BALANCEO INICIO
logPrograma += [pd_pxp.registroMomento("*) Iniciando Balanceo  " + metodoBalanceo, True)]
df_in = pd.DataFrame(df_transformada).drop([columnaObjetivo], axis=1)  # Todos menos lo que voy a predecir
df_out = pd.DataFrame(df_transformada[columnaObjetivo])  # Lo que voy a predecir

distribuciones = pd_pxp.mostrarDistribucionDelObjetivo(df_out, columnaObjetivo, "ANTES DEL BALANCEO")
if metodoBalanceo == 'RUS':
    # ---------- Submuestreo RUS:  Elimina muestras de la clase más representada aleatoriamente
    rus = RandomUnderSampler()  # random_state = 0
    df_in, df_out = rus.fit_resample(df_in, df_out)
elif metodoBalanceo == 'NEARMISS':
    # ---------- Submuestreo NearMiss : Elimina las muestras más cercanas de la clase más representada
    nm = NearMiss()
    df_in, df_out = nm.fit_resample(df_in, df_out)
elif metodoBalanceo == 'ROS':
    # ---------- sobremuestreo: ROS. Duplica muestras de la clase menos representadas
    ros = RandomOverSampler()  # random_state = 0
    df_in, df_out = ros.fit_resample(df_in, df_out)
elif metodoBalanceo == 'SMOTE':
    # ---------- sobremuestreo: SMOTE. Genera nuevas muestras sintéticas
    smote = SMOTE()
    df_in, df_out = smote.fit_resample(df_in, df_out)
elif metodoBalanceo == 'SMOTE-TOMEK':
    # ---------- smote-Tomek. Sobremuestreo con Smote seguido de un submuestreo con Uniones de Tomek
    smoteT = SMOTETomek()  # random_state = 0
    df_in, df_out = smoteT.fit_resample(df_in, df_out)
else:
    print("balanceo no implementado o desconocido :" + metodoBalanceo)

df_transformada = df_in
df_transformada[columnaObjetivo] = df_out.values;
distribuciones += pd_pxp.mostrarDistribucionDelObjetivo(df_out, columnaObjetivo, "DESPUES DEL BALANCEO")
logPrograma += [pd_pxp.registroMomento("*) Fin del Balanceo  " + metodoBalanceo, True)]
############################# //// BALANCEO FINAL



###################################  GUARDO ARCHIVOS
logPrograma += [pd_pxp.registroMomento("*) GUARDANDO ARCHIVO BALANCEADO ", True)]
df_transformada.to_csv(archivoBalanceado, ';')
logPrograma += [pd_pxp.registroMomento("*) Guardando Transformaciones ", True)]
dump(coltra, open(archivoGuardaTransformaciones, "wb"))

logPrograma += [pd_pxp.registroMomento("*) Armando resultados finales ", True)]
fileMem = io.StringIO()
with redirect_stdout(fileMem):
    print('#' * 10, "Archivos en uso ", '#' * 10)
    print("* Archivos de entrada: ")
    print(' - ' + archivoOriginal)
    print("* Archivos de salida : ")
    print(' - ' + archivoBalanceado)
    print(' - ' + archivoInfoBalanceo)
    print(' - ' + archivoGuardaTransformaciones)
    print('#' * 10, "Balanceo", '#' * 10)
    print(" Metodo de Balanceo: "+metodoBalanceo)
    print('#' * 10, "Distribuciones", '#' * 10)
    print(distribuciones)
    print('#' * 10, "Columnas", '#' * 10)
    print(df_Informacion.head)
    print('#' * 10, "Detalle de transformaciones", '#' * 10)
    print(column_trans)
    print('#' * 10, "Estructura de  "+archivoBalanceado, '#' * 10)
    print(df_transformada.columns.tolist())
    print('#' * 10, "Log de performance", '#' * 10)
    logPrograma += [
        pd_pxp.registroMomento("*) FIN DEL PROCESO ", False)]  # no lo muestro por pantalla por que es stdout un string
    for mom in logPrograma:
        print(mom)

# print(fileMem.getvalue())
with open(archivoInfoBalanceo, 'w') as fileDisk:
    fileDisk.write(fileMem.getvalue())  # Lo abro con w entonces lo pisa

# Imprimo
import winsound

winsound.Beep(940, 500)
winsound.Beep(340, 500)

print("fin")