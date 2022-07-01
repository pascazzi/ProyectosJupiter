#import os

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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import pandas as pd
import numpy
#import datetime
import scipy
#import shutil
import pxpLibreriaDataFrame as pd_pxp
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix

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


def PreProcesoDataFrame(logPrograma, df_leida):
    logPrograma += [pd_pxp.registroMomento("*) Inicio PreProcesoDataFrame "+__file__, True)]

    ###############################//// INICIO DE PREPROCESAMIENTO DE DATASET
    logPrograma += [pd_pxp.registroMomento("*) Transformando Columnas ", True)]

    # Elimino columnas absurdas (en este caso la creada por plsql)
    df_leida = df_leida.drop(df_leida.columns[[0]], axis='columns')

    # Borrado de filas que contienen un determinado valor
    # logPrograma += [pd_pxp.registroMomento("*) borrado de filas elegidas TIPO_ACME_ID=0 ", True)]
    # df_leida = df_leida.drop(df_leida[df_leida['TIPO_ACME_ID'] == 'I'].index)
    # df_leida = df_leida.drop(df_leida[df_leida['TIPO_ACME_ID'] == 'C'].index)

    # Eliminación de columnas que no utilizaremos
    df_leida = df_leida.drop(['ACME_ID'], axis=1)
    df_leida = df_leida.drop(['EDAD_PACIENTE_ID'], axis=1)
    df_leida = df_leida.drop(['RANGO_ANTIGUEDAD'], axis=1)
    df_leida = df_leida.drop(['HS_ID'], axis=1)
    df_leida = df_leida.drop(['RANGO_8HS_ID'], axis=1)
    df_leida = df_leida.drop(['CODIGO_POSTAL_ZONA_ID'], axis=1)
    df_leida = df_leida.drop(['INSTITUCION_ID'], axis=1)

    # Conversion de columnas a tipo texto
    df_leida['DIASEMANA_ID'] = df_leida['DIASEMANA_ID'].astype(str)
    # df_leida['ACME_ID'] = df_leida['ACME_ID'].astype(str)
    df_leida['SEXO_ID'] = df_leida['SEXO_ID'].astype(str)
    df_leida['AGENDA_ID'] = df_leida['AGENDA_ID'].astype(str)
    df_leida['SERVICIO_ID'] = df_leida['SERVICIO_ID'].astype(str)
    df_leida['GRUPO_SECTOR_ID'] = df_leida['GRUPO_SECTOR_ID'].astype(str)
    df_leida['SECTOR_ID'] = df_leida['SECTOR_ID'].astype(str)
    df_leida['RANGO_HORARIO_ID'] = df_leida['RANGO_HORARIO_ID'].astype(str)
    # df_leida['RANGO_ANTIGUEDAD'] = df_leida['RANGO_ANTIGUEDAD'].astype(str)
    df_leida['AREA_ID'] = df_leida['AREA_ID'].astype(str)
    df_leida['RANGO_4HS_ID'] = df_leida['RANGO_4HS_ID'].astype(str)
    # df_leida['INSTITUCION_ID'] = df_leida['INSTITUCION_ID'].astype(str)
    # df_leida['EDAD_DECADAS'] = df_leida['EDAD_DECADAS'].astype(str)
    # Normalización de columnas numericas
    # df_leida['EDAD_PACIENTE_ID'] = df_leida['EDAD_PACIENTE_ID'] / df_leida['EDAD_PACIENTE_ID'].max()
    df_leida['EDAD_DECADAS'] = df_leida['EDAD_DECADAS'] / 12 #df_leida['EDAD_DECADAS'].max()
    df_leida['ANTIGUEDAD_ID'] = df_leida['ANTIGUEDAD_ID'] / 365 #df_leida['ANTIGUEDAD_ID'].max()
    # df_leida['RANGO_ANTIGUEDAD'] = df_leida['RANGO_ANTIGUEDAD'] / 100 #df_leida['RANGO_ANTIGUEDAD'].max()

    # REEMPLAZO DE VALORES
    df_leida.loc[df_leida['ANTIGUEDAD_ID'] <0, 'ANTIGUEDAD_ID'] = 0
    ###############################//// FIN DE PREPROCESAMIENTO DE DATASET


    return df_leida
