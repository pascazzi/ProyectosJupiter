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
import scipy
import shutil
import datetime
import pxpLibreriaDataFrame as pd_pxp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from contextlib import redirect_stdout
from pickle import dump

logPrograma = []
logPrograma += [pd_pxp.registroMomento("*) Iniciando programa de entrenamiento", True)]

##################################################################
# Parametrizo
archivoGuardaModelo = "Diabetes_Modelo.json"
archivoGuardaPesos = "Diabetes_Peso.h5"
archivoGuardaTransformaciones = "Diabetes_Transformaciones.pkl"
archivoInfoEntrenamiento = "Diabetes_InfoEntrenamiento.txt"
archivoDeEntrenamiento = "DiabetesColor_ParaEntrenamientoCompleto.csv"  # "DiabetesColor_ParaEntrenamiento.csv"
archivoPrefijoPlot = "Diabetes_graph_"
columnaObjetivo='aparicion'
mostrarGraficos = False
mostrarMejoresCols = False
mostrarGraficoMejoresCols = False
kantidadColumnasBuscadas = 1
split_dataframe_para_test = 0.20
epocas = 20
lote = 5
split_validacion_fit_size = 0.20
##################################################################
# Parametrizo columnas
column_trans = ColumnTransformer(
    [('Colo', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['COLOR'])],
    remainder='passthrough')

##################################################################
# Levanto los datos
logPrograma += [pd_pxp.registroMomento("*) Levantando datos de " + archivoDeEntrenamiento, True)]
df_leida = pd.read_csv(archivoDeEntrenamiento, sep=';')

# df = pd.read_csv("pima-indians-diabetesHEADANEXO.csv", sep=';')
if mostrarGraficoMejoresCols:
    pd_pxp.ExplorarDatosPrevios(df_leida)

##################################################################
logPrograma += [pd_pxp.registroMomento("*) Transformando Columnas ", True)]
df_leida.rename(columns={columnaObjetivo:'objetivo'}, inplace=True)

coltra = column_trans.fit(df_leida)
transformacion = column_trans.transform(df_leida)
# La transformacion puede devolver distintas cosas... no uso pd.DataFrame.sparse.from_spmatrix por que funciona pero adentro queda sparce igual
if type(transformacion) == scipy.sparse.csr.csr_matrix:
    transformacion=transformacion.toarray()
df_transformada = pd.DataFrame(transformacion, columns=coltra.get_feature_names())

# # # salta columnas # # #n
# df_transformada=df_leida.drop(["COLOR"], axis=1)
##############################################################
logPrograma += [pd_pxp.registroMomento("*) Iniciando Split de dataframes ", True)]
train, test = train_test_split(df_transformada, test_size=split_dataframe_para_test, random_state=42, shuffle=True)

# PREPARO ENTRENAMIENTO
train_entrada = pd.DataFrame(train).drop(["objetivo"], axis=1)  # Todos menos lo que voy a predecir
train_salida = pd.DataFrame(train['objetivo'])  # Lo que voy a predecir

# PREPARO TEST
test_entrada = pd.DataFrame(test).drop(["objetivo"], axis=1)  # Todos menos lo que voy a predecir
test_salida = pd.DataFrame(test['objetivo'])  # Lo que voy a predecir

# Mostrar Mejores Columnas
if mostrarMejoresCols:
    pd_pxp.MostrarMejoresColumnas(df_leida, train_entrada, train_salida, kantidadColumnasBuscadas)
    cero = input("\nPresionar 0 para terminar")
    if cero == '0':
        print("Chau!")
        quit()

# Verifico Variabilidad
analisisVariabilidad = ''
sets = (('train_salida', train_salida), ('test_salida', test_salida))
for set_name, set_data in sets:
    analisisVariabilidad = analisisVariabilidad + '-' * 10 + ' ' + set_name + ' ' + '-' * 10 + "\n"
    analisisVariabilidad = analisisVariabilidad + set_data['objetivo'].value_counts().to_string() + "\n"
print(analisisVariabilidad)

# Entrenar y Fit
numpy.random.seed(7)
q_dim_entradas = train_entrada.columns.size

# crea el modelo
model = Sequential()
model.add(Dense(12, input_dim=q_dim_entradas, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# sigmoid  capa de salida, 1 neurona, Espero 1 salida: cero o uno. Recibe todas las salidas de la capa anterior
# por eso es densa. Pero debe devolver un numero entre cero y uno.

logPrograma += [pd_pxp.registroMomento("*) Compilando Modelo ", True)]
# Compila el modelo: Internamente armo el grafo.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# adam es la mas completa de las funciones de ajustar pesos (adam: variante de pesos por gradiente)
# funcion de perdida: binary_crossentropy ideal para salidas entre cero y uno.
# metrics: lo que va imprimiendo de salida.

logPrograma += [pd_pxp.registroMomento("*) Creando Modelo ", True)]
# Ajusta el modelo
history = model.fit(x=train_entrada, y=train_salida, epochs=epocas, validation_split=split_validacion_fit_size,
                    batch_size=lote,
                    shuffle=True)  # 72.64 #72.80

# batch_size - cada cuanto reajusto los pesos?
# epochs - cuantas veces recorro los datos ajustando pesos?

logPrograma += [pd_pxp.registroMomento("*) Evaluando Modelo ", True)]
####################################################################
#  EVALUACION : Incluye evaluate y .predict con identico dataset TEST
scoresTrain = model.evaluate(test_entrada, test_salida)
test_predecido = model.predict(test_entrada)
confundida = confusion_matrix(test_salida, test_predecido.round())
confundidaPxp = pd_pxp.matrizConfusionPxp(confundida)



logPrograma += [pd_pxp.registroMomento("*) Guardando Modelo ", True)]
#####################################################################
# Guardar resultado del entrenamiento
model_json = model.to_json()
with open(archivoGuardaModelo, "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights(archivoGuardaPesos)

#SACAR DESDE ACA
# guardo las transformaciones
dump(coltra, open(archivoGuardaTransformaciones, "wb"))
#SACARE HASTA ACA

##################################################################################
logPrograma += [pd_pxp.registroMomento("*) Armando resultados finales ", True)]
resultadoContraTest = "\nEvaluacion contra grupo de test :\n" + \
                      "     {metrica} : {valormetrica:.2f}% \n" + \
                      "     {metrica0} : {valormetrica0:.2f}%\n"
resultadoContraTest = resultadoContraTest.format(metrica=model.metrics_names[1],
                                                 valormetrica=scoresTrain[1] * 100,
                                                 metrica0=model.metrics_names[0],
                                                 valormetrica0=scoresTrain[0] * 100)

resultadoEvaluacion = confundidaPxp.obtenerResultadoEvaluacion()

fileMem = io.StringIO()
with redirect_stdout(fileMem):
    print(" Archivos de entrada: ")
    print(' -' + archivoDeEntrenamiento)
    print(
        " Archivos de salida : ")
    print(' -' + archivoGuardaModelo)
    print(' -' + archivoGuardaPesos)
    print(' -' + archivoInfoEntrenamiento)
    print(' -' + archivoGuardaTransformaciones)
    print(resultadoEvaluacion)
    print(confundida)
    print(resultadoContraTest)
    print('#' * 10, "MODEL SUMMARY", '#' * 10)
    model.summary()
    print('Epochs: ' + str(epocas))
    print('batch_size: ' + str(lote))
    print('split_validacion_fit_size: ' + str(split_validacion_fit_size))
    print('split_dataframe_para_test: ' + str(split_dataframe_para_test))
    print('#' * 10, "SHAPE ENTRENAMIENTO", '#' * 10)
    print(train_entrada.shape)
    print(train_entrada.columns.tolist())
    print('#' * 10, "SHAPE TEST", '#' * 10)
    print(test_entrada.shape)
    print(train_entrada.columns.tolist())
    print('#' * 10, "Analisis de variabilidad", '#' * 10)
    print(analisisVariabilidad)
    print('#' * 10, "Log de performance", '#' * 10)
    logPrograma += [pd_pxp.registroMomento("*) FIN DEL PROCESO ", True)]
    for mom in logPrograma:
        print(mom)

# print(fileMem.getvalue())
with open(archivoInfoEntrenamiento, 'w') as fileDisk:
    fileDisk.write(fileMem.getvalue())  # Lo abro con w entonces lo pisa

# Imprimo
pd_pxp.mostrarPerformanceEntrenamiento(history, archivoPrefijoPlot, mostrarGraficos)
pd_pxp.mostrarMatriz(confundida, archivoPrefijoPlot, mostrarGraficos)

prefijo = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S")+' '+' '#+str(tp)+' '
destino = 'Historia/'+prefijo+archivoInfoEntrenamiento
print('* Realizando copia en: '+destino)
# Copia el archivo desde la ubicación actual a Historia
shutil.copy(archivoInfoEntrenamiento, destino)
####################################### SALIDA FINAL
print("Modelo Guardado! Informacion en : " + archivoInfoEntrenamiento)
print('(' * 15, "SALIDA FINAL", ')' * 15)
print(resultadoEvaluacion)
print('Precision tp/(tp+fp): {mcPrecision:,.2%}'.format(mcPrecision=confundidaPxp.mcPrecision))
print('Recall    tp/(tp+fn): {mcRecall:,.2%}'.format(mcRecall=confundidaPxp.mcRecall))
print('Acurracy         : {mcAccuracy:,.2%}'.format(mcAccuracy=confundidaPxp.mcAccuracy))
