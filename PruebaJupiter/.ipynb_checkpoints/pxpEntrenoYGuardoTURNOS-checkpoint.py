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
import datetime
import shutil
import pxpLibreriaDataFrame as pd_pxp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from contextlib import redirect_stdout

logPrograma = []
logPrograma += [pd_pxp.registroMomento("*) Iniciando programa de entrenamiento "+__file__, True)]

##################################################################
# Parametrizo
##################################################################
archivoGuardaModelo = "Turnos_Modelo.json"
archivoGuardaPesos = "Turnos_Peso.h5"
archivoBalanceado = "Turnos_Tomados_Balanceados.csv"
#"TURNOS_TOMADOS_ENEROMARZO_peque8.csv"  # "DiabetesColor_ParaEntrenamiento.csv"
archivoInfoEntrenamiento = "Turnos_Info_Entrenamiento.txt"

archivoPrefijoPlot = "Turnos_graph_"
columnaObjetivo = 'AUSENTE'
mostrarGraficos = True

split_dataframe_para_test = 0.20
epocas = 150  # 150
lote = 20
split_validacion_fit_size = 0.20
mezcla = True

##################################################################
# Levanto Datos
df_transformada  = pd.read_csv(archivoBalanceado, sep=';')

#Elimino columnas absurdas (en este caso la creada al guardar el balanceo)
df_transformada = df_transformada.drop(df_transformada.columns[[0]],axis='columns')
print(df_transformada.shape)
print(df_transformada.sample(5))

#print(type(df_transformada.values[1,1]))
#input("pausa")

##################################################################
# Separo dataframes de train y test
logPrograma += [pd_pxp.registroMomento("*) Iniciando Split de dataframes ", True)]
train, test = train_test_split(df_transformada, test_size=split_dataframe_para_test, random_state=42, shuffle=mezcla)

# PREPARO ENTRENAMIENTO
train_entrada = pd.DataFrame(train).drop([columnaObjetivo], axis=1)  # Todos menos lo que voy a predecir
train_salida = pd.DataFrame(train[columnaObjetivo])  # Lo que voy a predecir

# PREPARO TEST
test_entrada = pd.DataFrame(test).drop([columnaObjetivo], axis=1)  # Todos menos lo que voy a predecir
test_salida = pd.DataFrame(test[columnaObjetivo])  # Lo que voy a predecir



# Verifico Variabilidad
analisisVariabilidad = ''
sets = (('train_salida', train_salida), ('test_salida', test_salida))
for set_name, set_data in sets:
    analisisVariabilidad = analisisVariabilidad + '-' * 10 + ' ' + set_name + ' ' + '-' * 10 + "\n"
    analisisVariabilidad = analisisVariabilidad + set_data[columnaObjetivo].value_counts().to_string() + "\n"
print(analisisVariabilidad)

# Entrenar y Fit
numpy.random.seed(7)
q_dim_entradas = train_entrada.columns.size

#################### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#################### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# crea el modelo
model = Sequential()
model.add(Dense(8, input_dim=q_dim_entradas, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.07))
model.add(Dense(1, activation='sigmoid'))
# sigmoid  capa de salida, 1 neurona, Espero 1 salida: cero o uno. Recibe todas las salidas de la capa anterior
# por eso es densa. Pero debe devolver un numero entre cero y uno.
'''
#CONVOLUTIONAL + DENSE
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation=’relu’))
model.add(Flatten())
model.add(Dense(10, activation=’softmax’))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''

logPrograma += [pd_pxp.registroMomento("*) Compilando Modelo ", True)]

# Compila el modelo: Internamente armo el grafo.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# adam es la mas completa de las funciones de ajustar pesos (adam: variante de pesos por gradiente)
# funcion de perdida: binary_crossentropy ideal para salidas entre cero y uno.
# metrics: lo que va imprimiendo de salida.

logPrograma += [pd_pxp.registroMomento("*) Creando Modelo ", True)]
# Ajusta el modelo
print(train_salida.head())
#weights = class_weight.compute_class_weight('balanced',numpy.unique(itt),itt)

history = model.fit(x=train_entrada, y=train_salida, epochs=epocas,
                    validation_split=split_validacion_fit_size,
                    batch_size=lote,
                    #class_weight=weights,
                    shuffle=mezcla)  # 72.64 #72.80

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
# guardo las transformaciones
#dump(coltra, open(archivoGuardaTransformaciones, "wb"))

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
    print(' -' + archivoBalanceado)
    print(
        " Archivos de salida : ")
    print(' -' + archivoGuardaModelo)
    print(' -' + archivoGuardaPesos)
    print(' -' + archivoInfoEntrenamiento)
    #print(' -' + archivoGuardaTransformaciones)
    print(resultadoEvaluacion)
    print(pd.DataFrame(confundida, range(2), range(2)).sample(2))
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
    logPrograma += [
        pd_pxp.registroMomento("*) FIN DEL PROCESO ", False)]  # no lo muestro por pantalla por que es stdout un string
    for mom in logPrograma:
        print(mom)

# print(fileMem.getvalue())
with open(archivoInfoEntrenamiento, 'w') as fileDisk:
    fileDisk.write(fileMem.getvalue())  # Lo abro con w entonces lo pisa

# Imprimo
import winsound

winsound.Beep(440, 500)
winsound.Beep(840, 500)
pd_pxp.mostrarPerformanceEntrenamiento(history, archivoPrefijoPlot, mostrarGraficos)
pd_pxp.mostrarMatriz(confundida, archivoPrefijoPlot, mostrarGraficos)

prefijo = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S") + ' ' + ' '  # +str(tp)+' '
destino = 'Historia/' + prefijo + archivoInfoEntrenamiento
print('* Realizando copia en: ' + destino)
# Copia el archivo desde la ubicación actual a Historia
shutil.copy(archivoInfoEntrenamiento, destino)

####################################### SALIDA FINAL
print("Modelo Guardado! Informacion en : " + archivoInfoEntrenamiento)
print('(' * 15, "SALIDA FINAL", ')' * 15)
print(resultadoEvaluacion)
print('Precision tp/(tp+fp): {mcPrecision:,.2%}'.format(mcPrecision=confundidaPxp.mcPrecision))
print('Recall    tp/(tp+fn): {mcRecall:,.2%}'.format(mcRecall=confundidaPxp.mcRecall))
print('Acurracy         : {mcAccuracy:,.2%}'.format(mcAccuracy=confundidaPxp.mcAccuracy))
