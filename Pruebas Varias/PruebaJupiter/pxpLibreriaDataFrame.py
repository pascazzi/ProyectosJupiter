import os
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

######################### AYUDA #############################
# pd.set_option("display.max_columns", 600)
# dfdescribe=df_transformada.describe()
# print(dfdescribe.columns)
################### opciones
# desired_width = 620
# pd.set_option('display.width', desired_width)
###################
######################### FIN AYUDA #############################

'''

def _ActualizaDicc(df1, diccionario, nombreColumna):
    columna = df1[nombreColumna]
    y_valor = np.zeros((len(columna)))
    for i1, raw_label1 in enumerate(columna):
        if raw_label1 not in diccionario:
            diccionario[raw_label1] = len(diccionario)
        y_valor[i1] = diccionario[raw_label1]

def crearActualizarDiccionario(df1, nombreColumna, nombre=None):
    nombreDiccionario = nombrarDiccionario(nombre, nombreColumna)
    existe = os.path.isfile(nombreDiccionario)
    if (existe):
        #levanto diccionario
        print(nombreDiccionario + ' existe')
        diccionario = _LevantarDiccionario(nombreDiccionario)
        _ActualizaDicc(df1, diccionario, nombreColumna)
    else:
        #creo diccionario
        print(nombreDiccionario + ' no existe')
        diccionario = {}
        _ActualizaDicc(df1, diccionario, nombreColumna)
    #guardo diccionario actualizado
    tf = open(nombreDiccionario, "w")
    json.dump(diccionario, tf)
    tf.close()
    return diccionario


def nombrarDiccionario(nombreColumna, nombre=None):
    if (nombre is None):
        nombreDiccionario = nombreColumna + '.json'
    else:
        nombreDiccionario = nombre + '.json'
    return nombreDiccionario

def _LevantarDiccionario(nombreDiccionario):
    tf = open(nombreDiccionario, "r")
    diccionario = json.load(tf)
    tf.close()
    return diccionario

def levantarDiccionario(nombreColumna, nombre=None):
   nombreDiccionario = nombrarDiccionario(nombreColumna, nombre)
   return _LevantarDiccionario(nombreDiccionario)

def TraducirANumeros(df1, nombreColumna, diccionario):
    df1["COLOR"] = df1[nombreColumna].map(diccionario)

def traducirDeNumeros(df1, nombreColumna, diccionario):
    diccionarioInvertido = {v: k for k, v in diccionario.items()}
    print(diccionarioInvertido)
    df1["COLOR"]=df1[nombreColumna].map(diccionarioInvertido)
    
'''


class matrizConfusionPxp:
    confundida= None
    def __init__(self,matrizConfusion):
        self.confundida=matrizConfusion
        tn, fp, fn, tp = matrizConfusion.ravel()
        self.tn=tn
        self.fp=fp
        self.fn=fn
        self.tp=tp
        self.mcPrecision = tp / (tp + fp)
        self.mcRecall = tp / (tp + fn)
        self.mcAccuracy = (tp + tn) / (tp + tn + fp + fn)  # tasa de aciertos
        self.mcTotal = tn + fp + fn + tp
        self.mcTotalAciertos = tp + tn

    def obtenerResultadoEvaluacion(self):
        resultadoEvaluacion = ''.join(('#' * 10, " RESULTADOS ", '#' * 10))
        resultadoEvaluacion += "\nMatriz de confusion :\n" + \
                               "     Precision tp/(tp+fp): {mcPrecision:,.2%} \n" + \
                               "     Recall    tp/(tp+fn): {mcRecall:,.2%} \n" + \
                               "     Casos totales    : {mcTotal}\n" + \
                               "     Aciertos totales : {mcTotalAciertos}\n" + \
                               "     Acurracy         : {mcAccuracy:,.2%} \n" + \
                               matrizToString(self.tn, self.fp, self.fn, self.tp) + \
                               " \n" + \
                               matrizToString(round(self.tn/self.mcTotal,2),\
                                              round(self.fp/self.mcTotal,2),\
                                              round(self.fn/self.mcTotal,2),\
                                              round(self.tp/self.mcTotal,2))
        #"{:.2%}".2%           #"{:.2%}".2%
        resultadoEvaluacion += "\nPrecision : % de certeza de que son positivos los atrapados\n" + \
                               "Recall    : % de positivos atrapados por la red\n" + \
                               "Accurracy: % de aciertos totales (positivos y negativos)\n"
        resultadoEvaluacion = resultadoEvaluacion.format(mcPrecision=self.mcPrecision,
                                                         mcRecall=self.mcRecall,
                                                         mcTotal=self.mcTotal,
                                                         mcTotalAciertos=self.mcTotalAciertos,
                                                         mcAccuracy=self.mcAccuracy)
        return resultadoEvaluacion

class registroMomento:
    titulo = 'Sin titulo aun'
    momento = datetime.datetime.now()
    linea = ''

    def __init__(self, titulo, mostrar=False):
        self.titulo = titulo
        self.momento = datetime.datetime.now()
        self.linea = self.momento.strftime("%m/%d/%Y %H:%M:%S")+' '+self.titulo
        if mostrar :
            print(self.linea )

    def __str__(self):
        return(self.linea)


def ExplorarDatosPrevios(df):
    nombres=[]
    datos=[]
    for col in df.columns:
        print(col + '   ' +str(df[col].nunique()))
        nombres.append(col)
        datos.append(df[col].nunique())

    plt.barh(nombres, datos, color='maroon')

    plt.xlabel("Courses offered")
    plt.ylabel("No. of students enrolled")
    plt.title("Students enrolled in different courses")
    plt.show()


import seaborn as sn
def mostrarMatriz(confundida, archivoPrefijoPlot, mostrar):
    df_cm = pd.DataFrame(confundida, range(2), range(2) )
    sn.set(font_scale=1.4) # for label size
    #sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
    sn.heatmap(df_cm, cmap="Greens", annot=True, fmt='d',  annot_kws={"size": 16}) # font size
    plt.xlabel("Prediccion")
    plt.ylabel("Realidad")
    plt.savefig(archivoPrefijoPlot + "_matrizConfusion.jpg", bbox_inches='tight')
    if mostrar:
        plt.show()
    plt.close()

def mostrarPerformanceEntrenamiento(history, archivoPrefijoPlot, mostrar):
    print(history.history.keys())
    # hacemos las grafica 1 precision
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precision del modelo')
    plt.ylabel('Precision')
    plt.xlabel('epocas')
    plt.legend(['Entrenamiento', 'test'], loc='upper left')
    plt.savefig(archivoPrefijoPlot+"_precision.jpg", bbox_inches='tight')
    if mostrar:
        plt.show()
    plt.close()
    # hacemos las grafica 2 perdidas
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perdidas del modelo')
    plt.ylabel('perdidas')
    plt.xlabel('epocas')
    plt.legend(['Entrenamiento', 'test'], loc='upper left')
    plt.savefig(archivoPrefijoPlot + "_perdida.jpg", bbox_inches='tight')
    if mostrar:
        plt.show()
    plt.close()


'''
El llamado era asi: no se si tiene sentido seguir usandolo
#mostrarMejoresCols = False
#kantidadColumnasBuscadas = 4
# Mostrar Mejores Columnas
if mostrarMejoresCols:
    pd_pxp.MostrarMejoresColumnas(df_transformada, train_entrada, train_salida, kantidadColumnasBuscadas)
    cero = input("\nPresionar 0 para terminar")
    if cero == '0':
        print("Chau!")
        quit()

def MostrarMejoresColumnas(df1, train_entrada, train_salida, kantidad):
    print('#' * 10, "INFORMACION del DATAFRAME", '#' * 10)
    df1.info()
    selector = SelectKBest(chi2, k=kantidad)
    selector.fit(train_entrada, train_salida)
    X_new = selector.transform(train_entrada)
    print(X_new.shape)

    vector_names = list(train_entrada.columns[selector.get_support(indices=True)])
    print('#' * 10, "MEJORES "+str(kantidad)+" COLUMNAS", '#' * 10)
    print(selector.get_support(indices=True))
    print(vector_names)

    print(selector.get_params())
'''

def mostrarPerformanceEntrenamientoJuntos(history):
    print(history.history.keys())
    fig, ax = plt.subplots(2, 1)
    # hacemos las grafica 1 precision
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].title.set_text('Precision del modelo')

    ax[0].set(xlabel="epocas", ylabel="Precision")
    ax[0].legend(['Entrenamiento', 'test'], loc='upper left')
    # hacemos las grafica 2 perdidas
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].title.set_text('Perdidas del modelo')
    ax[1].set(xlabel="epocas", ylabel="perdidas")
    ax[1].legend(['Entrenamiento', 'test'], loc='upper left')
    ax[1].legend(['Entrenamiento', 'test'], loc='upper left')
    plt.tight_layout()
    plt.show()

def matrizToString(tn, fp, fn, tp):
    texto = "\n " + \
                           "                          ESTIMACION\n " + \
                           "                    ---------------------\n " + \
                           "                     Negativos     Positivos\n " + \
                           "REALIDAD |Negativos |      {tn}    |   {fp}    |\n " + \
                           "         |Positivos |      {fn}    |   {tp}    |\n"
    texto = texto.format(tn=tn, fp=fp, fn=fn, tp=tp)
    return texto

def mostrarDistribucionDelObjetivo(df, columna, texto):
    positivos=df[columna].values.sum()
    negativos=df[columna].shape[0] - positivos
    totalFilas=df[columna].shape[0]
    texto_out = "********** "+texto+'\n'
    texto_out+= "Positivos: " + str(positivos)+'\n'
    texto_out+= "Negativos: " + str(negativos)+'\n'
    texto_out+= "Total    : " + str(totalFilas)+'\n'
    texto_out+= "*******************************"+'\n'
    return texto_out