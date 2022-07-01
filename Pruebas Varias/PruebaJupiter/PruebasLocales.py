import pandas as pd
#import os

def pruebaFun(df):
    df = df.drop(df.columns[[0]], axis='columns')
    return df

#print(os.path.basename(__file__))
#print(os.path.dirname(__file__))
print(__file__)
print("inicio")
df = pd.DataFrame()
df["nombre"] = ["Pablo","juan","pedro"]
df["apellido"] = ["pasca","ford","picapiedra"]
print(df)
print("llamo a funcion que borra columna 1")
df=pruebaFun(df)
print(df)