import pandas as pd

print("*** VERSION pd")
print(pd.__version__)
# dataframe vacio
df = pd.DataFrame()
# dataframe vacio pero con columnas
df2= pd.DataFrame(columns=['Nombre', 'Apellido', 'Sexo'])

print("*** DIR DF")
print(dir(df))
print("*** TYPE DF")
print(type(df))
#----------------------------------------------------
#agrego valores multiples
df['first_name'] = ['Josy', 'Vaughn', 'Neale', 'Teirtza']
df['last_name'] = ['Clarae', 'Halegarth', 'Georgievski', 'Teirtza']
df['gender'] = ['Female', 'Male', 'Male', 'Female']
#-----------------------------------------------------
print(df.columns)
print(df.sample(2))

#agrego filas
df2 = df.append({'Nombre': 'Josy', 'Apellido':'Clarae', 'Sexo':'Female'}, ignore_index=True)
df2 = df.append({'Nombre': 'Vaughn', 'Apellido':'Halegarth', 'Sexo':'Male'}, ignore_index=True)
print(df2.columns)
print(df2.sample(2))

# OTRA FORMA: Agregar columnas con nombres, pero filas vacias. Despues se isa iloc para rellenar.
print("** AHORA DF3 **")
df3 = pd.DataFrame(columns=['first_name', 'last_name', 'gender'],
                  index=range(3))
print(df3.columns)
print(df3.sample(2))
df3.iloc[0] = ('Josy', 'Clarae', 'Female')
df3.iloc[1] = ['Vaughn', 'Halegarth', 'Male']
df3.iloc[2] = ('Neale', 'Georgievski', 'Male')
print(df3.columns)
print(df3.sample(2))
print("\nFIN!")

