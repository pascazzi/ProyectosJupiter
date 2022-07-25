# INSTRUCTIVO DEPURADOR

def imprimeSaludos(nombre):
    contador=1
    print("hola")
    contador+=1
    print("Cómo estas ",nombre)
    contador+=1
    nombre=" Sr "+nombre
    contador+=1
    print("Quise decir, como está usted ",nombre)
    return contador


buchon=1
print("**HOLA**")
for x in range (5):
    buchon+=10
    print(x,sep=" - ",end="")

buchon = buchon + imprimeSaludos("Pablo")
buchon = buchon + imprimeSaludos("Juan")

print(buchon)
print("*** chau  ***")