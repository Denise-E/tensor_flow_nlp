import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. Definición de los strings a trabajar
sentences = [
    'I love my dog',
    'I love my cat',
    'I love my dog!', # No toqueniza !
    'I love my dog !',
    'Do you think my dog is amazing?'# Caso de distinta extensión de palabras
]

# 2. Tokenización de las sentencias
"""
Hay que manejar los casos en los que después le pasemos palabras con las que no capacitamos al motor, ya que no van a estar tokenizadas y pueden confundir al tokenizador
Si no tienen token asociado, no vamos a ver esa palabra en la sequence. Simplemente devuelve aquellas palabras con las que se lo capacitó.
Se hace un manejo para que aunqsea me devuelva el mismo length, devolviendo <OOV> para esas palabras no tokerizadas
"""
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") #Máx num de palabras que se deben conservar, si se le pasan más va a guardar las 100 más repetidas.
tokenizer.fit_on_texts(sentences) #El tokenizer recorre el texto que le pases y tokenizar num_words cantidad de palabras
word_index = tokenizer.word_index #Arma lista de palabras indexadas
print("INDEX WORDS", word_index)

# 3. Preparación palabras tokerizadas para capacitar al motor de TF - Regularización
sequences = tokenizer.texts_to_sequences(sentences)
print("SEQUENCES", sequences) #Crea la secuencia de tokens para cada oración. Lista de listas
#Con post me agrega los 0 al final y no al principio
padded = pad_sequences(sequences, padding='post') #Hace que todas las oraciones sean del mismo length, con 0 que no se asocia a ningún token
print("PADDED", padded)




