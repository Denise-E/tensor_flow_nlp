import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class SubjectModel:
    vocab_size = 10000 #Máx num de palabras que se deben conservar, si se le pasan más va a guardar las 10000 más repetidas.
    embedding_dim = 16
    max_length = 10000
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>" #Token para todas aquellas palabras no tokenizadas
    training_size = 20000
    tokenizer = None
    model = None

    @classmethod
    def __init__(cls):
        cls.model_config_and_training()
        
    @classmethod
    def model_config_and_training(cls):
        # 1. Definición de los strings de capacitación 
        sentences = [ 
            "Denuncia de Siniestro",
            "Denuncia de un siniestro",
            "Denuncia siniestro",
            "siniestro",
            "denuncia",
            "Presupuestos Solicitados",
            "presupuestos",
            "cotizacion",
            "Cotización de pólizas de auto",
            "Cotización de polizas de auto",
            "Cotizacion de pólizas de auto",
            "Cotizacion de polizas de auto",
            "polizas de auto",
            "pólizas de auto",
            "Cotización de pólizas del hogar",
            "Cotizacion de polizas del hogar",
            "polizas del hogar",
            "pólizas del hogar",
            "Cotización pólizas auto",
            "Cotización polizas auto",
            "Cotizacion pólizas auto",
            "Cotizacion polizas auto",
            "polizas auto",
            "pólizas auto",
            "Cotización pólizas hogar",
            "Cotización polizas hogar",
            "polizas hogar",
            "pólizas hogar",
            "Vacaciones!", 
            "Descuentos solo para vos",
            "solicitud Denuncia de Siniestro",
            "solicitud Denuncia de un siniestro",
            "solicitud Denuncia siniestro",
            "solicitud siniestro",
            "solicitud denuncia",
            "solicitud Presupuestos Solicitados",
            "solicitud presupuestos",
            "solicitud cotizacion",
            "solicitud Cotización de pólizas de auto",
            "solicitud Cotización de polizas de auto",
            "solicitud Cotizacion de pólizas de auto",
            "solicitud Cotizacion de polizas de auto",
            "solicitud polizas de auto",
            "solicitud pólizas de auto",
            "solicitud Cotización de pólizas del hogar",
            "solicitud Cotizacion de polizas del hogar",
            "solicitud polizas del hogar",
            "solicitud pólizas del hogar",
            "solicitud Cotización pólizas auto",
            "solicitud Cotización polizas auto",
            "solicitud Cotizacion pólizas auto",
            "solicitud Cotizacion polizas auto",
            "solicitud polizas auto",
            "solicitud pólizas auto",
            "solicitud Cotización pólizas hogar",
            "solicitud Cotización polizas hogar",
            "solicitud polizas hogar",
            "solicitud pólizas hogar",
            "pedido Denuncia de Siniestro",
            "pedido Denuncia de un siniestro",
            "pedido Denuncia siniestro",
            "pedido siniestro",
            "pedido denuncia",
            "pedido Presupuestos Solicitados",
            "pedido presupuestos",
            "pedido cotizacion",
            "pedido Cotización de pólizas de auto",
            "pedido Cotización de polizas de auto",
            "pedido Cotizacion de pólizas de auto",
            "pedido Cotizacion de polizas de auto",
            "pedido polizas de auto",
            "pedido pólizas de auto",
            "pedido Cotización de pólizas del hogar",
            "pedido Cotizacion de polizas del hogar",
            "pedido polizas del hogar",
            "pedido pólizas del hogar",
            "pedido Cotización pólizas auto",
            "pedido Cotización polizas auto",
            "pedido Cotizacion pólizas auto",
            "pedido Cotizacion polizas auto",
            "pedido polizas auto",
            "pedido pólizas auto",
            "pedido Cotización pólizas hogar",
            "pedido Cotización polizas hogar",
            "pedido polizas hogar",
            "pedido pólizas hogar",
            "Conocé Datadog",
            "Conoce Datadog",
            "Feliz cumpleaños",
            "conocé nuestras ofertas",
            "conoce nuestros beneficios",
            "nuevo mensaje",
            "descubre todas las novedades",
            "Últimos días",
            "Nuevo lanzamiento",
            "Nuevos lanzamientos",
            "pedido Cotización de polizas del hogar",
            "pedido Cotizacion de pólizas del hogar",
            "pedido Vacaciones en Mardel",
            "pedido Promoción en ropa",
            "pedido Cotizacion pólizas hogar",
            "pedido Cotizacion polizas hogar",
            "pedido Carga de presupuestos",
            "pedido presupuestos a cargar",
            "pedido Por favor cotizar",
            "solicitud Cotización de polizas del hogar",
            "solicitud Cotizacion de pólizas del hogar",
            "solicitud Vacaciones en Mardel",
            "solicitud Promoción en ropa",
            "solicitud Cotizacion pólizas hogar",
            "solicitud Cotizacion polizas hogar",
            "solicitud Carga de presupuestos",
            "solicitud presupuestos a cargar",
            "solicitud Por favor cotizar",
            "solicitud cotización"
        ] #116 casos, 41 tokens
        # 1 = se vincula a nuetsros tramites, 0 = no se vincula
        training_labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] 

        # 2. Tokenización de las sentencias
        # No distingue minusculas y mayúsculas, tampoco simbolos. Si reconoce acentos.
        tokenizer = Tokenizer(num_words=cls.vocab_size, oov_token=cls.oov_tok) 
        tokenizer.fit_on_texts(sentences)
        cls.tokenizer = tokenizer
        word_indexed = tokenizer.word_index
        print("indexed  WORDS", word_indexed)

        training_sentences = tokenizer.texts_to_sequences(sentences)
        training_padded = pad_sequences(training_sentences,maxlen=cls.max_length, padding='post') #Hay que deifnirle max_length por concordancia con resto del proceso
        print("PADDED", training_padded)
        #print("PADDED SHAPE", training_padded.shape) #Filas x columns 

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(cls.vocab_size, cls.embedding_dim, input_length=cls.max_length), 
            tf.keras.layers.GlobalAveragePooling1D(), 
            tf.keras.layers.Dense(24, activation='relu'), 
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        #Model configuration
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        cls.model = model

        # Model testing
        testing_sentences = [
            "Cotización de polizas del hogar", 
            "Cotizacion de pólizas del hogar", 
            "Vacaciones en Mardel", 
            "Promoción en ropa",
            "Cotizacion pólizas hogar",
            "Cotizacion polizas hogar",
            "Carga de presupuestos",
            "presupuestos a cargar",
            "Por favor cotizar",
            "cotización"
        ] #10
        testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sentences, maxlen=cls.max_length, padding='post') 
        testing_labels = [1,1,0,0,1,1,1,1,1,1]

        training_padded = np.array(training_padded)
        training_labels = np.array(training_labels)
        testing_padded = np.array(testing_padded)
        testing_labels = np.array(testing_labels)

        num_epochs = 1000 #Trains the model for a fixed number of epochs (dataset iterations).
        print("TEST RESULTS")
        history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
        """
        Result fields:
        accuracy = training data related
        val_accuracy = testing data related
        """

    @classmethod
    def model_prediction_tests(cls, sentence: str) -> list[list[int]]:
        sequences = cls.tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, maxlen=cls.max_length, padding=cls.padding_type, truncating=cls.trunc_type)
        return cls.model.predict(padded).tolist() 


subject_model = SubjectModel()
sentences = ["Solicitud cotizacion póliza del hogar", "Solicitud póliza del hogar"]
prediction_1 = subject_model.model_prediction_tests(sentence=sentences)
print("SENTENCES:",sentences)
print("PREDICTION:", prediction_1)
