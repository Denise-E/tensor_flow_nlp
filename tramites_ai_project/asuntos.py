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
            "denuncia de Siniestro",
            "denuncia de un siniestro",
            "denuncia siniestro",
            "siniestro",
            "denuncia",
            "presupuestos Solicitados",
            "presupuestos",
            "cotizacion",
            "cotización de pólizas de auto",
            "cotización de polizas de auto",
            "cotizacion de pólizas de auto",
            "cotizacion de polizas de auto",
            "polizas de auto",
            "pólizas de auto",
            "cotización de pólizas del hogar",
            "cotizacion de polizas del hogar",
            "polizas del hogar",
            "pólizas del hogar",
            "cotización pólizas auto",
            "cotización polizas auto",
            "cotizacion pólizas auto",
            "cotizacion polizas auto",
            "polizas auto",
            "pólizas auto",
            "cotización pólizas hogar",
            "cotización polizas hogar",
            "polizas hogar",
            "pólizas hogar",
            "vacaciones!", 
            "descuentos solo para vos",
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
            "conocé Datadog",
            "conoce Datadog",
            "feliz cumpleaños",
            "conocé nuestras ofertas",
            "conoce nuestros beneficios",
            "nuevo mensaje",
            "descubre todas las novedades",
            "últimos días",
            "nuevo lanzamiento",
            "nuevos lanzamientos",
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
            "solicitud cotización",
            "necesito cotización",
            "necesito póliza",
            "necesito presupuesto",
            "cotización de pólizas de la casa",
            "cotizacion de polizas de la casa",
            "cotización de pólizas casa",
            "cotizacion de polizas casa",
            "cotización de pólizas vivienda",
            "cotizacion de polizas vivienda",
            "cotización de pólizas depto",
            "cotizacion de polizas depto",
            "cotización de pólizas departamento",
            "cotizacion de polizas departamento",
            "cotización de polizas veiculo",
            "cotizacion de pólizas veiculo",
            "cotizacion de polizas veiculo",
            "cotisasión de pólizas veiculo",
            "cotisasión de polizas veiculo",
            "cotisasion de pólizas veiculo",
            "cotisasion de polizas veiculo",
            "cotisasión de pólizas de auto",
            "cotisasión de polizas de auto",
            "cotisasion de pólizas de auto",
            "cotisasion de polizas de auto",
            "cotización de pólizas vehiculo",
            "cotización de polizas vehiculo",
            "cotizacion de pólizas vehiculo",
            "cotizacion de polizas vehiculo",
            "cotisasión de pólizas vehiculo",
            "cotisasión de polizas vehiculo",
            "cotisasion de pólizas vehiculo",
            "cotisasion de polizas vehiculo",
            "polisas de auto",
            "pólisas de auto",
            "¡gana un millón de dólares ahora!",
            "oferta especial: ¡Descuento del 50 porciento de descuento solo por hoy!",
            "aumenta tu puntaje de crédito al instante",
            "¡compra seguidores y likes para tus redes sociales!",
            "conoce a solteros locales cerca de ti",
            "reunión de equipo esta tarde a las 3 p.m.",
            "confirmación de reserva para la conferencia",
            "actualización semanal del proyecto",
            "recuerda enviar el informe antes del viernes",
            "invitación a la presentación del nuevo producto",
            "¡gana un iPhone gratis!",
            "¡oferta exclusiva: descuento del 70% en productos electrónicos!",
            "aumenta tus seguidores en redes sociales al instante",
            "¡deshazte de la deuda en 24 horas!",
            "¡sorteo de vacaciones todo incluido!",
            "¡conoce solteros calientes en tu área!",
            "¡baja de peso rápidamente con esta pastilla milagrosa!",
            "¡dinero fácil y rápido: hazte rico en una semana!",
            "¡prueba gratis nuestro producto y gana una tarjeta de regalo!",
            "¡increíble oportunidad de inversión con altos rendimientos!",
            "¡descubre el secreto para una piel perfecta!",
            "¡aprovecha esta oferta única: préstamos sin intereses!",
            "¡gana un viaje de lujo a las Bahamas!",
            "¡incrementa tus ingresos con este sistema probado!",
            "¡obtén un préstamo sin verificación de crédito!",
            "¡haz crecer tu negocio con nuestro software revolucionario!",
            "¡oferta limitada: suscríbete ahora y recibe un descuento adicional!",
            "¡descubre cómo ganar dinero desde casa!",
            "¡productos de belleza gratis solo por registrarte!",
            "¡tu cuenta ha sido seleccionada para recibir un premio especial!",
        ] #116 casos, 51 tokens
        # 1 = se vincula a nuestros tramites, 0 = no se vincula
        training_labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 

        # 2. Tokenización de las sentencias
        # No distingue minusculas y mayúsculas, tampoco simbolos. Si reconoce acentos.
        tokenizer = Tokenizer(num_words=cls.vocab_size, oov_token=cls.oov_tok) 
        tokenizer.fit_on_texts(sentences)
        cls.tokenizer = tokenizer
        word_indexed = tokenizer.word_index
        print("Indexed  WORDS", word_indexed)

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
            "cotización de polizas del hogar", 
            "cotizacion de pólizas del hogar", 
            "vacaciones en Mardel", 
            "promoción en ropa",
            "cotizacion pólizas hogar",
            "cotizacion polizas hogar",
            "carga de presupuestos",
            "presupuestos a cargar",
            "por favor cotizar",
            "cotización",
            
            "recordatorio de pago de factura pendiente",
            "confirmación de reserva de hotel",
            "actualización de política de privacidad",
            "información sobre cambios en el horario de atención al cliente",
            "recordatorio de cita médica",
            "confirmación de registro en el evento",
            "notificación de entrega de paquete",
            "resumen mensual de cuenta bancaria",
            "invitación a evento de networking empresarial",
            "actualización sobre el estado de tu solicitud",
            "confirmación de reserva de vuelo",
            "recordatorio de fecha de vencimiento de suscripción",
            "informe de rendimiento trimestral",
            "recordatorio de reunión de equipo",
            "confirmación de compra en línea",
            "invitación a participar en encuesta de satisfacción",
            "notificación de cambio de contraseña de cuenta",
            "resumen de actividad en la cuenta de redes sociales",
            "recordatorio de renovación de membresía",
            "invitación a seminario web sobre desarrollo profesional"
        ] #10
        testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sentences, maxlen=cls.max_length, padding='post') 
        testing_labels = [1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        training_padded = np.array(training_padded)
        training_labels = np.array(training_labels)
        testing_padded = np.array(testing_padded)
        testing_labels = np.array(testing_labels)

        num_epochs = 3000 #Trains the model for a fixed number of epochs (dataset iterations).
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
        print("PADDED", padded)
        prediction = cls.model.predict(padded)
        #return [prediction.tolist() , np.argmax(prediction,axis=-1)]
        return [prediction.tolist() , (prediction > 0.6).astype("int32")] #Valor a ajustar una vez que vaya sumando casos para la capacitación del modelo


subject_model = SubjectModel()
sentences = ["solicitud cotizacion póliza del hogar", "solicitud póliza del hogar", "que seas feliz", "vacaciones en Mardel", "notificación membresia", "notificación membresía"]
prediction_1 = subject_model.model_prediction_tests(sentence=sentences)
print("SENTENCES:",sentences)
print("PREDICTION:", prediction_1) 

# 0.8957 - 0.80
# Predict iterando 1000 veces [[0.8933809995651245], [0.892857551574707], [0.8909379839897156]]
# Predict iterando 2000 veces [[0.8990367650985718], [0.8960583806037903], [0.8848827481269836]]

"""
Con más casos que retornan 0
["solicitud cotizacion póliza del hogar", "solicitud póliza del hogar", "que seas feliz", "vacaciones en Mardel", "notificación membresia", "notificación membresía"]
[0.9988852143287659], [0.9937644004821777], [0.5036671161651611], [0.6806588172912598], [0.6968783736228943], [0.6968783736228943]]
"""
