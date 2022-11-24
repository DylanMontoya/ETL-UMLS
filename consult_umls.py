import pickle

import numpy as np
from nltk.tokenize import LineTokenizer, RegexpTokenizer, word_tokenize

#%% Funcion para crear etiquetas y funcion para hacer etiquetado en la EHR

with open('my_vocabulary.pkl', 'rb') as file:
    my_vocabulary = pickle.load(file)
    my_vocabulary = my_vocabulary.reset_index(drop = True)
    # my_vocabulary = my_vocabulary['STR'].apply(lambda x: [item for item in x if item not in stopwords.words('spanish')])
    

def Etiqueta(i):

    # Creo los conjuntos de mis entidades

    anatomia= set(['anatomical structure', 'body system', 'body part, organ, or organ component', 
                   'body location or region', 'body space or junction'])

    signo_sintoma = set(['finding','injury or poisoning', 'physiologic function', 'pathologic function', 
                         'sign or symptom', 'organism function'])

    problema_clinico = set(['congenital abnormality', 'disease or syndrome', 'mental or behavioral dysfunction', 
                            'anatomical abnormality'])

    sustancia = set(['body substance', 'chemical', 'pharmacologic substance', 'biologically active substance', 
                     'hazardous or poisonous substance', 'substance', 'antibiotic', 'clinical drug'])

    procedimiento = set(['laboratory or test result', 'health care activity', 'laboratory procedure', 
                         'diagnostic procedure', 'therapeutic or preventive procedure', 'research activity'])

    atributo = set(['temporal concept', 'qualitative concept', 'quantitative concept', 'spatial concept'])
    
    # Para hacer etiquetado
    
    if my_vocabulary.values[i][-1] in anatomia: etiqueta = 'Anatomia'

    elif my_vocabulary.values[i][-1] in signo_sintoma: etiqueta = 'SignoSintoma'

    elif my_vocabulary.values[i][-1] in problema_clinico: etiqueta = 'ProblemaClinico'

    elif my_vocabulary.values[i][-1] in sustancia: etiqueta = 'Sustancia'
    
    elif my_vocabulary.values[i][-1] in procedimiento: etiqueta = 'Procedimiento'
    
    elif my_vocabulary.values[i][-1] in atributo: etiqueta = 'Atributo'
        
    return etiqueta

def etiquetado(tokens):
    
    #tokens = [ t.lower() for t in word_tokenize(data) ] #if t.isalpha()
    #filtered_tokens = [ t for t in tokens if t not in stopwords.words('spanish') ]
    etiquetasbio = ['O'] * len(tokens)
    eti_bi = ['O'] * len(tokens)
    eti_tri = ['O'] * len(tokens)
    arreglo = np.transpose(np.vstack((tokens, etiquetasbio, eti_bi, eti_tri))) #
    
    # arreglo = np.transpose(np.vstack((tokens, etiquetasbio)))
    
    token = 0
    while token < len(arreglo):
        for i in range(len(my_vocabulary)):
            if arreglo[:,0][token] == my_vocabulary['STR'].values[i]:
                arreglo[token, 1] = 'B-' + Etiqueta(i)
                
                if token < len(arreglo[:,0]) - 1:
                    if len(arreglo[:,0][token + 1]) > 2:
                        
                        for j in range(len(my_vocabulary)):
                            if arreglo[:,0][token] + ' ' + arreglo[:,0][token+1] == my_vocabulary['STR'].values[j]:
                                arreglo[token, 2] = 'B-' + Etiqueta(i=j)
                                arreglo[token+1, 2] = 'I-' + Etiqueta(i=j)
                                # token = token + 1 
                    else:
                    
                        for k in range(len(my_vocabulary)):
                            if arreglo[:,0][token] + ' ' + arreglo[:,0][token + 1] + ' ' + arreglo[:,0][token + 2] == my_vocabulary['STR'].values[k]:
                                arreglo[token, 3] = 'B-' + Etiqueta(i=k)
                                arreglo[token+1, 3] = 'I-' + Etiqueta(i=k)
                                arreglo[token+2, 3] = 'I-' + Etiqueta(i=k)                       
                                # token = token + 2
                                
        token += 1
    
    return arreglo
