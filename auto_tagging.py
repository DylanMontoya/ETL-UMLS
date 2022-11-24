import pickle
import re

import numpy as np
from consult_umls import Etiqueta, etiquetado, my_vocabulary
from nltk.tokenize import LineTokenizer, RegexpTokenizer


# PARA HACER CONSULTAS DE UNA PALABRA MEDICA
def test_consult(word):
    for i in range(len(my_vocabulary)):
        if my_vocabulary['STR'].values[i] == word:#'{}'.format(ngram_exam[j][0]):
            #print(i)
            #print(my_vocabulary.values[i])
            return print(my_vocabulary.values[i][1], '->', my_vocabulary.values[i][-1], '->', Etiqueta(i))
        
#%% ETIQUETADO EHR
def ehr_process(path):
    
    a,b = 'áéíóúüÁÉÍÓÚÜ','aeiouuAEIOUU' #'áéíóúüñÁÉÍÓÚÜ','aeiouunAEIOUU'
    trans = str.maketrans(a,b)
    #example = 'sangrado vaginal sangre examen para dolor de cabeza analisis de embarazo utero contraido dolor de cabeza'

    with open(path, encoding = 'utf-8') as f:
        data = f.read().translate(trans)

        List_lines = LineTokenizer(blanklines='keep').tokenize(data)
        
        list_lines_tokens = []
        
        for line in List_lines:
            tokenizer = RegexpTokenizer('\#[\w]+|[A-Za-zá-ú]+[\-.@][\w]+|[A-Za-z0-9]*[\d[\.\,]*\d]*\%?[A-Za-zá-ú]*|[^\w\s]|[\w]+')
            # -> \#[\w]+|[A-Za-zá-ú]+[\-.@][\w]+|[A-Za-z0-9]*[\d[\.\,]*\d]*\%?[A-Za-zá-ú]*|[^\w\s]|[\w]+
            # -> \#[\w]+|[A-Za-zá-ú]+[\-.@][A-Za-z]+|[A-Za-z0-9]*[\d[\.\,]*\d]*\%?[A-Za-zá-ú]*|[^\w\s]|[\w]+
            list_lines_tokens.append(tokenizer.tokenize(line.lower()))
            
        tokens = [ item for lines in  list_lines_tokens for item in lines if item.isalpha() ]
        
    return data, tokens


data, tokens = ehr_process('C:/Users/Acer/Desktop/umls python/100 notas/1212953') #1148025.xmi

# Autoetiquetado de la historia clinica tokenizada (formato inception)
Hc = etiquetado(tokens)

# Se validan las columnas inicales y se estructura el array final
for i in range(len(Hc)):
    if Hc[:,3][i] == 'O':
        Hc[i, 3] = Hc[i,2]
        
for j in range(len(Hc)):
    if Hc[:,3][j] == 'O':
        Hc[j, 3] = Hc[j,1]

Hc = np.delete(Hc, [1,2], 1)

# Guardo el array final en su respectiva carpeta
with open('./pred/pred_1212953.pkl', 'wb') as file:
    pickle.dump(Hc, file)
    


#%% Historias etiquetadas Jose y Dylan
with open('data_hc.pkl', 'rb') as file:
    data_hc = pickle.load(file)
    
index_hc = data_hc['1148025.xmi']
    
for i, eti in enumerate(index_hc['labels']):
    # Cambio para B-
    if eti == 'B-Anatomía':
        index_hc['labels'][i] = 'B-Anatomia'
        
    elif eti == 'B-Problema clínico':
        index_hc['labels'][i] = 'B-ProblemaClinico'
    
    elif eti == 'B-Signo o síntoma':
        index_hc['labels'][i] = 'B-SignoSintoma'
    # Cambio para I-        
    elif eti == 'I-Anatomía':
        index_hc['labels'][i] = 'I-Anatomia'
        
    elif eti == 'I-Problema clínico':
        index_hc['labels'][i] = 'I-ProblemaClinico'
    
    elif eti == 'I-Signo o síntoma':
        index_hc['labels'][i] = 'I-SignoSintoma'
        
index_hc = np.transpose(np.array([index_hc['sentence'], index_hc['labels']]))

# Esto se hace para la variable real y que los tokens sean igual a los del modelo NER
test_hc = np.array([ i for i in np.transpose([index_hc[:, 0], index_hc[:, 1]]) if i[0].isalpha() ])

with open('./test/test_1148025.pkl', 'wb') as file:
    pickle.dump(test_hc, file)


#%% Metricas de validacion

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, multilabel_confusion_matrix)


with open('./pred/pred_1148025.pkl', 'rb') as file:
    pred_hc = pickle.load(file)
    
with open('./test/test_1148025.pkl', 'rb') as file:
    test_hc = pickle.load(file)
    
# Seleccionar solo B-  
a, b = [], []
  
for i in range(len(pred_hc)):
    # Cambio para B-
    if pred_hc[i][1] == 'B-SignoSintoma':
        a.append(pred_hc[i][0])
        b.append(pred_hc[i][1])

c, d = [], []
for i in range(len(test_hc)):
    # Cambio para B-
    if test_hc[i][1] == 'B-SignoSintoma':
        c.append(test_hc[i][0])
        d.append(test_hc[i][1])




        
print(confusion_matrix(test_hc[:, 1], pred_hc[:, 1]))

print(classification_report(test_hc[:, 1], pred_hc[:, 1]))

style.use('classic')
cm = confusion_matrix(test_hc[:, 1], pred_hc[:, 1]) #, labels=['Anat', 'Atri', 'Nega', 'O', 'ProCli', 'Proce', 'SigSin', 'Sust']
disp = ConfusionMatrixDisplay(confusion_matrix = cm) #, display_labels=['Anat', 'Atri', 'Nega', 'O', 'ProCli', 'Proce', 'SigSin', 'Sust']
disp.plot()

multilabel_confusion_matrix(test_hc[:, 1], pred_hc[:, 1])











