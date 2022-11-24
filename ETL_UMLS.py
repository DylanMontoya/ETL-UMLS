import mysql.connector
import pandas as pd
import pickle
#%% Conectar a la base de datos
def connectdatabase():
    
    conn = mysql.connector.connect(user = 'root', 
                                   password = '', 
                                   host = 'localhost', 
                                   port = '3306', 
                                   db = 'snomed_spa')
    
    if conn:
        print('Conectado  correctamente')
        
    return conn

#%% ETL

def Extract():
    """ 
    Extraccion de todos los datos mrconso y mrsty
    """
                    
    """ Columnas de MRCONSO -> 'CUI','LAT','TS','LUI','STT','SUI','ISPREF',
                                    'AUI','SAUI','SCUI','SDUI','SAB','TTY','CODE',
                                    'STR','SRL','SUPPRESS','CVF' """
    conn = connectdatabase()

    concept = conn.cursor()
    concept.execute("SELECT * FROM mrconso")

    r1 = concept.fetchall()

    mrconso = pd.DataFrame(r1)
    mrconso.columns = ['CUI','LAT','TS','LUI','STT','SUI','ISPREF',
                       'AUI','SAUI','SCUI','SDUI','SAB','TTY','CODE',
                       'STR','SRL','SUPPRESS','CVF']

    """ Columnas de MRSTY -> 'CUI','TUI','STN','STY','ATUI','CVF'  """

    semantic_type = conn.cursor()
    semantic_type.execute("SELECT * FROM mrsty")

    r2 = semantic_type.fetchall()

    mrsty = pd.DataFrame(r2)
    mrsty.columns = ['CUI','TUI','STN','STY','ATUI','CVF']
    
    return mrconso, mrsty

#%%
def Transform(mrconso, mrsty):
    """
    De los datos obtenidos los transformamos a los requeridos para hacer las consultas
    """
    ##### Procesar las columnas necesarias
    
    new_mrconso = mrconso[['CUI', 'STR']]
    
    new_mrsty = mrsty[['CUI', 'TUI', 'STY']]

    # Preprocesamiento de texto para mrconso -> STR
    
    new_mrconso['STR'] = new_mrconso['STR'].str.lower()

    a,b = 'áéíóúüñÁÉÍÓÚÜ','aeiouunAEIOUU'
    trans = str.maketrans(a,b)

    new_mrconso['STR'] = new_mrconso['STR'].str.translate(trans)

    # Preprocesamiento de texto para mrsty -> STY
    
    new_mrsty['STY'] = new_mrsty['STY'].str.lower()
    
    return new_mrconso, new_mrsty

# Backup de las tablas procesadas
# with open('new_mrconso.pkl', 'wb') as file:
#     pickle.dump(Transform(Extract()[0], Extract()[1])[0], file)

# with open('new_mrsty.pkl', 'wb') as file:
#     pickle.dump(Transform(Extract()[0], Extract()[1])[1], file)

#%% Cargo los datos con los tipos semanticos de preferencia
def Load(new_mrconso, new_mrsty):
    """
    Organizo tipos semanticos a utilizar
    """
    
    semantic_type = ['anatomical structure', 'congenital abnormality', 'body system', 
                     'body part, organ, or organ component', 'body location or region', 
                     'body space or junction', 'body substance', 'finding', 
                     'laboratory or test result', 'injury or poisoning', 'physiologic function', 
                     'pathologic function', 'disease or syndrome', 'mental or behavioral dysfunction', 
                     'health care activity', 'laboratory procedure', 'diagnostic procedure', 
                     'therapeutic or preventive procedure', 'research activity', 'temporal concept', 
                     'qualitative concept', 'quantitative concept', 'spatial concept', 'chemical', 
                     'pharmacologic substance', 'biologically active substance', 'hazardous or poisonous substance', 
                     'substance', 'sign or symptom', 'anatomical abnormality', 'antibiotic', 'clinical drug', 
                     'organism function']

    my_sty = pd.DataFrame()

    for w in range(len(semantic_type)):
        my_sty = my_sty.append(new_mrsty[new_mrsty['STY'] == '{}'.format(semantic_type[w])])

    """
    Oganizo vocabulario a utilizar
    Uno los CUI que sean iguales de my_sty en mrconso
    """
    my_vocabulary = new_mrconso.merge(my_sty, on='CUI', suffixes=('_voc','_tip'))

    # Elimino valores duplicados en la columna STR
    my_vocabulary = my_vocabulary.drop_duplicates(subset=['STR'])
    
    print('Vocabulario creado con exito')
    
    return my_vocabulary

# Cargo tablas
# mrconso, mrsty = Extract()[0], Extract()[1]
my_vocabulary = Load(Transform(Extract()[0], Extract()[1])[0], 
                     Transform(Extract()[0], Extract()[1])[1])

# Save my_vocabulary as pickle file
with open('my_vocabulary.pkl', 'wb') as file:
    pickle.dump(my_vocabulary, file)
    
# Save my_vocabulary as csv file
my_vocabulary.to_csv('exampple.csv')
