from threading import Thread
import time
from tqdm import tqdm


def progress_bar():
    for i in tqdm([1,2,3,4,5,6,7]):
        time.sleep(1)

thread = Thread(target=progress_bar)
thread.start()


'''                              ИМПОРТ                                     '''
###############################################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

from datetime import datetime as dt # работа со временем
import json                     # преобразования строк со словарями в словари
import seaborn as sns




from pathlib import Path 

train = pd.read_pickle( Path("data", "all_auto_ru_09_09_2020.pickle_bz2"),
                compression='bz2')
test = pd.read_pickle( Path("data", "test.pickle_bz2"),
                compression='bz2')







'''            ОТСЕВ ЛИШНИХ КОЛОНОК НЕОБРАБОТАННЫХ ДАТАСЕТОВ                '''
###############################################################################

'''Эта ячейка объединяет все нужное и выкидывает все лишнее'''
test = test.drop_duplicates()
train = train.drop_duplicates()
train['sell_id'] = -1 # нужно, чтобы не потерять колонку для предсказаний



train_df = train.drop(['start_date',
                       'vehicleConfiguration',
                       'Состояние', 
                       'Таможня', 
                       'hidden',
                       'Комплектация'],axis='columns')

test_df = test.drop(['parsing_unixtime',
                     'vehicleConfiguration',
                     'complectation_dict',
                     'image',
                     'model_info',
                     'priceCurrency',
                     'Состояние',
                     'Таможня',
                     'equipment_dict',
                     'super_gen',
                     'vendor',
                     'car_url'],axis='columns')

'''engineDisplacement особенный и обработается тут'''
##############################################################################
#Функция для приведения значений столбца engineDisplacement к единому формату
def get_EngineDisplacement(x):
    row = str(x)
    engine = re.findall('\d\.\d', row)
    if engine == []:
        return None
    return float(engine[0])



#Функция для извлечения значения объёма двигателя из столбца name
def pull_EngineDisplacement(x):
    row = str(x)
    engine = re.findall('\d\.\d', row)
    if len(engine) !=0:
        return float(engine[0])
    else:
        return 0
    
    
#Приведение столбца 'engineDisplacement' к единому формату 
train_df['engineDisplacement'] = train_df.engineDisplacement.apply(get_EngineDisplacement)
test_df['engineDisplacement'] = test_df.engineDisplacement.apply(get_EngineDisplacement)

#Заполнение пропусков в столбце 'engineDisplacement' датасета test_mod нолями


#Извлечение значений объёма двигателя из столбца 'name' для заполнения пропусков в столбце 'engineDisplacement'
#в датасете train_mod
train_df['engineDisplacement'] = train_df['name'].apply(pull_EngineDisplacement)
###############################################################################





train_df.rename(columns = {'model' : 'model_name'}, inplace = True)

test_df['price'] = -1 # это колонка, по которой можно различить два датасета 
df = pd.concat([test_df,train_df],ignore_index=True)


'''                          ОБРАБОТКА                                      '''
###############################################################################
### body_Type ###
def get_bodyType(x):
    row = str(x).lower()
    bodyType = re.findall('[а-яё]+', row)
    if bodyType == []:
        return None
    return str(bodyType[0])

df['bodyType'] = df.bodyType.apply(get_bodyType)

### color ###
df.replace({'color':
                {'040001': 'чёрный',
                'EE1D19': 'красный',
                '0000CC': 'синий',
                'CACECB': 'серебристый',
                '007F00': 'зелёный',
                'FAFBFB': 'белый',
                '97948F': 'серый',
                '22A0F8': 'голубой',
                '660099': 'фиолетовый',
                '200204': 'коричневый',
                'C49648': 'бежевый',
                'DEA522': 'золотистый',
                '4A2197': 'пурпурный',
                'FFD600': 'жёлтый',
                'FF8649': 'оранжевый',
                'FFC0CB': 'розовый'}},inplace=True)


### numberOfDoors ###
'''совру модели, что у старых авто были двери'''
df.at[16944 , 'numberOfDoors'] = 2 
df.at[120656, 'numberOfDoors'] = 2 

### fuelType ###
df = df.drop([58731])
df.reset_index(drop=True, inplace=True)

### modelDate ###
'''и так хороша'''


### productionDate ###
'''и так хороша'''

### fuelType ###
df.replace({'vehicleTransmission':
                {'автоматическая': 'AUTOMATIC',
                'вариатор': 'VARIATOR',
                'механическая': 'MECHANICAL',
                'роботизированная': 'ROBOT'}},inplace=True)

### EngineDisplacement ###
df['engineDisplacement'].fillna(0, inplace = True)


### enginePower ###
df.enginePower = pd.concat(
    [ df[df.price==-1].enginePower.apply(lambda x : (int)(x.split(' ')[0])),
      df[df.price!=-1].enginePower.convert_dtypes(int) ],  ignore_index=True)

### Владельцы ###
df.replace({'Владельцы':
                {'1\xa0владелец': 1,
                '2\xa0владельца': 2,
                '3 или более'   : 3}},inplace=True)
df.Владельцы.fillna(0,inplace=True)

### Руль ###
df.replace({'Руль':
                {'Левый': 'LEFT',
                'Правый': 'RIGHT'}},inplace=True)

### ПТС ###
df.replace({'ПТС':
                {'Оригинал': 'ORIGINAL',
                'Дубликат': 'DUPLICATE'}},inplace=True)
df.ПТС.fillna('ORIGINAL',inplace=True)

### Владение ###
bin_own = []
for i in df.Владение.isna():
    if i==True: bin_own.append(0)
    else: bin_own.append(1)
        
df.Владение = pd.Series(bin_own)

### price ###
df.drop(df[df.price.isna()].index, axis='rows',inplace=True)

### Использованное ###
df.drop('name',axis='columns',inplace=True)


######## Предобработка для модели
df.replace({'bodyType':
                {'кабриолет':'экстра',
'родстер':'экстра'  ,
'фургон' :'экстра'  ,
'микровэн':'экстра' ,
'лимузин':'экстра'  ,
'тарга'  :'экстра' , 
'фастбек':'экстра'}},inplace=True)

df.replace({'color':
                {
'бежевый':'редкий'   , 
'голубой':'редкий'   , 
'золотистый':'редкий' ,
'фиолетовый':'редкий', 
'жёлтый':'редкий'    , 
'пурпурный':'редкий' , 
'оранжевый':'редкий' , 
'розовый':'редкий' }},inplace=True)

df['model_product_time'] = df.productionDate - df.modelDate

df['car_age'] = 2020.8 - df.productionDate 
df.drop(['modelDate','productionDate'],axis='columns',inplace=True)