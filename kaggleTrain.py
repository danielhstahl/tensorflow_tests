import sqlite3
import tensorflow as tf
import numpy as np
#pip install keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.layers.normalization import BatchNormalization

##This program uses the lending club loan data https://www.kaggle.com/wendykan/lending-club-loan-data


def create_model(unitsPerHidden, inputDim):
    model = Sequential()
    #4 specifies the "width" of the data set (how many explanatory variables)
    #10 is the number of  nodes
    model.add(Dense(units=unitsPerHidden, input_dim=inputDim))
    #scales ths input
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5)) ##this can be used to avoid overfitting
    model.add(Dense(units=unitsPerHidden))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5)) ##this can be used to avoid overfitting
    #binary
    model.add(Dense(1, activation='sigmoid'))

    #can add custom metrics...for a confusion matrix, for example
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

model=create_model(10, 4)

conn = sqlite3.connect('database.sqlite', check_same_thread=False)

trainCursor = conn.cursor()
testCursor=conn.cursor()
arraySize=1000
numTrain=100000
numTest=200000
def create_sql(numSample, c):
    c.execute("""
    SELECT funded_amnt,  
    replace(CASE WHEN int_rate IS NULL THEN '10%' ELSE int_rate END, '%', '')*.01 as int_rate,CASE WHEN annual_inc is null then 0 else annual_inc END as annual_inc,  
    CASE WHEN dti is NULL then .5 ELSE dti end as dti, 
    CASE WHEN loan_status in 
        ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') 
        THEN 1 ELSE 0 
    END as DidDefault FROM loan WHERE 
    id IN (SELECT id FROM loan ORDER BY RANDOM() LIMIT ?)""", [(numSample)])

create_sql(numTrain, trainCursor)
create_sql(numTest, testCursor)

def sql_generator(c, arraysize):
    while True:
        rawData=c.fetchmany(arraysize)
        modelData=np.array(rawData)
        yield modelData[:, 0:4], modelData[:, 4] #tuple

def sql_generator_train():
    return sql_generator(trainCursor, arraySize)

def sql_generator_test():
    return sql_generator(testCursor, arraySize)

def stepsPer(numSamples, arraysize):
    return numSamples/arraysize


''' gen=sql_generator_train()
totalNum=0
for i in range(0, stepsPer(numTrain, arraySize)):
    results=gen.next()
    totalNum+=len(results[0])
    print(len(results[0]))
    
print(totalNum) '''

fitHistory=model.fit_generator( generator=sql_generator_train(), steps_per_epoch=10, validation_data=sql_generator_test(), validation_steps=20, epochs=10)

print(model.predict([5000, .1, 10000, .2]))
#evaluation=model.evaluate_generator(generator=sql_generator_test(), steps=stepsPer(numTest, arraySize))

conn.close()

#print(evaluation)