import sqlite3
import tensorflow as tf
import numpy as np
import sys
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


def add_random_col(colname, numRandom):
    c=conn.cursor()
    try:
        c.execute('ALTER TABLE loan ADD COLUMN '+colname+' Integer')
    except:   
        print(sys.exc_info())
        pass # handle the error
    c.execute("UPDATE loan SET "+colname+" =0")
    c.execute("UPDATE loan SET "+colname+" = 1 WHERE id IN (SELECT id FROM loan ORDER BY RANDOM() LIMIT ?)", [(numRandom)])

def create_sql_train(colname):
    c=conn.cursor()
    c.execute("""
    SELECT funded_amnt,  
    replace(CASE WHEN int_rate IS NULL THEN '10%' ELSE int_rate END, '%', '')*.01 as int_rate,CASE WHEN annual_inc is null then 0 else annual_inc END as annual_inc,  
    CASE WHEN dti is NULL then .5 ELSE dti end as dti, 
    CASE WHEN loan_status in 
        ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') 
        THEN 1 ELSE 0 
    END as DidDefault FROM loan WHERE """+colname+""" = 1
    """)
    return c

def create_sql_test(colname, numSample):
    c=conn.cursor()
    c.execute("""
    SELECT funded_amnt,  
    replace(CASE WHEN int_rate IS NULL THEN '10%' ELSE int_rate END, '%', '')*.01 as int_rate,CASE WHEN annual_inc is null then 0 else annual_inc END as annual_inc,  
    CASE WHEN dti is NULL then .5 ELSE dti end as dti, 
    CASE WHEN loan_status in 
        ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') 
        THEN 1 ELSE 0 
    END as DidDefault FROM loan WHERE """+colname+""" = 0
    AND id IN (SELECT id FROM loan ORDER BY RANDOM() LIMIT ?)
    """, [(numTrain)])
    return c



batchSize=1000
numTrain=100000
numTest=200000

colName="TrainSet"
add_random_col(colName,  numTrain)

#create_sql_test(colName, numTest, testCursor)
def stepsPer(numSamples, batchSize):
    return numSamples/batchSize

def sql_generator_train(batchSize):
    while True:
        c=create_sql_train(colName)
        while True:
            rawData=c.fetchmany(batchSize)
            if len(rawData)==0:
                break
            modelData=np.array(rawData)
            yield modelData[:, 0:4], modelData[:, 4] #tuple

def sql_generator_test(sampleSize, batchSize):
    while True:
        c=create_sql_test(colName, sampleSize)
        while True:
            rawData=c.fetchmany(batchSize)
            if len(rawData)==0:
                break
            modelData=np.array(rawData)
            yield modelData[:, 0:4], modelData[:, 4] #tuple

##One epoch is a single pass over all the data...
## step per epoch means how many times the optimization is run 
## in this case, the optimization is run 100 times (once over each batch of 1000 samples)

fitHistory=model.fit_generator( generator=sql_generator_train(batchSize), steps_per_epoch=10, validation_data=sql_generator_test(numTest, batchSize), validation_steps=20, epochs=9)
print(model.predict(np.array([[5000, .1, 10000, .2]])))
print(model.predict(np.array([[5000, .1, 10000, .8]])))
print(model.predict(np.array([[5000, .15, 50000, .2]])))



#evaluation=model.evaluate_generator(generator=sql_generator_test(), steps=stepsPer(numTest, arraySize))

#conn.close()

#print(evaluation)