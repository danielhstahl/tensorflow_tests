import sqlite3
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
#4 specifies the "width" of the data set (how many explanatory variables)
model.add(Dense(units=64, input_shape(4, )))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


##model.fit(x_train, y_train, epochs=5, batch_size=32)
model.train_on_batch(x_batch, y_batch)


conn = sqlite3.connect('database.sqlite')

c = conn.cursor()

#number, string, number, string, string, number, string, string, number
for row in c.execute("SELECT funded_amnt, term, replace(int_rate, '%', '')*.01 as int_rate, emp_length, home_ownership, annual_inc, CASE WHEN loan_status in ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END as DidDefault, purpose, dti FROM loan"):
    print(row)


# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)