import streamlit as st
from streamlit_plotly_events import plotly_events
from keras.datasets import fashion_mnist
from tensorflow.keras.saving import load_model
from sklearn.metrics import confusion_matrix
from random import randint
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


# Diccionarios

model = {'Base': 1, # Aquí declarar los modelos
         'Dropout': 2,
         'Batch Normalization': 3,
         'Dropout + Batch Normalization': 4}

labels = {0: 'T-shirt/top',
          1: 'Trouser', 
          2: 'Pullover', 
          3: 'Dress', 
          4: 'Coat',
          5: 'Sandal',
          6: 'Shirt', 
          7: 'Sneaker',
          8: 'Bag',
          9: 'Ankle boot'}

# Funciones

def get_test_data():
    
    data = fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    x_test = x_test / 255.0

    return x_test, y_test

# Página

st.set_page_config(
    page_title='Fashion MNIST CNN',
    layout='wide',
)

st.title('Fashion MNIST Convolutional Neural Network')

x, y = get_test_data()

select_model = st.selectbox('Modelo', ('Base', 'Dropout', 'Batch Normalization', 
                             'Dropout + Batch Normalization'))
history = pd.read_csv(f'History/history_{model[select_model]}.csv')
fig1 = px.line(history, labels={'index':'Epoch'}, color_discrete_sequence=px.colors.qualitative.G10)
plotly_events(fig1)

model = load_model(f'Modelos/best_model_{model[select_model]}.h5')
score = model.evaluate(x, y)
st.write(f'Test Loss: {round(score[0]*100,2)}%')
st.write(f'Test Accuracy: {round(score[1]*100,2)}%')

df = pd.DataFrame(model.predict(x))
df_modified = df.apply(lambda row: (row == row.max()).astype(int), axis=1)
y_pred = df_modified.apply(lambda row: row.idxmax(), axis=1).values

conf_matrix = confusion_matrix(y, y_pred, labels=range(10))
fig2 = px.imshow(conf_matrix, labels={'x':'Predicted', 'y':'True'}, color_continuous_scale="Reds")
plotly_events(fig2)

if st.button('Prueba'):

    i = randint(0, len(y))

    st.image(x[i], width= 200)
    st.write(f'Prediccion: {y_pred[i]}. {labels[y_pred[i]]}')
    st.write(f'Real: {y[i]}. {labels[y[i]]}')
