import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
nltk.download('stopwords')
fake_news_path = "/kaggle/input/fake-and-real-news-dataset/Fake.csv"
real_news_path = "/kaggle/input/fake-and-real-news-dataset/True.csv"
fake_news = pd.read_csv(fake_news_path)
real_news = pd.read_csv(real_news_path)
fake_news.head(3)
real_news.head(3)
real = real_news.copy()
fake = fake_news.copy()
real['Label'] = 'Real'
fake['Label'] = 'Fake'
news = pd.concat([real, fake], axis=0, ignore_index=True)
news.reset_index()
news.head()
print(f"Samples available: {news.shape[0]}\n#features of dataset: {news.shape[1]}")
news_ds = news.sample(1000).drop(['title', 'date', 'subject'], axis=1)
news_ds.head(3)
CLASS_NAMES = ['Fake', 'Real']
class_mapper = {
    'Fake':0,
    'Real':1
}
news_ds['Label'] = news_ds['Label'].map(class_mapper)
news_ds.head(3)
class_dist = px.histogram(data_frame=news,y='Label',color='Label',title='Fake vs Real news Original dataset',text_auto=True)
class_dist.update_layout(showlegend=False)
class_dist.show()
subject_dist = px.histogram(data_frame=news,
x='subject',color='subject',title='Fake vs Real news Subject Distribution',text_auto=True,facet_col='Label')
subject_dist.update_layout(showlegend=False)
subject_dist.show()
news.date.unique().max()
list(filter(lambda x:len(x)>20, news.date.unique()))
news = news[news['date'].map(lambda x:len(x)) <= 20]
news.date = pd.to_datetime(news['date'], format='mixed')
news.head()
date_dist = px.histogram(data_frame=news,x='date',color='Label')
date_dist.show()
subject_dist = px.histogram(data_frame=news,x='date',color='subject')
subject_dist.show()
real_sub_dist = px.histogram(data_frame=news[news['Label']=='Real'],x='date',color='subject')
real_sub_dist.show()
import string
stop_words = stopwords.words('english')
def text_preprocessing(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    pure_text = ' '.join(filtered_words)
    pure_text = pure_text.translate(str.maketrans('', '', string.punctuation)).strip()
    return pure_text
X = news_ds.text.apply(text_preprocessing).to_numpy()
y = news_ds.Label.to_numpy().astype('float32').reshape(-1, 1)

train_X, test_X, train_y, test_y = train_test_split(X, y,train_size=0.9,stratify=y,random_state=7)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,train_size=0.9,stratify=train_y,random_state=7)
model_name = "BERTFakeNewsDetector"
model_callbacks = ModelCheckpoint(model_name, save_best_only=True)
bert_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_name,padding='max_length',do_lower_case=True,add_special_tokens=True)
def tokenize(df):
    inputs = tokenizer(df.tolist(),padding=True,truncation=True,return_tensors='tf').input_ids
    return inputs
train_X_encoded = tokenize(train_X)
val_X_encoded = tokenize(val_X)
test_X_encoded = tokenize(test_X)
def prepare_datasets(encoded, true_df, true_target_df):
    return tf.data.Dataset.from_tensor_slices((encoded, true_target_df)).shuffle(true_df.shape[0]).batch(8).prefetch(tf.data.AUTOTUNE)
train_ds = prepare_datasets(train_X_encoded, train_X, train_y)
test_ds = prepare_datasets(test_X_encoded, test_X, test_y)
val_ds = prepare_datasets(val_X_encoded, val_X, val_y)
model = TFAutoModelForSequenceClassification.from_pretrained(bert_name,num_labels=1)
model.compile(
        optimizer = Adam(learning_rate=1e-5), metrics = [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall')
        ]
    )
model_history = model.fit(train_ds,validation_data=val_ds,callbacks=model_callbacks,epochs=5,batch_size=16)
model_history = pd.DataFrame(model_history.history)
model_history
model.save(model_name)
fig = make_subplots(rows=2, cols=2, subplot_titles=('Loss', 'Accuracy', 'Precision', 'Recall'))
fig.add_trace(go.Scatter(y=model_history['loss'], mode='lines', name='Training Loss'), row=1, col=1)
fig.add_trace(go.Scatter(y=model_history['val_loss'], mode='lines', name='Validation Loss'), row=1, col=1)
fig.add_trace(go.Scatter(y=model_history['Accuracy'], mode='lines', name='Training Accuracy'), row=1, col=2)
fig.add_trace(go.Scatter(y=model_history['val_Accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=2)
fig.add_trace(go.Scatter(y=model_history['Precision'], mode='lines', name='Training Precision'), row=2, col=1)
fig.add_trace(go.Scatter(y=model_history['val_Precision'], mode='lines', name='Validation Precision'), row=2, col=1)
fig.add_trace(go.Scatter(y=model_history['Recall'], mode='lines', name='Training Recall'), row=2, col=2)
fig.add_trace(go.Scatter(y=model_history['val_Recall'], mode='lines', name='Validation Recall'), row=2, col=2)
fig.update_layout(title='Model Training History')
fig.update_xaxes(title_text='Epoch', row=1, col=1)
fig.update_xaxes(title_text='Epoch', row=1, col=2)
fig.update_xaxes(title_text='Epoch', row=2, col=1)
fig.update_xaxes(title_text='Epoch', row=2, col=2)
fig.update_yaxes(title_text='Loss', row=1, col=1)
fig.update_yaxes(title_text='Accuracy', row=1, col=2)
fig.update_yaxes(title_text='Precision', row=2, col=1)
fig.update_yaxes(title_text='Recall', row=2, col=2)
fig.show()
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose = 0)
print(f"Test Loss      : {test_loss}")
print(f"Test Accuracy  : {test_acc}")
print(f"Test Precision : {test_precision}")
print(f"Test Recall    : {test_recall}")
def make_prediction(text, model=model):
    text = np.array([text])
    inputs = tokenize(text)
    return np.abs(np.round(model.predict(inputs, verbose=1).logits))
for _ in range(5):
    index = np.random.randint(test_X.shape[0])
    
    text = test_X[index]
    real = test_y[index]
    model_pred = make_prediction(text)
    
    print(f"Original Text:\n\n{text}\n\nTrue: {CLASS_NAMES[int(real)]}\t\tPredicted: {CLASS_NAMES[int(model_pred)]}\n{'-'*100}\n")