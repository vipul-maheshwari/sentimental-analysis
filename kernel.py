import os
import pickle
import logging
import pkg_resources
import time
script_start_time = time.time()
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.stem import PorterStemmer
import re
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from transformers import TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Create a 'logs' folder if it doesn't exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a FileHandler for the logger
log_file = os.path.join(log_folder, 'log.txt')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Reading the dataset, renaming the column and replacing the sentiment values to 1 and 0
df = pd.read_csv('IMDB Dataset.csv')
df.rename(columns={'sentiment':'label'},inplace=True)
df.label.replace({'positive' : 1, 'negative' : 0}, inplace=True)
df.info()

# Plotting the distribution of Sentiment
plt.figure(figsize=[15,7])
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
sns.countplot(data=df, x='label')

# Function to preprocess the text data
def data_preprocessing(text):
    
    text = text.lower() # Lowercase
    text = re.sub('<br />', '', text) # Remove the html tags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # Remove the http and www url links
    text = re.sub(r'\@\w+|\#\w+', '', text) # Remove the special characters like @, # etc. from the text
    text = re.sub(r'[^\w\s]', '', text) # Remove the punctuation
    tokenized_text = word_tokenize(text) # Tokenize the words
    # Remove the stop words
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in tokenized_text if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    # Return the filtered text
    return " ".join(filtered_text)


start_time = time.time() # Log the start of the data preprocessing
df['review'] = df['review'].map(data_preprocessing) # Preprocessing the data using the function and applying the map to process the multiple rows of data at one time.
elapsed_time = time.time()-start_time # Calculating the time taken to preprocess the data
logger = logging.getLogger(__name__) # Log the information with the elapsed time
logger.info('Data Preprocessing Completed. Elapsed Time: {:.2f} seconds'.format(elapsed_time))

# Training the model (Creating a Pipeline which includes TfidfTransformer and SVC)
# Creating the pipeline
user_created_pipeline = Pipeline([('Vect',TfidfVectorizer()), ('model',SVC(probability=True))])

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.3, random_state=0)

# Printing the shape of the train and test data
print("Shape of the train data: ", x_train.shape)
print("Shape of the test data: ", x_test.shape)


# Start time for the model training
start_time = time.time()
user_created_pipeline.fit(x_train, y_train) # Training the model
elapsed_time = time.time()-start_time # Calculating the time taken to train the model
logger = logging.getLogger(__name__) # Log the information with the elapsed time
logger.info('Model Training Completed. Elapsed Time: {:.2f} seconds'.format(elapsed_time))

# Predicting the sentiment of the test data
y_pred = user_created_pipeline.predict(x_test)

# Printing the analysis of the model
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)
print(classification_report(y_pred,y_test))

# Saving the model
pickle.dump(user_created_pipeline,open('sentiment_analysis_model.p','wb'))

# Log the total time taken for the script to run
logger = logging.getLogger(__name__)
logger.info('Script Completed. Total Time: {:.2f} seconds'.format(time.time()-script_start_time))