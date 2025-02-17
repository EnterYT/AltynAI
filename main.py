import json
import numpy as np
import random
import nltk
import pickle
import telebot
from nltk.stem import WordNetLemmatizer
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')

# Load trained model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
TOKEN = '7062509186:AAEQMlxiPHY3wdkPr88tUhZVBqjLzHxF-BA'
bot = telebot.TeleBot(TOKEN)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I don't understand."

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    intent = predict_class(message.text)
    response = get_response(intent)
    bot.reply_to(message, response)

print("Bot is running...")
bot.polling()
