import random
import json
import webbrowser
import numpy as np
import matplotlib.pyplot as m
import speech_recognition as sr
import torch
import threading
import time
from model_project import NeuralNet
from nltk_utils_project import bag_of_words, tokenize

def graph():
    year = [100, 100, 100, 100, 100]
    c12 = [100, 100, 100, 100, 100]
    c10 = [100, 100, 100, 100, 100]
    x = np.arange(len(year))

    m.bar(x, c12, width=0.2, label='class 12', edgecolor="b")
    m.bar(x + 0.2, c10, width=0.2, label='class 10', edgecolor="b")
    m.title("past 5 years board results")
    m.xlabel("class")
    m.ylabel("percentage")
    m.legend()
    m.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents_project.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "bot"


r = sr.Recognizer()





def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(tag)
                x=tag
                if x == "fees":
                    url = "https://en.wikipedia.org/w"
                    webbrowser.get().open(url)
                if x == "form":
                    url = "file:///C:/Users/LENOVO/OneDrive/Desktop/form2.html"
                    webbrowser.get().open(url)
                if x== "graph":
                    thr = threading.Thread(target=graph)
                    thr.start()




                return random.choice(intent['responses'])





    return "I do not understand..."




print(threading.active_count())