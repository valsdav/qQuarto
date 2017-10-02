import telegram
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

training_name = os.environ["NAME"]
token = os.environ["TOKEN"]
channel = os.environ["CHANNEL"]

bot = telegram.Bot(token=token)

def send_text(text):
    bot.send_message(chat_id=channel, text=training_name +"@ " +text)

def send_file(path):
    bot.send_document(chat_id=channel, document=open(path, "rb"))

def send_image(path):
    bot.sendPhoto(chat_id=channel, photo=open(path, "rb"))

def send_graph(losses):
    i = 100
    data = np.array(losses)
    data1 = data[int(i*0.01)-1:]
    index = list(range(i, (len(data)+1)*100,100))

    f, (a1,a2) = plt.subplots(2,1,figsize=(20, 20))
    a1.plot(index, data1)
    interval = 50
    m = []
    for i in range(0, len(data1), interval):
        m.append(np.mean(data1[i:i+interval]))
    a2.plot(range(len(m)),m)

    plt.savefig("/tmp/losses.png")
    send_image("/tmp/losses.png")

def send_details(rewards, losses, pieces):
    f, xarr = plt.subplots(2,2,figsize=(15, 10))
    xarr[0,0].scatter(rewards, losses, c=range(len(pieces)))
    xarr[0,1].scatter(rewards, pieces,c=range(len(pieces)))
    xarr[1,0].scatter(pieces, losses,c=rewards)
    xarr[1,1].scatter(pieces, losses,c=range(len(rewards)))
    plt.savefig("/tmp/details.png")
    send_image("/tmp/details.png")
