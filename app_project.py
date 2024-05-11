from tkinter import *
from gtts import gTTS

import playsound
import os
import random
from chat_progect2 import get_response,bot_name
import speech_recognition as sr
import threading
import pyttsx3 as pp



BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvatica 16"
FONT_BOLD = "Helvetica 14 bold"



r= sr.Recognizer()

#def botspeak(word):
#    engine.say(word)
#   engine.runAndWait()

def Bot_speak(audio_string):
    tts= gTTS(text=audio_string , lang="en")
    R=random.randint(1,10000)
    audio_file= "audio-" + str(R) + ".mp3"
    tts.save(audio_file)
    playsound.playsound(audio_file)
    print(audio_string)
    os.remove(audio_file)



def record_audio(ask=False):
    with sr.Microphone() as source:
        audio = r.listen(source)
        if ask:
            Bot_speak(ask)
        msg = ""
        try:
            msg = r.recognize_google(audio)
        except sr.UnknownValueError:
            Bot_speak("sorry i did not understand that")
        except sr.RequestError:
            Bot_speak("sorry, speech recognition not working")
        return msg









class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()


    def run(self):
        self.window.mainloop()



    def _setup_main_window(self):
        self.window.title("School Bot")
        self.window.resizable(width=True,height=True)
        self.window.configure(width=500,height=550, bg=BG_COLOR)

        #head label
        head_label = Label(self.window,bg = BG_COLOR,fg = TEXT_COLOR,font = FONT_BOLD,pady = 10,text= "Welcome")
        head_label.place(relwidth = 1)

        #tiny divider
        line = Label(self.window,width = 470,bg = BG_GRAY )
        line.place(relwidth = 1,rely= 0.7 ,relheight = 0.012)

        #text widget
        self.text_widget = Text(self.window,width = 20,height =2,bg = BG_COLOR,fg = TEXT_COLOR,font = FONT,padx = 5,pady= 5)
        self.text_widget.place(relheight = 0.745,relwidth =1 ,rely= 0.08)
        self.text_widget.configure(cursor = "arrow",state = DISABLED)

        #SCROOLBAR
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight =1,relx = 0.985)
        scrollbar.configure(command = self.text_widget.yview)

        #bottom label widget
        bottom_label = Label(self.window,bg =BG_GRAY,height =80)
        bottom_label.place(relwidth = 1,rely = 0.825)

        #message entry box
        self.msg_entry = Entry(bottom_label,bg= "#2C3E50",fg = TEXT_COLOR,font = FONT)
        self.msg_entry.place(relwidth = 0.74, relheight = 0.06, rely = 0.008,relx = 0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self._on_enter_pressed)

        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             activebackground="ghost white",
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely = 0.008, relheight=0.06, relwidth=0.22)

    def takequery(self):
        r = sr.Recognizer()
        r.pause_threshold = 1
        print("bot is listening try to speak")
        with sr.Microphone() as m:
            audio = r.listen(m)
            msg = r.recognize_google(audio)
            print(msg)
            self.msg_entry.insert(0,END)
            get_response()


    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg,"You")

    def _insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0,END)
        msg1 = f"{sender}:{msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,msg1)
        self.text_widget.configure(state=NORMAL)

        msg2 = f"{bot_name}:{get_response(msg)}\n\n"
        msg22 = str(msg2)
        msg22 = msg22[4:100]
        #botspeak(msg2)


        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)



        self.text_widget.see(END)
        Bot_speak(msg22)




















Bot_speak("how can i help you")



if __name__=="__main__":
    app = ChatApplication()
    app.run()


