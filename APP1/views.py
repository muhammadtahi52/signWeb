from django.shortcuts import render


def home(request):
    return render(request, 'home.html')


def service(request):
    return render(request, 'service.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')

def signToVoice(request):
    return render(request, 'signToVoice.html')

# views.py
def signToText(request):
    return render(request, 'signToText.html')



def TextToSign(request):
    return render(request, 'TextToSign.html')

def VoiceToSign(request):
    return render(request, 'VoiceToSign.html')
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=2)
classifier = Classifier("D:/ModelTraining/pythonProject1/Model/keras_model.h5",
                        "D:/ModelTraining/pythonProject1/Model/labels.txt")

# Constants and variables
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]
predicted_sentence = ""

# Initialize GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")


def get_suggestions(current_sentence):
    """
    A function to get word suggestions based on the current sentence using GPT-2.
    """
    # Encode the input sentence and generate predictions
    inputs = tokenizer.encode(current_sentence, return_tensors='tf')
    outputs = model.generate(inputs, max_length=len(inputs[0]) + 10, num_return_sequences=1, num_beams=5,
                             early_stopping=True)

    # Decode the output predictions
    suggestions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the last predicted word
    predicted_words = suggestions.split()
    last_word = predicted_words[-1] if predicted_words else ""

    return [last_word] if last_word else []


def gen_frames():
    global predicted_sentence
    predicted_sentence = ""
    cap = cv2.VideoCapture(0)

    consecutive_counts = {label: 0 for label in labels}

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = imgSize / h
                w_cal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (w_cal, imgSize))
                w_gap = math.ceil((imgSize - w_cal) / 2)
                imgWhite[:, w_gap: w_cal + w_gap] = imgResize
            else:
                k = imgSize / w
                h_cal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, h_cal))
                h_gap = math.ceil((imgSize - h_cal) / 2)
                imgWhite[h_gap: h_cal + h_gap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < len(labels):
                consecutive_counts[labels[index]] += 1
                if consecutive_counts[labels[index]] >= 5:
                    predicted_sentence += labels[index]
                    consecutive_counts[labels[index]] = 0

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                                                             b'Content-Type: text/plain\r\n\r\n' + predicted_sentence.encode() + b'\r\n')

    cap.release()


def index(request):
    return render(request, 'signToText.html')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def fetch_predicted_sentence(request):
    global predicted_sentence
    return JsonResponse({'predicted_sentence': predicted_sentence, 'suggestions': get_suggestions(predicted_sentence)})


@csrf_exempt
def add_space(request):
    global predicted_sentence
    predicted_sentence += " "
    return JsonResponse({'predicted_sentence': predicted_sentence})


@csrf_exempt
def delete_letter(request):
    global predicted_sentence
    predicted_sentence = predicted_sentence[:-1]
    return JsonResponse({'predicted_sentence': predicted_sentence})
