from flask import Flask, render_template, redirect, url_for, request

from data_collection.process import *
from data_collection.midi import *
from main import SpectrogramCNN

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        file_path = 'uploads/audio.wav'
        file.save(file_path)
        waveform_to_png(file_path)

        return render_template('index.html', image_url='./static/processed/waveform_image.png')    

@app.route('/process', methods=['GET', 'POST'])
def process():
    quantize = request.form.get('quantize', 'off')
    bpm = int(request.form.get('bpm'))
    sound = request.form.get('sound')
    print(quantize, bpm)
    file_path = f"uploads/audio.wav"

    processed = process_full_loop(file_path)

    model = SpectrogramCNN()
    model.load_state_dict(torch.load('model/model.pth'))

    note_times = []

    for index, i in enumerate(processed):
        time, spectrogram = i
        outputs = model(spectrogram)
        _, predicted = torch.max(outputs, 1)
        note_times.append((int(predicted[0].item()), time))

    note_times_to_midi(note_times, bpm, sound)

    return render_template('index.html',
                           image_url='./static/processed/waveform_image.png',
                           midi_img_url='./static/processed/midi_image.png',
                           audio_url='./static/processed/midi_audio.wav'
                           )    


if __name__ == '__main__':
    app.run(debug=True)