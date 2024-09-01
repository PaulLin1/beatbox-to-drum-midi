from flask import Flask, render_template, redirect, url_for, request, send_file

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
    print(quantize, bpm, sound)
    
    file_path = f"uploads/audio.wav"

    processed = process_full_loop(file_path)

    model = SpectrogramCNN()
    model.load_state_dict(torch.load('model/model.pth'))

    note_times = []

    for i in processed:
        time, spectrogram = i
        volume = spectrogram.mean()
        outputs = model(spectrogram)
        _, predicted = torch.max(outputs, 1)
        note_times.append((predicted[0].item(), time, volume))

    note_times_to_midi(note_times, bpm, sound, quantize)

    return render_template('index.html',
                           image_url='./static/processed/waveform_image.png',
                           midi_img_url='./static/processed/midi_image.png',
                           audio_url='./static/processed/midi_audio.wav'
                           )    

@app.route('/download-midi', methods=['GET', 'POST'])
def download():
    file_path = './static/processed/output.mid'
    
    # Send the file to the client
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)