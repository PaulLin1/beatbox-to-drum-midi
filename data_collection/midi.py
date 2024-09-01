import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo, second2tick
from pydub import AudioSegment
import matplotlib.pyplot as plt
from fractions import Fraction

def note_times_to_midi(note_times, bpm, sound, quantize, ticks_per_beat=480):
    tempo = bpm2tempo(bpm)

    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    curr_time = 0

    volumes = [i[2] for i in note_times]
    avg_volume = sum(volumes) / len(volumes)

    for note, time, volume in note_times:
        if note == 0:
            note = 42
        elif note == 1:
            note = 38
        elif note == 2:
            note = 36

        tick = second2tick(time, ticks_per_beat, tempo)

        track.append(Message('note_on', note=note, velocity=min(127, max(30, int(64 * volume / avg_volume))), time=tick - curr_time))
        track.append(Message('note_off', note=note, velocity=64, time=1))
        curr_time = tick + 1

    midi.save('./static/processed/output.mid')
    midi_to_audio(bpm, sound, quantize)

def midi_to_audio(bpm, sound, quantize):
    if sound == 'trap':
        snare = AudioSegment.from_wav('./static/sounds/trap/snare.wav')
        kick = AudioSegment.from_wav('./static/sounds/trap/kick.wav')
        hihat = AudioSegment.from_wav('./static/sounds/trap/hh.wav')
    elif sound == 'realistic':
        snare = AudioSegment.from_wav('./static/sounds/realistic/snare.wav')
        kick = AudioSegment.from_wav('./static/sounds/realistic/kick.wav')
        hihat = AudioSegment.from_wav('./static/sounds/realistic/hh.wav')
    elif sound == 'jazz':
        snare = AudioSegment.from_wav('./static/sounds/jazz/snare.wav')
        kick = AudioSegment.from_wav('./static/sounds/jazz/kick.wav')
        hihat = AudioSegment.from_wav('./static/sounds/jazz/hh.wav')

    midi_to_sound = {
        42: hihat,
        38: snare,
        36: kick
    }

    output_audio = AudioSegment.silent(duration=10000)
    midi = mido.MidiFile('./static/processed/output.mid')
    curr_time = 0
    times = []
    notes = []
    
    for msg in midi.play():
        curr_time += msg.time
        if msg.type == 'note_on' and msg.note in midi_to_sound:
            sound = midi_to_sound[msg.note]
            times.append(curr_time)
            notes.append(msg.note)

    # Quantize times if enabled
    if quantize == "on":
        quantized_times = [round((time * (bpm / 60)) * 2) / 2 * (60 / bpm) for time in times]
    else:
        quantized_times = times

    # Apply sounds based on quantized or original times
    for time, note in zip(quantized_times, notes):
        sound = midi_to_sound[note]
        output_audio = output_audio.overlay(sound, position=int(time * 1000))

    output_audio.export('./static/processed/midi_audio.wav', format='wav')

    note_names = {
        42: 'hihat',
        38: 'snare',
        36: 'kick'
    }
    note_labels = [note_names[note] for note in notes]

    # Convert times to beats
    if quantize == "on":
        beats = [round((time * (bpm / 60)) * 2) / 2 for time in times]
    else:
        beats = [time * (bpm / 60) for time in times]

    plt.figure(figsize=(10, 4))
    plt.scatter(beats, note_labels, color='blue', alpha=0.6, edgecolors='k')
    plt.title('MIDI Note Events in Beats')
    plt.xlabel('Beats')
    plt.ylabel('Instrument')
    plt.grid(True)
    # Set x-axis ticks to whole beats
    max_beat = max(beats)
    plt.xticks(range(int(max_beat) + 1))
    plt.savefig('./static/processed/midi_image.png')
    plt.close()
