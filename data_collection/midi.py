import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo, second2tick
from pydub import AudioSegment
import matplotlib.pyplot as plt
from fractions import Fraction

def note_times_to_midi(note_times, bpm, sound, ticks_per_beat=480):
    tempo = bpm2tempo(bpm)

    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    curr_time = 0

    for note, time in note_times:
        tick = second2tick(time, ticks_per_beat, tempo)

        track.append(Message('note_on', note=note, velocity=64, time=tick - curr_time))
        track.append(Message('note_off', note=note, velocity=64, time=1))
        curr_time = tick + 1

    midi.save('./static/processed/output.mid')
    midi_to_audio(bpm, sound)

def midi_to_audio(bpm, sound):
    if sound == 'trap':
        snare = AudioSegment.from_wav('./static/sounds/trap/snare.wav')
        kick = AudioSegment.from_wav('./static/sounds/trap/kick.wav')
        hihat = AudioSegment.from_wav('./static/sounds/trap/hh.wav')
    elif sound == 'realistic':
        snare = AudioSegment.from_wav('./static/sounds/realistic/snare.wav')
        kick = AudioSegment.from_wav('./static/sounds/realistic/kick.wav')
        hihat = AudioSegment.from_wav('./static/sounds/realistic/hh.wav')
    elif sound == 'jazz':
        snare = AudioSegment.from_wav('./static/static/sounds/jazz/snare.wav')
        kick = AudioSegment.from_wav('./static/static/sounds/jazz/kick.wav')
        hihat = AudioSegment.from_wav('./static/static/sounds/jazz/hh.wav')

    midi_to_sound = {
        0: hihat,
        1: snare,
        2: kick
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
            output_audio = output_audio.overlay(sound, position=curr_time * 1000)
            
            times.append(curr_time)
            notes.append(msg.note)

    output_audio.export('./static/processed/midi_audio.wav', format='wav')

    note_names = {
        0: 'hihat',
        1: 'snare',
        2: 'kick'
    }

    note_labels = [note_names[note] for note in notes]

    # Convert times to beats
    beats = [str(Fraction(int(time * (bpm / 60)), 4)) for time in times]

    plt.figure(figsize=(10, 4))
    plt.scatter(beats, note_labels, color='blue', alpha=0.6, edgecolors='k')
    plt.title('MIDI Note Events in Beats')
    plt.xlabel('Beats')
    plt.ylabel('Instrument')
    plt.grid(True)
    plt.savefig('./static/processed/midi_image.png')
    plt.close()
