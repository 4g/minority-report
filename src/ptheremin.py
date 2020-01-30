#!/bin/env python

"""A software simulation of a theremin.

A 2-dimension area serves for control of the instrument; the user drags the mouse on this area to control the frequency and amplitude.  Several modes are provided that adds virtual "frets" to allow for the playing of the equal tempered tuning (or subsets of it).

For better musical sound run your sound card into a guitar amp or similar.

Requires Python 2.3+ and PyGTK 2.4+ (not tested on anything older).

http://ptheremin.sourceforge.net
"""

import array
import math
import struct
import threading
import time
import wave

import pyaudio
import serial

import gi

SCALES = ("chromatic", "diatonic major", "pentatonic major", "pentatonic minor", "blues")
INIT_FREQ = 20

NAME="PTheremin"
VERSION="0.2.1"


# from "Musical Instrument Design" by Bart Hopkin
sharp = 1.05946
equal_temp_freqs = [16.352, 16.352*sharp, 18.354, 18.354*sharp, 20.602, 21.827, 21.827*sharp, 24.500, 24.500*sharp, 27.500, 27.500*sharp, 30.868]
equal_temp_labels = ['C*', 'C*#', 'D*', 'D*#', 'E*', 'F*', 'F*#', 'G*', 'G*#', 'A*', 'A*#', 'B*']

equal_temp_tuning = zip(equal_temp_labels, equal_temp_freqs)

diatonic_major_intervals = (0, 2, 4, 5, 7, 9, 11)
pentatonic_major_intervals = (0, 2, 4, 7, 9)
pentatonic_minor_intervals = (0, 3, 5, 7, 10)
blues_intervals = (0, 3, 5, 6, 7, 10)

# build up several octaves of notes
NOTES = []
for octave in range(11):
    for label,freq in equal_temp_tuning:
        NOTES.append((label.replace('*', "%d" % octave), (2**octave)*freq))

def just_freqs(notes):
    return [freq for label,freq in notes]

class PlaybackThread(threading.Thread):
    """A thread that manages audio playback."""

    def __init__(self, name):

        threading.Thread.__init__(self, name=name)
        self.name = name

        #####################################################
        # Constants that come from the pyAudio sample scripts
        # Short int audio format
        self.WIDTH=2
        self.CHANNELS = 2
        self.RATE = 44100
        # Signed short range
        self.maxVolume = 10000

        self.fs = 44100 # the sample frequency
        self.ft = INIT_FREQ # the base frequency of the instrument
        self.vol = 1
        ######################################################

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.p.get_format_from_width(self.WIDTH),
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  output=True)

        self.paused = False
        self.alive = True
        self.recording = array.array('h') # *way* faster than a list for data access


    def run(self):
        def tone_gen(fs):
            """A tone sample generator."""
            x = 0
            pi = math.pi
            sin = math.sin
            ft = self.ft
            sample = 0
            prev_sample = 0
            while 1:
                prev_sample = sample
                sample = sin(2*pi*ft*x/fs)

                # The idea here is to keep the waveform continuous by only changing
                # the frequency at the end of the previous frequency's period.  And
                # it works!
                if ft != self.ft and 0.01 > sample > -0.01 and prev_sample < sample:
                    ft = self.ft
                    x = 0

                x += 1
                yield sample*self.vol*0.95 # don't max out the range otherwise we clip


        # to optimize loop performance, dereference everything ahead of time
        tone = tone_gen(self.fs)
        pack_func = struct.pack

        record_func = self.recording.append

        while self.alive:
            wave = ""
            if not self.paused:
                clean = next(tone)
                val_f = clean
                val_i = int(val_f*(2**15 - 1))

                sample = pack_func("h", val_i)
                for c in range(self.CHANNELS):
                    # write one sample to each channel
                    self.stream.write(sample,1)
                record_func(val_i)
            else:
                time.sleep(0.1)


    def stop(self):
        print (" [*] Stopping toner...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print(" [*] Toner Done.")
        self.alive = False

    def set_new_freq(self, freq, vol):
        """Updates the input frequency."""
        self.ft = freq
        self.vol = vol


    def get_wav_data(self):
        return self.recording


    def clear_wav_data(self):
        self.recording = []

def iir_2pole(coeff1, coeff2):
    """A two-pole IIR filter generator from that one guy's filter design page that I always use."""

    xv = [0, 0, 0]
    yv = [0, 0, 0]

    def iir(sample):
        while 1:
            xv[0] = xv[1]
            xv[1] = xv[2]
            xv[2] = sample
            yv[0] = yv[1]
            yv[1] = yv[2]
            yv[2] = xv[0] + xv[2] - 2*xv[1] + coeff1*yv[0] + coeff2*yv[1]
            yield yv[2]


def discrete_tones(tones):
    """Makes a discrete-tone filter that latches to particular tones."""

    def filt(x):
        closest = tones[0]
        err = 500000
        mean = 0

        iir = iir_2pole(-.9979871157, 1.997850878)

        for i,tone in enumerate(tones):
            tone_err = abs(x - tone)
            if tone_err < err:
                closest = tone
                err = tone_err
            elif tone_err > err:
                if i > 0:
                    mean = (x - closest)/2

                break

        return closest + mean

    return filt


class ThereminApp(object):
    """The GUI part of the theremin."""
    def new_tone_filter(self):
        self.root_notes = [n for i,n in enumerate(self.shifted_notes) if i % 12 == 0]

        if self.scale == 'chromatic':
            key_notes = NOTES

        elif self.scale == 'diatonic major':
            key_notes = [n for i,n in enumerate(self.shifted_notes) if i % 12 in diatonic_major_intervals]

        elif self.scale == 'pentatonic major':
            key_notes = [n for i,n in enumerate(self.shifted_notes) if i % 12 in pentatonic_major_intervals]

        elif self.scale == 'pentatonic minor':
            key_notes = [n for i,n in enumerate(self.shifted_notes) if i % 12 in pentatonic_minor_intervals]

        elif self.scale == 'blues':
            key_notes = [n for i,n in enumerate(self.shifted_notes) if i % 12 in blues_intervals]

        self.tone_filter = discrete_tones(just_freqs(key_notes))
        self.discrete_notes = key_notes

    def scale_changed(self, button, scale_name):
        if button.get_active():
            self.scale = scale_name
            self.new_tone_filter()


    def mode_changed(self, button, mode):
        if button.get_active():
            self.mode = mode
            self.new_tone_filter()


    def key_changed(self, button, key):
        self.key = key.get_active_text()

        self.shifted_notes = list(NOTES)
        shifts = {
            'A': 9,
            'A#': 10,
            'B': 11,
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
        }

        for i in range(shifts[self.key]):
            self.shifted_notes.append(self.shifted_notes.pop(0))

        self.new_tone_filter()


    def master_volume_changed(self, slider):
        self.master_volume = math.log10(slider.get_value())
        self.set_tone(self.freq, self.vol)


    def set_tone(self, freq, vol):
        self.freq = freq
        self.vol = vol

        if self.mode == 'discrete':
            closest = self.tone_filter(freq)
        else:
            closest = freq

        self.threads['playback'].set_new_freq(closest, vol*self.master_volume)


    def pause(self, button):
        if button.get_active():
            self.threads['playback'].paused = False
        else:
            self.threads['playback'].paused = True


    def __init__(self):

        self.threads = {}

        self.threads['playback'] = PlaybackThread("playback")

        self.freq = INIT_FREQ
        self.freq = 0
        self.freq_max = 2000
        self.freq_min = 20

        self.mode = 'continuous'
        self.scale = 'chromatic'
        self.key = 'C'
        self.shifted_notes = NOTES
        self.discrete_notes = NOTES
        self.root_notes = [x for i,x in enumerate(NOTES) if i % 12 == 0]
        self.master_volume = math.log10(7.2)
        self.vol = 10

        self.tone_filter = discrete_tones(just_freqs(NOTES))

        for thread in self.threads.values():
            thread.start()


def main():
    import getopt
    import sys

    app = ThereminApp()



if __name__ == '__main__': main()
