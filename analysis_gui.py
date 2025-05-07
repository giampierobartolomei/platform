# -*- coding: utf-8 -*-
"""
Minimal Video Sync GUI with Animated CSV & Audio Plots

Features:
- Load and sync two videos side by side
- Unified Play/Pause button
- Slider with start/end time labels
- Animated frame-by-frame plots for Echo "Result", Gaze, and Audio intensity
"""
import sys
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import downsampling
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
plt.interactive(False)


class VideoSyncApp(QWidget):
    FPS = 20  # frames per second for CSV data

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Platform Environment 1.0")
        self.resize(950, 700)
        self._build_ui()
        self._init_state()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # --- Video row ---
        self.player1 = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.player2 = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        vid1, vid2 = QVideoWidget(), QVideoWidget()
        self.player1.setVideoOutput(vid1)
        self.player2.setVideoOutput(vid2)
        row = QHBoxLayout()
        row.addWidget(vid1); row.addWidget(vid2)
        vid1.setFixedSize(400,300); vid2.setFixedSize(400,300)
        main.addLayout(row)

        # --- Controls ---
        ctrl = QHBoxLayout()
        btn1 = QPushButton("Load Video 1"); btn1.clicked.connect(self.load_video1)
        btn2 = QPushButton("Load Video 2"); btn2.clicked.connect(self.load_video2)
        self.btn_play = QPushButton("Play"); self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl.addWidget(btn1); ctrl.addWidget(btn2); ctrl.addWidget(self.btn_play)
        main.addLayout(ctrl)

        # --- Slider with time labels ---
        sl = QHBoxLayout()
        self.lbl_start = QLabel("0.0s")
        self.slider    = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek)
        self.lbl_end   = QLabel("0.0s")
        sl.addWidget(self.lbl_start); sl.addWidget(self.slider); sl.addWidget(self.lbl_end)
        main.addLayout(sl)

        # --- CSV loaders ---
        csv = QHBoxLayout()
        self.btn_echo = QPushButton("Load Echo CSV"); self.btn_echo.clicked.connect(self.load_echo)
        self.btn_look = QPushButton("Load Look CSV"); self.btn_look.clicked.connect(self.load_look)
        csv.addWidget(self.btn_echo); csv.addWidget(self.btn_look)
        main.addLayout(csv)

        # --- Matplotlib canvas: Echo, Gaze, Audio ---
        self.fig, (self.ax_echo, self.ax_look, self.ax_audio) = plt.subplots(1, 3, figsize=(12, 3))
        self.canvas = FigureCanvas(self.fig)
        main.addWidget(self.canvas, stretch=1)

        # --- Statistics bar ---
        stats_row = QHBoxLayout()
        self.lbl_stats = QLabel("Gazes: –    Look time: –s    Echo classes: –")
        stats_row.addWidget(self.lbl_stats)
        main.addLayout(stats_row)


    def _init_state(self):
        self.is_playing = False
        self.res        = None
        self.look_vec   = None
        self.audio_int  = None
        self.t          = None
        self.n          = 0
        self.line_res   = None
        self.line_look  = None
        self.line_audio = None

        # Video signals
        for p in (self.player1, self.player2):
            p.durationChanged.connect(self._set_duration)
            p.positionChanged.connect(self._update_frame)

    def _init_state(self):
        self.is_playing = False
        self.res        = None
        self.look_vec   = None
        self.audio_int  = None
        self.t          = None
        self.n          = 0
        self.line_res   = None
        self.line_look  = None
        self.line_audio = None

        # Video signals
        for p in (self.player1, self.player2):
            p.durationChanged.connect(self._set_duration)
            p.positionChanged.connect(self._update_frame)


    # --- Video loaders & sync ---
    def load_video1(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video 1")
        if p:
            self.player1.setMedia(QMediaContent(QUrl.fromLocalFile(p)))

    def load_video2(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video 2")
        if p:
            self.player2.setMedia(QMediaContent(QUrl.fromLocalFile(p)))
            self._extract_audio(p)
            # re-init plots if CSV already loaded
            self._init_plot_data()

    def _set_duration(self, _):
        d1, d2 = self.player1.duration(), self.player2.duration()
        if d1 > 0 and d2 > 0:
            dur = min(d1, d2)
            self.slider.setRange(0, dur)
            self.slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.lbl_end.setText(f"{dur/1000:.1f}s")
            # reset start
            self.slider.setValue(0)
            self.player1.setPosition(0)
            self.player2.setPosition(0)

    def toggle_play(self):
        ms = self.slider.value()
        self.player1.setPosition(ms)
        self.player2.setPosition(ms)
        if not self.is_playing:
            self.player1.play()
            self.player2.play()
            self.btn_play.setText("Pause")
        else:
            self.player1.pause()
            self.player2.pause()
            self.btn_play.setText("Play")
        self.is_playing = not self.is_playing

    def _seek(self, ms: int):
        self.player1.setPosition(ms)
        self.player2.setPosition(ms)

    # --- CSV loaders ---
    def load_echo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Echo CSV")
        if not p:
            return
        try:
            df = downsampling.perform_downsampling(p)
            self.res = df['Result'].astype(int).values
            self._init_plot_data()
        except Exception as e:
            QMessageBox.critical(self, "Echo CSV Error", str(e))

    def load_look(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Look CSV")
        if not p:
            return
        try:
            df = pd.read_csv(p)
            if '|' in df.columns:
                series = df['|']
            else:
                if df.shape[1] < 5:
                    raise ValueError("Need ≥5 columns")
                series = df.iloc[:, 4]
            self.look_vec = series.fillna(0).astype(int).values
            self._init_plot_data()
        except Exception as e:
            QMessageBox.critical(self, "Look CSV Error", str(e))

    # --- Audio extraction ---
    def _extract_audio(self, path: str):
        try:
            clip = VideoFileClip(path)
            # get stereo (n×2) at 44.1 kHz
            arr = clip.audio.to_soundarray(fps=44100)
            # average to mono
            mono = arr.mean(axis=1)
            sr = 44100
            # time vector in seconds
            self.audio_t = np.arange(len(mono)) / sr
            self.audio_wave = mono
        except Exception as e:
            print("Audio extract error:", e)
            self.audio_t = self.audio_wave = None

    # --- Plot setup ---
    def _init_plot_data(self):
        if self.res is None or self.look_vec is None:
            return

        # align lengths
        lengths = [len(self.res), len(self.look_vec)]
        if self.audio_int is not None:
            lengths.append(len(self.audio_int))
        self.n = min(lengths)
        self.t = np.arange(self.n) / self.FPS

        # clear and draw axes
        for ax in (self.ax_echo, self.ax_look, self.ax_audio):
            ax.clear()

        # Echo
        self.line_res, = self.ax_echo.plot([], [], lw=1)
        self.ax_echo.set(xlabel='[s]', title='Echo Log',
                         xlim=(0, self.t[-1]), ylim=(-0.5, 9.5))
        self.ax_echo.set_yticks([9,2,1,0])
        self.ax_echo.set_yticklabels(['NULL','IGNORING','CAR','AIRPLANE'])

        # Gaze
        self.line_look, = self.ax_look.plot([], [], lw=1)
        self.ax_look.set(xlabel='[s]', title='Gaze Detection',
                         xlim=(0, self.t[-1]), ylim=(-0.2,1.2))
        self.ax_look.set_yticks([0,1])
        self.ax_look.set_yticklabels(['NO LOOK','LOOK'])

        # Audio
        self.line_audio, = self.ax_audio.plot([], [], lw=0.5)
        self.ax_audio.set(xlabel='[s]', title='Audio Waveform',
                          xlim=(0, self.audio_t[-1]), 
                          ylim=(self.audio_wave.min(), self.audio_wave.max()))


        # compute and show stats right away
        self._compute_stats()

        self.canvas.draw_idle()

    # --- Frame update ---
    def _update_frame(self, pos_ms: int):
        # sync slider & label
        self.slider.blockSignals(True)
        self.slider.setValue(pos_ms)
        self.slider.blockSignals(False)
        self.lbl_start.setText(f"{pos_ms/1000:.1f}s")

        if self.t is None:
            return

        idx = min(int(pos_ms/1000 * self.FPS), self.n - 1)

        # update each trace
        self.line_res.set_data(self.t[:idx+1], self.res[:idx+1])
        self.line_look.set_data(self.t[:idx+1], self.look_vec[:idx+1])
        if self.audio_t is not None:
            # how many samples correspond to pos_ms
            idx_samp = int(pos_ms/1000 * 44100)
            idx_samp = min(idx_samp, len(self.audio_t)-1)
            self.line_audio.set_data(self.audio_t[:idx_samp], 
                                     self.audio_wave[:idx_samp])
        # stats don’t change per frame, so no need to recompute each time

        self.canvas.draw_idle()

        # stop at end
        if pos_ms >= self.slider.maximum():
            self.player1.pause()
            self.player2.pause()
            self.btn_play.setText("Play")
            self.is_playing = False


    def _compute_stats(self):
        # number of gazes = rising edges in look_vec
        rises = np.where(np.diff(np.concatenate(([0], self.look_vec[:self.n]))) == 1)[0]
        n_gazes    = len(rises)
        time_look  = self.look_vec[:self.n].sum() / self.FPS

        # echo class percentages
        valid = self.res[:self.n][self.res[:self.n] != 9]
        if valid.size:
            pct = pd.Series(valid).value_counts(normalize=True) * 100
            classes_txt = ", ".join(f"{lbl}:{pct.get(code,0):.1f}%"
                                    for code,lbl in [(0,"AIRPLANE"),
                                                     (1,"CAR"),
                                                     (2,"IGNORING")
                                                     ])
        else:
            classes_txt = "-"

        self.lbl_stats.setText(
            f"Gazes: {n_gazes}    Look time: {time_look:.1f}s    Echo classes: {classes_txt}"
        )



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = VideoSyncApp()
    w.show()
    sys.exit(app.exec_())
