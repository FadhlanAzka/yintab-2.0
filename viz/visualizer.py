import os
import numpy as np
import pygame
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from settings import FIG_SIZE, UPDATE_INTERVAL_MS, FMIN, FMAX


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def show_visualization(
    audio_path: str,
    duration: float,
    sr: int,
    tempo: float,
    beat_times: np.ndarray,
    onset_times: np.ndarray,
    times: np.ndarray,
    f0: np.ndarray,
    notes: np.ndarray
):
    fig = plt.figure(figsize=FIG_SIZE)
    ax_ana  = fig.add_axes([0.08, 0.30, 0.84, 0.60])
    ax_prog = fig.add_axes([0.08, 0.20, 0.84, 0.05])
    ax_play = fig.add_axes([0.18, 0.07, 0.18, 0.08])
    ax_pause= fig.add_axes([0.41, 0.07, 0.18, 0.08])
    ax_stop = fig.add_axes([0.64, 0.07, 0.18, 0.08])

    # warna per note unik (berdasar kemunculan pertama)
    cmap = plt.get_cmap("tab20")
    N = getattr(cmap, "N", 20)

    unique_notes, seen = [], set()
    for n in notes:
        if n != "N/A" and n not in seen:
            seen.add(n); unique_notes.append(n)
    note2color = {n: cmap((i % N)/max(1, (N-1))) for i, n in enumerate(unique_notes)}

    for n in unique_notes:
        idx = notes == n
        ax_ana.scatter(times[idx], f0[idx], s=20, color=note2color[n], label=n)

    for t in onset_times:
        ax_ana.axvline(t, linestyle="--", alpha=0.5, linewidth=1)
    for t in beat_times:
        ax_ana.axvline(t, linestyle="-", alpha=0.3, linewidth=1)

    pointer_line = ax_ana.axvline(0.0, color="k", linewidth=2, alpha=0.9)

    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=note2color[n], markersize=8, label=n)
        for n in unique_notes
    ] + [
        plt.Line2D([0], [0], color="0.2", lw=1, ls="--", label="Onset"),
        plt.Line2D([0], [0], color="0.2", lw=1, ls="-", alpha=0.3, label="Beat"),
        plt.Line2D([0], [0], color="k", lw=2, label="Pointer"),
    ]
    ax_ana.legend(handles=legend_items, loc="upper right", title=f"Tempo ≈ {tempo:.1f} BPM", ncol=3)

    ax_ana.set_title(
        f"Pitch (YIN) + Onset + Beat Track + Player\n"
        f"File: {os.path.basename(audio_path)} | Durasi: {duration:.2f}s | sr={sr} Hz"
    )
    ax_ana.set_ylabel("Frekuensi Fundamental F₀ (Hz)")
    ax_ana.set_xlim(0, duration)
    ax_ana.set_ylim(bottom=FMIN * 0.9, top=FMAX * 1.1)
    ax_ana.grid(True, which="both", axis="both", alpha=0.25)

    progress_bar = ax_prog.barh([0], [0], height=0.6)[0]
    ax_prog.set_xlim(0, duration)
    ax_prog.set_ylim(-0.5, 0.5)
    ax_prog.set_xticks([]); ax_prog.set_yticks([])
    time_text = ax_prog.text(duration/2, 0.55, "00:00 / 00:00",
                             ha="center", va="bottom", fontsize=10, fontweight="bold")

    btn_play  = Button(ax_play,  "Play")
    btn_pause = Button(ax_pause, "Pause")
    btn_stop  = Button(ax_stop,  "Stop")

    # ====== Pygame init & state ======
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)

    state = {
        "start_at": 0.0,
        "t0_ms": None,
        "paused": True,
        "dragging": False,
        "drag_pos": 0.0,
        "pos_frozen": 0.0
    }

    def start_play_from(offset_sec: float):
        offset_sec = float(np.clip(offset_sec, 0.0, duration))
        state["start_at"] = offset_sec
        state["t0_ms"] = pygame.time.get_ticks()
        try:
            pygame.mixer.music.play(start=offset_sec)
        except TypeError:
            pygame.mixer.music.play()
            try:
                pygame.mixer.music.set_pos(offset_sec)
            except Exception:
                pass
        state["paused"] = False

    def get_play_pos():
        if not state["paused"] and state["t0_ms"] is not None:
            now_ms = pygame.time.get_ticks()
            elapsed = max(0, now_ms - state["t0_ms"]) / 1000.0
            return min(state["start_at"] + elapsed, duration)
        return state["pos_frozen"]

    def do_play(event=None):
        if state["paused"]:
            start_play_from(state["pos_frozen"])

    def do_pause(event=None):
        state["pos_frozen"] = get_play_pos()
        pygame.mixer.music.pause()
        state["paused"] = True
        state["t0_ms"] = None

    def do_stop(event=None):
        pygame.mixer.music.stop()
        state["paused"] = True
        state["t0_ms"] = None
        state["pos_frozen"] = 0.0
        progress_bar.set_width(0.0)
        time_text.set_text(f"{_format_time(0.0)} / {_format_time(duration)}")
        pointer_line.set_xdata([0.0, 0.0])
        fig.canvas.draw_idle()

    btn_play.on_clicked(do_play)
    btn_pause.on_clicked(do_pause)
    btn_stop.on_clicked(do_stop)

    # Drag-seek
    def on_press(event):
        if event.inaxes == ax_prog and event.xdata is not None:
            state["dragging"] = True
            state["drag_pos"] = float(np.clip(event.xdata, 0.0, duration))

    def on_motion(event):
        if state["dragging"] and event.inaxes == ax_prog and event.xdata is not None:
            state["drag_pos"] = float(np.clip(event.xdata, 0.0, duration))

    def on_release(event):
        if state["dragging"]:
            state["dragging"] = False
            state["pos_frozen"] = state["drag_pos"]
            start_play_from(state["pos_frozen"])

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # Animator
    def update(_frame):
        pos = state["drag_pos"] if state["dragging"] else get_play_pos()
        if not state["paused"] and pos >= duration - 1e-3 and not pygame.mixer.music.get_busy():
            do_stop()
        progress_bar.set_width(pos)
        time_text.set_text(f"{_format_time(pos)} / {_format_time(duration)}")
        pointer_line.set_xdata([pos, pos])
        return progress_bar, time_text, pointer_line

    # ⬇️ ini sudah diperbaiki, simpan animasi ke variabel ani
    ani = animation.FuncAnimation(
        fig,
        update,
        interval=UPDATE_INTERVAL_MS,
        cache_frame_data=False
    )

    plt.show()
    return ani  # kembalikan ani supaya tidak di-garbage collect

def save_visualization(out_dir):
    """
    Factory -> kembalikan fungsi penyimpan PNG:
      save_visualization(out_dir)(audio_path, duration, sr, tempo,
                                  beat_times, onset_times, times, f0, notes)
    Hasil: <out_dir>/<stem>_viz.png

    Catatan:
    - Styling & layout meniru show_visualization (axes, legend, colors).
    - Tanpa pygame & animation; tombol hanya digambar statis.
    """
    def _save(
        audio_path: str,
        duration: float,
        sr: int,
        tempo: float,
        beat_times: np.ndarray,
        onset_times: np.ndarray,
        times: np.ndarray,
        f0: np.ndarray,
        notes: np.ndarray
    ):
        import matplotlib
        #matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig = plt.figure(figsize=FIG_SIZE)
        ax_ana  = fig.add_axes([0.08, 0.30, 0.84, 0.60])

        # Warna per-note: identik dengan show_visualization
        cmap = plt.get_cmap("tab20")
        N = getattr(cmap, "N", 20)
        unique_notes, seen = [], set()
        for n in notes:
            if n != "N/A" and n not in seen:
                seen.add(n); unique_notes.append(n)
        note2color = {n: cmap((i % N)/max(1, (N-1))) for i, n in enumerate(unique_notes)}

        for n in unique_notes:
            idx = (notes == n)
            ax_ana.scatter(times[idx], f0[idx], s=20, color=note2color[n], label=n)

        for t in onset_times:
            ax_ana.axvline(t, linestyle="--", alpha=0.5, linewidth=1)
        for t in beat_times:
            ax_ana.axvline(t, linestyle="-", alpha=0.3, linewidth=1)

        # Pointer awal di 0
        ax_ana.axvline(0.0, color="k", linewidth=2, alpha=0.9)

        legend_items = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=note2color[n], markersize=8, label=n)
            for n in unique_notes
        ] + [
            plt.Line2D([0], [0], color="0.2", lw=1, ls="--", label="Onset"),
            plt.Line2D([0], [0], color="0.2", lw=1, ls="-", alpha=0.3, label="Beat"),
            plt.Line2D([0], [0], color="k", lw=2, label="Pointer"),
        ]
        ax_ana.legend(handles=legend_items, loc="upper right", title=f"Tempo ≈ {tempo:.1f} BPM", ncol=3)

        ax_ana.set_title(
            f"File: {os.path.basename(audio_path)} | Durasi: {duration:.2f}s | sr={sr} Hz"
        )
        ax_ana.set_ylabel("F₀ (Hz)")
        ax_ana.set_xlim(0, duration)
        ax_ana.set_ylim(bottom=FMIN * 0.9, top=FMAX * 1.1)
        ax_ana.grid(True, which="both", axis="both", alpha=0.25)        

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_png = out_dir_path / f"{Path(audio_path).stem}_viz.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_png

    return _save