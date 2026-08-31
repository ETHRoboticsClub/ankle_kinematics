"""
Interactive 3D visualization of the parallel-ankle mechanism (matplotlib).

Two sliders (pitch, roll) drive the foot pose; everything else updates live:
  - foot orientation
  - foot anchor points
  - required motor angles (via joint_to_motor)
  - crank-arm positions
  - pleuel rods connecting motor cranks to foot anchors

Run:
    python3.11 visualize.py            # right leg (the measured one)
    python3.11 visualize.py left       # left leg (mirrored)

Sliders are auto-ranged to the mechanism's actual reachable envelope.
Dashed line through each motor = its rotation axis; faint circle = the path
its crank tip sweeps. Pleuels turn red if a pose leaves the workspace.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry import LEFT_LEG, RIGHT_LEG, foot_rotation
from kinematics import foot_anchor_world, crank_tip_world
from mapping import joint_to_motor, tilt_angle


# --- Visual style ---------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 9,
    "axes.titlesize": 11,
})

BG_FIG    = "#0e1116"
BG_AX     = "#11151c"
TICK_COL  = "#7a8190"
TEXT_COL  = "#c8ccd4"

COLOR_PIVOT  = "#ffffff"
COLOR_SHIN   = "#9aa5b5"
COLOR_MOTOR  = "#4f9eff"
COLOR_CRANK  = "#9fc8ff"
COLOR_PLEUEL = "#69e6a1"
COLOR_FOOT   = "#ff6b6b"
COLOR_FOOT_FACE = (1.0, 0.42, 0.42, 0.18)
COLOR_AXIS_X = "#ff5555"
COLOR_AXIS_Y = "#55cc77"
COLOR_AXIS_Z = "#5588ff"

DEG = np.pi / 180.0


def foot_polygon(geom, theta_p, theta_r):
    """4-corner foot rectangle, rotated with the foot."""
    R = foot_rotation(geom, theta_p, theta_r)
    corners = np.array([
        [ 0.10, -0.07, -0.018],
        [ 0.10,  0.07, -0.018],
        [-0.06,  0.07, -0.018],
        [-0.06, -0.07, -0.018],
    ])
    return (R @ corners.T).T


def reach_ok(geom, tp, tr):
    """Is this foot pose inside both pleuels' workspace?"""
    for m in (geom.upper, geom.lower):
        P = foot_anchor_world(geom, m, tp, tr)
        d = m.C - P
        A = 2 * m.r * np.dot(d, m.u_hat)
        B = 2 * m.r * np.dot(d, m.v_hat)
        K = m.L**2 - np.dot(d, d) - m.r**2
        if abs(K / np.hypot(A, B)) > 1.0:
            return False
    return True


def envelope(geom, axis, limit=89.0):
    """Reachable range in degrees along one joint axis (0 = pitch, 1 = roll)."""
    out = []
    for sign in (-1, +1):
        a = 0.0
        while a < limit:
            a += 0.5
            args = (a * sign * DEG, 0.0) if axis == 0 else (0.0, a * sign * DEG)
            if not reach_ok(geom, *args):
                break
        out.append((a - 0.5) * sign)
    return out[0], out[1]


def main():
    leg = sys.argv[1].lower() if len(sys.argv) > 1 else "right"
    geom = RIGHT_LEG if leg.startswith("r") else LEFT_LEG
    state = {"prev": (0.0, 0.0)}

    fig = plt.figure(figsize=(11, 9), facecolor=BG_FIG)
    fig.canvas.manager.set_window_title("Ankle Parallel Mechanism")
    ax = fig.add_subplot(111, projection="3d", facecolor=BG_AX)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.22, top=0.96)

    # Sliders
    ax_pitch = fig.add_axes([0.20, 0.11, 0.65, 0.025], facecolor="#1a1f29")
    ax_roll  = fig.add_axes([0.20, 0.07, 0.65, 0.025], facecolor="#1a1f29")
    p_lo, p_hi = envelope(geom, 0)
    r_lo, r_hi = envelope(geom, 1)
    s_pitch = Slider(ax_pitch, "pitch [deg]", p_lo, p_hi, valinit=0,
                     color="#4f9eff", track_color="#22293a", initcolor="none")
    s_roll  = Slider(ax_roll,  "roll  [deg]", r_lo, r_hi, valinit=0,
                     color="#ff6b6b", track_color="#22293a", initcolor="none")
    for s in (s_pitch, s_roll):
        s.label.set_color(TEXT_COL)
        s.valtext.set_color(TEXT_COL)

    info_text = fig.text(
        0.5, 0.025, "", ha="center", va="center",
        fontsize=10, family="monospace", color="#aaaeb6",
    )
    title_text = fig.text(
        0.5, 0.965, f"Parallel ankle mechanism — {geom.side} leg — "
        f"reachable pitch {p_lo:+.0f}..{p_hi:+.0f}°, roll {r_lo:+.0f}..{r_hi:+.0f}°",
        ha="center", va="center", fontsize=12, color=TEXT_COL, weight="bold",
    )

    def style_axes():
        # Pane / grid colors
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.07, 0.09, 0.12, 1.0))
            axis._axinfo["grid"]["color"] = (0.20, 0.23, 0.28, 0.55)
            axis._axinfo["grid"]["linewidth"] = 0.5
            axis.label.set_color(TEXT_COL)
            for t in axis.get_ticklabels():
                t.set_color(TICK_COL)
            axis.set_tick_params(colors=TICK_COL, labelsize=8)
        ax.set_xlabel("x  (forward)  [m]")
        ax.set_ylabel("y  (right)    [m]")
        ax.set_zlabel("z  (up)       [m]")

    def draw():
        ax.clear()
        style_axes()
        tp = np.deg2rad(s_pitch.val)
        tr = np.deg2rad(s_roll.val)

        # Solve mapping; capture any "pose unreachable" warning to flag it.
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            tA, tB = joint_to_motor(geom, tp, tr, state["prev"])
        state["prev"] = (tA, tB)
        unreachable = any("unreachable" in str(w.message) for w in caught)

        pk_up = crank_tip_world(geom.upper, tA)
        pk_lo = crank_tip_world(geom.lower, tB)
        pf_up = foot_anchor_world(geom, geom.upper, tp, tr)
        pf_lo = foot_anchor_world(geom, geom.lower, tp, tr)
        pm_up = geom.upper.C
        pm_lo = geom.lower.C

        pleuel_color = "#ff3838" if unreachable else COLOR_PLEUEL

        # Shin: a fat line from pivot up past the highest motor
        shin_top = np.array([0, 0, max(pm_up[2], pm_lo[2]) * 1.08])
        ax.plot([0, shin_top[0]], [0, shin_top[1]], [0, shin_top[2]],
                color=COLOR_SHIN, lw=8, solid_capstyle="round", alpha=0.85)

        # Reference axes at pivot (small)
        L = 0.04
        ax.plot([0, L], [0, 0], [0, 0], color=COLOR_AXIS_X, lw=1.5, alpha=0.9)
        ax.plot([0, 0], [0, L], [0, 0], color=COLOR_AXIS_Y, lw=1.5, alpha=0.9)
        ax.plot([0, 0], [0, 0], [0, L], color=COLOR_AXIS_Z, lw=1.5, alpha=0.9)

        # Pivot marker
        ax.scatter(0, 0, 0, s=80, color=COLOR_PIVOT, marker="o",
                   edgecolor="black", linewidth=0.8, zorder=10)

        # Foot polygon (filled, semi-transparent)
        poly = foot_polygon(geom, tp, tr)
        face = Poly3DCollection([poly], facecolor=COLOR_FOOT_FACE,
                                edgecolor=COLOR_FOOT, lw=1.2)
        ax.add_collection3d(face)

        # Motors (big squares) + their rotation axes
        for pm, m in ((pm_up, geom.upper), (pm_lo, geom.lower)):
            ax.scatter(*pm, s=140, color=COLOR_MOTOR, marker="s",
                       edgecolor="black", linewidth=0.8, zorder=8)
            ends = np.array([pm - m.n_unit * m.r * 1.1,
                             pm + m.n_unit * m.r * 1.1])
            ax.plot(*ends.T, color=COLOR_MOTOR, lw=1.6, ls=(0, (4, 3)), alpha=0.9)
            # circle the crank tip sweeps out
            t = np.linspace(0, 2 * np.pi, 80)
            circ = (pm[:, None] + m.r * (np.cos(t) * m.u_hat[:, None]
                                         + np.sin(t) * m.v_hat[:, None]))
            ax.plot(*circ, color=COLOR_MOTOR, lw=0.7, alpha=0.35)

        # Crank arms
        for pm, pk in ((pm_up, pk_up), (pm_lo, pk_lo)):
            ax.plot(*zip(pm, pk), color=COLOR_CRANK, lw=3.5,
                    solid_capstyle="round", zorder=6)
            ax.scatter(*pk, s=50, color=COLOR_CRANK, edgecolor="black",
                       linewidth=0.6, zorder=7)

        # Pleuels (turn red when the pose is outside the workspace)
        for pk, pf in ((pk_up, pf_up), (pk_lo, pf_lo)):
            ax.plot(*zip(pk, pf), color=pleuel_color, lw=3.5,
                    solid_capstyle="round", zorder=5)

        # Foot anchors
        for pf in (pf_up, pf_lo):
            ax.scatter(*pf, s=90, color=COLOR_FOOT, marker="o",
                       edgecolor="black", linewidth=0.8, zorder=9)

        # Frame: auto-scaled to the mechanism, equal aspect so angles read true
        pts = np.array([pk_up, pk_lo, pf_up, pf_lo, pm_up, pm_lo,
                        shin_top, [0, 0, 0], *poly])
        ctr = (pts.max(axis=0) + pts.min(axis=0)) / 2
        span = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.60
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        ax.set_box_aspect([1, 1, 1])

        l_up = np.linalg.norm(pk_up - pf_up) * 100
        l_lo = np.linalg.norm(pk_lo - pf_lo) * 100
        tilt = np.rad2deg(tilt_angle(tp, tr))
        over = tilt > np.rad2deg(geom.max_tilt) + 1e-9
        flag = "  ⚠ OUT OF REACH" if unreachable else (
               "  ⚠ PAST KREUZGELENK CONE" if over else "")
        info_text.set_text(
            f"motor upper = {np.rad2deg(tA):+7.2f}°    "
            f"motor lower = {np.rad2deg(tB):+7.2f}°    "
            f"|  U-joint tilt = {tilt:5.2f}° / {np.rad2deg(geom.max_tilt):.0f}°    "
            f"|  L_up = {l_up:5.2f} cm (set {geom.upper.L*100:.2f})    "
            f"L_lo = {l_lo:5.2f} cm (set {geom.lower.L*100:.2f})"
            f"{flag}"
        )
        info_text.set_color("#ff7777" if (unreachable or over) else "#aaaeb6")
        fig.canvas.draw_idle()

    s_pitch.on_changed(lambda _: draw())
    s_roll.on_changed(lambda _: draw())

    # --- Camera-preset buttons -------------------------------------------
    # (elev, azim) for each preset
    presets = [
        ("Iso",     30, -45),
        ("3/4",     18, -55),
        ("Front",    0, -90),  # looking along +y (from -y toward origin)
        ("Side",     0,   0),  # looking along -x (from +x toward origin)
        ("Top",     89, -90),
        ("Back",     8,  90),
    ]
    btn_w, btn_h, btn_gap = 0.07, 0.035, 0.008
    btn_y = 0.165
    start_x = 0.5 - (len(presets) * btn_w + (len(presets)-1) * btn_gap) / 2

    button_refs = []  # keep references alive

    def make_setter(elev, azim):
        def _set(_event):
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()
        return _set

    for i, (label, elev, azim) in enumerate(presets):
        bx = start_x + i * (btn_w + btn_gap)
        ax_btn = fig.add_axes([bx, btn_y, btn_w, btn_h])
        b = Button(ax_btn, label, color="#1a1f29", hovercolor="#2a3142")
        b.label.set_color(TEXT_COL)
        b.label.set_fontsize(9)
        b.on_clicked(make_setter(elev, azim))
        button_refs.append(b)

    # Keyboard shortcuts: 1..6 for the presets above
    def on_key(event):
        if event.key in "123456":
            idx = int(event.key) - 1
            if idx < len(presets):
                _, e, a = presets[idx]
                ax.view_init(elev=e, azim=a)
                fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial camera angle
    ax.view_init(elev=18, azim=-55)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
