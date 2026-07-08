import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


def detect_starting_beep(video_path, search_window_s=12.0):
    """
    Detect the starting beep in the video's audio track.

    Strategy:
    - Load the first `search_window_s` seconds of audio.
    - Compute per-frame energy in the 800–4000 Hz band (where electronic
      start beeps and starter pistols live).
    - Find the first frame whose band energy exceeds a dynamic threshold
      (mean + 3 × std of the whole window), which is where the beep is.

    Returns (beep_timestamp_s, confidence_ratio) or (None, 0.0) on failure.
    confidence_ratio is peak_energy / mean_energy — higher = sharper spike.
    """
    if not _LIBROSA_AVAILABLE:
        print("  [BEEP] librosa not available — skipping audio detection.")
        return None, 0.0

    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True,
                             offset=0.0, duration=search_window_s)
    except Exception as e:
        print(f"  [BEEP] Could not load audio from '{video_path}': {e}")
        return None, 0.0

    # STFT → frequency bins × time frames
    n_fft    = 2048
    hop_len  = 512
    D        = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times    = librosa.frames_to_time(np.arange(D.shape[1]),
                                      sr=sr, hop_length=hop_len)

    # Isolate 800–4000 Hz band
    band_mask   = (freqs >= 800) & (freqs <= 4000)
    band_energy = D[band_mask, :].mean(axis=0)   # (n_frames,)

    if band_energy.max() == 0:
        print("  [BEEP] Audio track appears silent.")
        return None, 0.0

    # Dynamic threshold: mean + 3 × std catches a sharp beep over ambient noise
    mean_e = float(np.mean(band_energy))
    std_e  = float(np.std(band_energy))
    threshold = mean_e + 3.0 * std_e

    above = np.where(band_energy > threshold)[0]
    if len(above) == 0:
        # Fall back to a softer threshold (mean + 2 × std)
        threshold = mean_e + 2.0 * std_e
        above = np.where(band_energy > threshold)[0]

    if len(above) == 0:
        print("  [BEEP] No distinct beep found in the audio.")
        return None, 0.0

    # Find the peak of the first group of above-threshold frames.
    # Then backtrack to find where energy first started rising above ambient
    # (mean + 1σ) — that is the true beep onset, not its mid-point.
    peak_frame    = int(above[np.argmax(band_energy[above])])
    onset_thr     = mean_e + 1.0 * std_e
    beep_frame    = peak_frame
    for i in range(peak_frame, -1, -1):
        if band_energy[i] < onset_thr:
            beep_frame = i + 1   # first frame that crossed the onset level
            break

    beep_time  = float(times[beep_frame])
    confidence = float(band_energy[peak_frame]) / (mean_e + 1e-9)

    print(f"  [BEEP] Onset @ {beep_time:.3f}s  "
          f"(peak @ {times[peak_frame]:.3f}s, confidence {confidence:.1f}×)")
    return beep_time, confidence

# ─────────────────────────────────────────────────────────────────────────── #
#  DEFAULT CONFIGURATION                                                       #
#  All tunable thresholds live here so the retry loop can override them.      #
# ─────────────────────────────────────────────────────────────────────────── #
DEFAULT_CONFIG = {
    # START: adaptive baseline – collect this many still frames, then trigger
    # when velocity exceeds baseline × start_motion_multiplier.
    'baseline_frames':         25,
    'start_motion_multiplier': 8,
    'start_threshold_floor':   0.005,  # absolute minimum threshold

    # FINISH: sharp stop after minimum race time.
    # Lower finish_still_min_frames = catches wall touch sooner.
    'min_race_duration':          44.0,   # seconds; skip flip-turn region
    'finish_velocity_threshold':  0.015,  # normalised units/frame
    'finish_still_min_frames':    8,      # ~0.27s at 30fps = catches wall touch

    # STROKE DETECTION — wrist anatomy (runs always)
    'recovery_offset':  0.05,
    'entry_offset':     0.08,
    'smoothing_frames': 2,
    'min_cycle_time':   0.45,  # global dedup gate shared by all detectors
    'max_cycle_time':   3.0,

    # STROKE DETECTION — splash frame-diff (runs in parallel)
    # A hand entry creates a burst of bright pixels in a water-level ROI
    # anchored to the swimmer's shoulder position.
    'splash_enabled':        True,
    'splash_roi_half_w':     0.30,  # ROI half-width in normalised coords
    'splash_roi_y_above':    0.05,  # ROI top edge above shoulder_y
    'splash_roi_y_below':    0.18,  # ROI bottom edge below shoulder_y
    'splash_smoothing':      3,     # frames to smooth the diff signal
    # Adaptive rolling-median threshold: adapts to camera panning speed.
    # threshold = rolling_median(last N frames) × relative_multiplier
    'splash_rolling_window':     45,   # frames for the rolling median baseline
    'splash_relative_multiplier': 2.0, # spike must be this many × rolling median
}


def analyze_swim_video(input_video_path, output_video_path, config=None):
    """
    Analyse a swim race video.
    config: dict of threshold overrides (keys from DEFAULT_CONFIG).
    Returns a result dict; keys are None when detection failed.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    print(f"Loading AI Models to analyse '{input_video_path}'...")

    # ── AUDIO: BEEP DETECTION ───────────────────────────────────────────────
    # Run before the video loop (audio is independent of pose estimation).
    print("  Scanning audio track for starting beep…")
    beep_time, beep_confidence = detect_starting_beep(
        input_video_path, search_window_s=cfg.get('beep_search_window_s', 12.0)
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    fps   = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ── AUTO-START ─────────────────────────────────────────────────────────
    # beep_time       → when the starting signal fired (from audio)
    # first_movement_time → when the body first moved (velocity spike on block)
    # race_start_time → whichever is available; beep preferred as the "clock zero"
    # reaction_time   → first_movement_time − beep_time (how long on the block)
    first_movement_time = None
    race_start_time     = beep_time   # pre-set from audio; None if not detected
    reaction_time       = None

    prev_shoulder_x  = None
    baseline_vels    = []
    start_threshold  = cfg['start_threshold_floor']   # refined once baseline is ready

    # ── AUTO-FINISH ────────────────────────────────────────────────────────
    # After the wall touch the swimmer stops instantly.
    # Detect as: velocity < threshold for finish_still_min_frames consecutive frames.
    race_finish_time  = None
    finish_still_streak = 0
    finish_still_start  = None

    # ── STROKE DETECTION ───────────────────────────────────────────────────
    arm_cycle_count      = 0
    arm_cycle_timestamps = []
    stroke_timestamps    = []
    current_stroke_rate  = 0.0
    is_recovering        = False

    # Per-wrist smoothing histories and recovery latches
    lw_y_history   = []
    rw_y_history   = []
    lw_recovering  = False
    rw_recovering  = False

    # Splash frame-diff detector state
    prev_gray           = None
    splash_diff_history = []          # short smoothing window
    splash_rolling      = []          # longer rolling window for adaptive baseline
    in_splash           = False       # True while score is above adaptive threshold

    detection_status = "Calibrating baseline..."

    print("Analysing frames… this may take a few minutes.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts        = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results   = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            avg_shoulder_x = (ls.x + rs.x) / 2.0
            avg_shoulder_y = (ls.y + rs.y) / 2.0
            x_vel = abs(avg_shoulder_x - prev_shoulder_x) if prev_shoulder_x is not None else 0.0

            # ── STEP 1: ADAPTIVE START ─────────────────────────────────────
            if first_movement_time is None:
                # Phase A — calibrate from stillness
                if len(baseline_vels) < cfg['baseline_frames']:
                    if prev_shoulder_x is not None and x_vel < 0.03:
                        baseline_vels.append(x_vel)
                    if len(baseline_vels) >= 10:
                        mean_v = float(np.mean(baseline_vels))
                        std_v  = float(np.std(baseline_vels))
                        start_threshold = max(
                            mean_v + cfg['start_motion_multiplier'] * std_v,
                            cfg['start_threshold_floor']
                        )
                        detection_status = f"Baseline ready (thresh={start_threshold:.4f})"

                # Phase B — detect first motion on the block
                if prev_shoulder_x is not None and x_vel > start_threshold:
                    first_movement_time = ts
                    # If beep was not found in audio, fall back to motion as start
                    if race_start_time is None:
                        race_start_time = first_movement_time
                    # Calculate reaction time
                    if beep_time is not None:
                        reaction_time = round(first_movement_time - beep_time, 3)
                    detection_status = (
                        f"START @ {race_start_time:.2f}s"
                        + (f"  RT={reaction_time:.3f}s" if reaction_time is not None else "")
                    )
                    print(f"  [AUTO-DETECT] First MOVEMENT @ {first_movement_time:.2f}s  "
                          f"(ΔX={x_vel:.4f}, thresh={start_threshold:.4f})")
                    if beep_time is not None:
                        print(f"  [REACTION TIME] {reaction_time:.3f}s  "
                              f"(moved {reaction_time:.3f}s after beep)")

            # ── STEP 2: SHARP-STOP FINISH ──────────────────────────────────
            if race_start_time is not None and race_finish_time is None and first_movement_time is not None:
                elapsed = ts - race_start_time
                if elapsed > cfg['min_race_duration']:
                    if x_vel < cfg['finish_velocity_threshold']:
                        if finish_still_start is None:
                            finish_still_start = ts
                        finish_still_streak += 1
                        if finish_still_streak >= cfg['finish_still_min_frames']:
                            race_finish_time  = finish_still_start
                            detection_status  = f"FINISH @ {race_finish_time:.2f}s"
                            print(f"  [AUTO-DETECT] Race FINISH @ {race_finish_time:.2f}s")
                    else:
                        finish_still_streak = 0
                        finish_still_start  = None

            prev_shoulder_x = avg_shoulder_x

            # ── STEP 3: SINGLE-ARM CYCLE STROKE DETECTION ─────────────────
            # Strokes are only counted after the swimmer is actually in the water,
            # i.e. after first body movement (not just the beep).
            race_is_live = (
                first_movement_time is not None
                and race_finish_time is None
                and ts > first_movement_time
            )

            if race_is_live:
                rec_thresh   = avg_shoulder_y + cfg['recovery_offset']
                entry_thresh = avg_shoulder_y + cfg['entry_offset']
                cycle_fired  = False

                # ── LEFT WRIST ──────────────────────────────────────────────
                lw_y_history.append(lw.y)
                if len(lw_y_history) > cfg['smoothing_frames']:
                    lw_y_history.pop(0)
                slw = sum(lw_y_history) / len(lw_y_history)

                if slw < rec_thresh:
                    lw_recovering = True
                elif slw > entry_thresh and lw_recovering:
                    lw_recovering = False
                    cycle_fired   = True

                # ── RIGHT WRIST ─────────────────────────────────────────────
                rw_y_history.append(rw.y)
                if len(rw_y_history) > cfg['smoothing_frames']:
                    rw_y_history.pop(0)
                srw = sum(rw_y_history) / len(rw_y_history)

                if srw < rec_thresh:
                    rw_recovering = True
                elif srw > entry_thresh and rw_recovering:
                    rw_recovering = False
                    cycle_fired   = True

                # ── DEDUPLICATE & COUNT ─────────────────────────────────────
                # Accept the first trigger in any min_cycle_time window.
                if cycle_fired:
                    last_ts = arm_cycle_timestamps[-1] if arm_cycle_timestamps else 0.0
                    if (ts - last_ts) >= cfg['min_cycle_time']:
                        arm_cycle_count += 1
                        arm_cycle_timestamps.append(ts)
                        stroke_timestamps.append(ts)

                        # Median of last 3 gaps → stable, outlier-resistant SPM
                        if len(arm_cycle_timestamps) >= 2:
                            gaps = [
                                arm_cycle_timestamps[i] - arm_cycle_timestamps[i - 1]
                                for i in range(
                                    max(1, len(arm_cycle_timestamps) - 3),
                                    len(arm_cycle_timestamps)
                                )
                            ]
                            median_gap = sorted(gaps)[len(gaps) // 2]
                            if cfg['min_cycle_time'] <= median_gap <= cfg['max_cycle_time']:
                                current_stroke_rate = (1.0 / median_gap) * 60.0 * 2.0

            # ── STEP 4: SPLASH FRAME-DIFF DETECTOR ────────────────────────
            if cfg['splash_enabled']:
                if prev_gray is not None:
                    # Swimmer-anchored ROI at water level
                    rx1 = max(0,     int((avg_shoulder_x - cfg['splash_roi_half_w']) * width))
                    rx2 = min(width, int((avg_shoulder_x + cfg['splash_roi_half_w']) * width))
                    ry1 = max(0,      int((avg_shoulder_y - cfg['splash_roi_y_above']) * height))
                    ry2 = min(height, int((avg_shoulder_y + cfg['splash_roi_y_below']) * height))

                    roi_diff = cv2.absdiff(curr_gray[ry1:ry2, rx1:rx2],
                                           prev_gray[ry1:ry2, rx1:rx2])
                    diff_score = float(np.mean(roi_diff))

                    # Always update smoothing and rolling windows
                    splash_diff_history.append(diff_score)
                    if len(splash_diff_history) > cfg['splash_smoothing']:
                        splash_diff_history.pop(0)
                    smoothed_diff = sum(splash_diff_history) / len(splash_diff_history)

                    splash_rolling.append(smoothed_diff)
                    if len(splash_rolling) > cfg['splash_rolling_window']:
                        splash_rolling.pop(0)

                    # Adaptive threshold = rolling median × relative_multiplier
                    # Rolling median tracks current camera-pan baseline automatically.
                    if len(splash_rolling) >= cfg['splash_smoothing']:
                        sorted_roll  = sorted(splash_rolling)
                        roll_median  = sorted_roll[len(sorted_roll) // 2]
                        adaptive_thr = roll_median * cfg['splash_relative_multiplier']

                        if race_is_live:
                            if smoothed_diff > adaptive_thr:
                                in_splash = True
                            elif in_splash:
                                # Trailing edge of splash peak → stroke entry
                                in_splash = False
                                last_ts = arm_cycle_timestamps[-1] if arm_cycle_timestamps else 0.0
                                if (ts - last_ts) >= cfg['min_cycle_time']:
                                    arm_cycle_count += 1
                                    arm_cycle_timestamps.append(ts)
                                    stroke_timestamps.append(ts)

                                    if len(arm_cycle_timestamps) >= 2:
                                        gaps = [
                                            arm_cycle_timestamps[i] - arm_cycle_timestamps[i - 1]
                                            for i in range(
                                                max(1, len(arm_cycle_timestamps) - 3),
                                                len(arm_cycle_timestamps)
                                            )
                                        ]
                                        median_gap = sorted(gaps)[len(gaps) // 2]
                                        if cfg['min_cycle_time'] <= median_gap <= cfg['max_cycle_time']:
                                            current_stroke_rate = (1.0 / median_gap) * 60.0 * 2.0

                    # Draw ROI on frame for visual debugging
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 165, 255), 1)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Always update prev_gray (outside pose block so no frames are skipped)
        prev_gray = curr_gray

        # ── OVERLAY DASHBOARD ───────────────────────────────────────────────
        # Race clock counts from the BEEP (official start), not first movement.
        clock_origin  = race_start_time   # beep time if available, else first movement
        race_elapsed  = ts - clock_origin if clock_origin is not None else 0.0

        # Race Time = finish − beep, shown once finish is locked; counts up live until then
        if race_finish_time is not None:
            race_time_s       = race_finish_time - clock_origin
            mins              = int(race_time_s // 60)
            secs              = race_time_s % 60
            race_time_str     = f"FINAL {mins}:{secs:05.2f}"
            race_time_colour  = (0, 215, 255)   # gold
        elif clock_origin is not None:
            mins              = int(race_elapsed // 60)
            secs              = race_elapsed % 60
            race_time_str     = f"{mins}:{secs:05.2f}"
            race_time_colour  = (255, 255, 255)
        else:
            race_time_str     = "--:--.--"
            race_time_colour  = (160, 160, 160)

        # Reaction time string
        if reaction_time is not None:
            rt_str   = f"{reaction_time:.3f}s"
            rt_color = (0, 255, 128)   # green-ish
        elif beep_time is not None and first_movement_time is None:
            rt_str   = "waiting…"
            rt_color = (160, 160, 160)
        else:
            rt_str   = "no beep"
            rt_color = (100, 100, 100)

        overlay_h = 300 if beep_time is not None else 255
        cv2.rectangle(frame, (10, 10), (580, overlay_h), (0, 0, 0), -1)
        cv2.putText(frame, detection_status, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)
        cv2.putText(frame, f"Signal       : BOTH WRISTS (dedup {cfg['min_cycle_time']}s)", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)
        cv2.putText(frame, f"Arm Cycles   : {arm_cycle_count}  "
                            f"(~{arm_cycle_count * 2} total strokes)", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Tempo        : {current_stroke_rate:.1f} SPM", (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Race Time    : {race_time_str}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, race_time_colour, 2)
        if beep_time is not None:
            cv2.putText(frame, f"Reaction Time: {rt_str}", (20, 255),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, rt_color, 2)

        out.write(frame)

    cap.release()
    out.release()

    # ── PANDAS SUMMARY ──────────────────────────────────────────────────────
    finish_ref    = race_finish_time if race_finish_time else (stroke_timestamps[-1] if stroke_timestamps else None)
    clock_origin  = race_start_time  # beep if detected, else first movement
    race_duration = (finish_ref - clock_origin) if (finish_ref and clock_origin) else 0.0
    avg_tempo     = (arm_cycle_count / race_duration) * 60 * 2 if (arm_cycle_count > 0 and race_duration > 0) else 0.0

    if arm_cycle_count > 0 and clock_origin is not None:
        metrics = [
            "Beep Time (s)",
            "First Movement (s)",
            "Reaction Time (s)",
            "Race Start Clock (s)",
            "Race Finish (s)",
            "Race Duration (s)",
            "Arm Cycles Detected",
            "Estimated Total Strokes",
            "Average Tempo (SPM)",
            "Final Tempo (SPM)",
            "Tracking Arm",
        ]
        values = [
            round(beep_time, 3)           if beep_time           is not None else "Not detected",
            round(first_movement_time, 3) if first_movement_time is not None else "Not detected",
            round(reaction_time, 3)       if reaction_time       is not None else "N/A",
            round(clock_origin, 2),
            round(race_finish_time, 2)    if race_finish_time    is not None else "Not detected",
            round(race_duration, 2),
            arm_cycle_count,
            arm_cycle_count * 2,
            round(avg_tempo, 1),
            round(current_stroke_rate, 1),
            "both",
        ]
        df = pd.DataFrame({"Metric": metrics, "Value": values})
        print("\n" + "=" * 55)
        print("  RACE EXECUTION SUMMARY")
        print("=" * 55)
        print(df.to_string(index=False))
        print("=" * 55 + "\n")
    else:
        print("\nNo strokes or race boundaries detected.\n")

    print(f"Analysis complete → '{output_video_path}'")

    return {
        'beep_time':               beep_time,
        'beep_confidence':         round(beep_confidence, 2),
        'first_movement_time':     first_movement_time,
        'reaction_time_s':         reaction_time,
        'race_start_time':         clock_origin,
        'race_finish_time':        race_finish_time,
        'race_duration_s':         round(race_duration, 2),
        'arm_cycle_count':         arm_cycle_count,
        'estimated_total_strokes': arm_cycle_count * 2,
        'average_tempo_spm':       round(avg_tempo, 1),
        'final_tempo_spm':         round(current_stroke_rate, 1),
        'tracking_arm':            'both',
        'start_threshold_used':    round(start_threshold, 5),
        'output_file':             output_video_path,
    }


# ─────────────────────────────────────────────────────────────────────────── #
#  AUTO-RETRY MAIN                                                             #
#  Tries progressively looser finish parameters until both start AND finish   #
#  are detected. Saves analysis_status.json after every attempt so you can    #
#  check progress from the Cursor iOS app or any file viewer.                 #
# ─────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    INPUT_VIDEO = "my_race.mp4"
    BASE_NAME   = "analyzed_race_output"
    EXT         = ".mp4"

    # Each entry loosens one constraint at a time.
    # Attempt 1 uses the tightest (most accurate) settings.
    # Later attempts widen the finish window if needed.
    RETRY_CONFIGS = [
        # 1 — baseline settings
        {'min_race_duration': 44.0, 'finish_velocity_threshold': 0.015, 'finish_still_min_frames': 8},
        # 2 — slightly shorter guard + more lenient velocity
        {'min_race_duration': 40.0, 'finish_velocity_threshold': 0.020, 'finish_still_min_frames': 8},
        # 3 — allow camera to slow gradually after touch
        {'min_race_duration': 38.0, 'finish_velocity_threshold': 0.025, 'finish_still_min_frames': 6},
        # 4 — most lenient; catches any sustained slowdown after the race
        {'min_race_duration': 35.0, 'finish_velocity_threshold': 0.035, 'finish_still_min_frames': 5},
    ]

    # Find the next free version slot (increments across all runs, not just this session)
    def _next_version():
        v = 1
        while os.path.exists(f"{BASE_NAME}_v{v:02d}{EXT}"):
            v += 1
        return v

    final_result = None
    final_attempt = 0

    for attempt, cfg_override in enumerate(RETRY_CONFIGS, start=1):
        v = _next_version()
        output_path = f"{BASE_NAME}_v{v:02d}{EXT}"

        print(f"\n{'━' * 52}")
        print(f"  ATTEMPT {attempt} / {len(RETRY_CONFIGS)}")
        print(f"  min_race_duration = {cfg_override['min_race_duration']}s   "
              f"finish_vel = {cfg_override['finish_velocity_threshold']}   "
              f"still_frames = {cfg_override['finish_still_min_frames']}")
        print(f"  Output → {output_path}")
        print(f"{'━' * 52}")

        result = analyze_swim_video(INPUT_VIDEO, output_path, cfg_override)
        final_result  = result
        final_attempt = attempt

        start_ok  = result is not None and result['race_start_time']  is not None
        finish_ok = result is not None and result['race_finish_time'] is not None

        print(f"\n  ▸ Start:  {'✓ detected' if start_ok  else '✗ not found'}")
        print(f"  ▸ Finish: {'✓ detected' if finish_ok else '✗ not found'}")

        # Write status file after every attempt so iOS / remote monitoring works
        status = {
            'status':             'success' if (start_ok and finish_ok) else 'partial' if (start_ok or finish_ok) else 'failed',
            'attempt':            attempt,
            'total_attempts':     len(RETRY_CONFIGS),
            'start_detected':     start_ok,
            'finish_detected':    finish_ok,
            'beep_time_s':        result.get('beep_time')           if result else None,
            'beep_confidence':    result.get('beep_confidence')     if result else None,
            'first_movement_s':   result.get('first_movement_time') if result else None,
            'reaction_time_s':    result.get('reaction_time_s')     if result else None,
            'race_start_s':       result.get('race_start_time')     if result else None,
            'race_finish_s':      result.get('race_finish_time')    if result else None,
            'race_duration_s':    result.get('race_duration_s')     if result else None,
            'arm_cycles':         result.get('arm_cycle_count')     if result else None,
            'estimated_strokes':  result.get('estimated_total_strokes') if result else None,
            'avg_tempo_spm':      result.get('average_tempo_spm')   if result else None,
            'output_file':        output_path,
            'last_updated':       datetime.now().isoformat(),
        }
        with open('analysis_status.json', 'w') as f:
            json.dump(status, f, indent=2)
        print(f"  Status saved → analysis_status.json")

        if start_ok and finish_ok:
            print(f"\n  Both detections confirmed on attempt {attempt}. Done.")
            break
        elif attempt < len(RETRY_CONFIGS):
            print(f"\n  Retrying with looser parameters…")

    if final_result and not (final_result.get('race_finish_time')):
        print(f"\n  ⚠  Finish not detected after {final_attempt} attempts.")
        print("  The race window defaulted to last detected stroke for the summary.")
