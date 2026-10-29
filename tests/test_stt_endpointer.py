import numpy as np

import STTModule

FRAME_S = 0.032


def _endpointer(**overrides):
    params = dict(frame_s=FRAME_S, threshold=0.5, end_silence_s=0.32, min_speech_s=0.096,
                  pre_roll_s=0.064, start_timeout_s=1.0, max_s=3.0)
    params.update(overrides)
    return STTModule.Endpointer(**params)


def _frame(value):
    return np.full(STTModule.FRAME_LENGTH, value, dtype=np.int16)


def _run(ep, probs):
    """Feed one frame per probability (frame value = index, so audio can be traced)."""
    status = None
    for i, p in enumerate(probs):
        status = ep.feed(_frame(i), p)
        if status in ("done", "timeout"):
            return status, i
    return status, len(probs) - 1


def test_times_out_when_nobody_speaks():
    status, _ = _run(_endpointer(), [0.0] * 100)
    assert status == "timeout"


def test_short_blips_do_not_start_a_command():
    # 2-frame blips (64 ms) are below min_speech (96 ms)
    status, _ = _run(_endpointer(), [0.9, 0.9, 0.0] * 20)
    assert status == "timeout"


def test_command_ends_after_trailing_silence():
    ep = _endpointer()
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 20
    status, idx = _run(ep, probs)
    assert status == "done"
    # 10 speech frames, then end_silence (10 frames) of quiet
    assert idx == 5 + 10 + 10 - 1
    assert ep.started


def test_pauses_shorter_than_end_silence_keep_listening():
    probs = [0.9] * 5 + [0.0] * 5 + [0.9] * 5 + [0.0] * 20
    status, idx = _run(_endpointer(), probs)
    assert status == "done"
    assert idx > 15  # did not stop during the mid-sentence pause


def test_hysteresis_keeps_borderline_frames_as_speech():
    # 0.4 is below the start threshold (0.5) but above the silence threshold (0.35)
    probs = [0.9] * 5 + [0.4] * 30 + [0.0] * 20
    status, idx = _run(_endpointer(), probs)
    assert status == "done" and idx >= 35


def test_max_length_cap():
    status, idx = _run(_endpointer(max_s=1.0), [0.9] * 200)
    assert status == "done"
    assert (idx + 1) * FRAME_S >= 1.0


def test_audio_includes_pre_roll_before_speech():
    ep = _endpointer()
    _run(ep, [0.0] * 10 + [0.9] * 5 + [0.0] * 20)
    audio = ep.audio()
    first_frame_value = int(audio[0])
    # Speech starts at frame 10; pre-roll keeps a couple of frames before it
    assert first_frame_value < 10
    assert len(audio) % STTModule.FRAME_LENGTH == 0


def test_energy_vad_fallback():
    vad = STTModule.EnergyVAD(rms_threshold=500)
    assert vad.prob(_frame(0)) == 0.0
    assert vad.prob(_frame(1000)) == 1.0


def test_silero_vad_rejects_silence():
    vad = STTModule.SileroVAD()
    vad.reset()
    assert vad.prob(_frame(0)) < 0.5
