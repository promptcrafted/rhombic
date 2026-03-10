"""
Harmony of the Spheres — Background music derived from corpus data.

The 8 tracked primes of the tessitura map to overtone frequencies.
Card inscription values, when factored, determine which prime-voices
are active at each moment. The Fiedler eigenvalue ratio (2.30x) sets
the pulsation rate. The amplification ratio (6.11x) governs harmonic
enrichment as the piece progresses.

Fundamental frequency: SAMMA = 282 Hz (the Tappetino Proof value).
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from pathlib import Path

SR = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "audio" / "music"
OUT.mkdir(parents=True, exist_ok=True)

# ── The 8 Prime Voices (Tessitura Threads) ──
PRIMES = [11, 17, 19, 23, 29, 31, 67, 89]

# Fundamental: SAMMA = 282 Hz (the Tappetino Proof value)
FUNDAMENTAL = 282.0

# Fiedler Ratio as Rhythmic Pulse (2.30x = one gentle throb per ~3.5s)
FIEDLER_RATIO = 2.30
PULSE_FREQ = FIEDLER_RATIO / 8.0

# Amplification Ratio governs harmonic enrichment over time
AMP_RATIO = 6.11

# ── Card Values (deck order) ──
# Each card carries ALL analytically significant values: combined inscription,
# morpheme sub-values, Hebrew operator derivations, and Colel neighbors.
# The prime threads appear through these operations, not just raw totals.
# Format: (name, [values that matter for prime activation])
CARD_VALUES = [
    ("UAT_ASETEDOJ", [1296, 396, 900, 297]),        # 0: 6^4, UAT=396, ASETEDOJ=900, Shin=300
    ("BRAL_ALEBAL",  [202, 133, 69, 134, 201]),      # 1: BRAL=7x19, ALEBAL=3x23, Colel=2x67
    ("ALBAL_MAT",    [405, 64, 341, 342, 31]),        # 2: MAT=11x31, MAAT=2x3^2x19, AL=31
    ("NIALMATAL",    [463, 91, 341, 31, 460]),         # 3: NIAL=7x13, MAT=11x31, AL=31, -Gimel=460
    ("BANIAL",       [94, 90, 91, 31, 98]),            # 4: -Daleth=90(Anael), NIAL=91, AL=31
    ("ALBAL",        [64, 31, 33, 69, 59]),            # 5: AL=31, +Heh=69=3x23
    ("BARILTE",      [448, 143, 305, 442, 454]),       # 6: BARIL=11x13, TE=305, -Vav=442=2x13x17
    ("ULITAL",       [771, 770, 341, 31, 764]),        # 7: Colel=770=2x5x7x11, MAT-morpheme echo
    ("MAAT",         [342, 341, 19, 31, 334]),         # 8: 2x3^2x19, breath=11x31, -Cheth=334
    ("COMJL",        [153, 17, 152, 144, 162]),        # 9: T(17)=153, 17=prime, -Teth=144
    ("ULE",          [435, 29, 31, 425, 445]),         # 10: 3x5x29, AL echo, -Yod=425
    ("ADONAJ",       [136, 17, 116, 156]),             # 11: 2^3x17, -Kaph=116, +Kaph=156
    ("TELAMJ",       [386, 356, 416, 31]),             # 12: -Lamed=356, +Lamed=416, AL morpheme
    ("VAJNE",        [72, 67, 17, 55, 391, 667, 418]), # 13: VAJN=67, VAJ=17, NE=5x11, 17x23, 23x29, 2x11x19
    ("VELIBAA",      [55, 54, 11, 41, 5]),             # 14: 5x11, breath=54=2x27, VEL=41(prime)
    ("DAJIBAA",      [29, 28, 89]),                    # 15: 29=prime, breath=28, +Samekh=89!
    ("BELIAL",       [78, 31, 47]),                    # T: T(12)=78, AL=31, BEL=47(prime)
    ("DONACE",       [133, 19, 63, 203]),              # 16: 7x19, -Ayin=63=7x3^2, +Ayin=203
    ("GEJ",          [18, 17, 98, 19]),                # 17: Colel=17=prime, -Peh=17 self-ref, +1=19
    ("ECAT",         [309, 219, 399, 23]),             # 18: -Tzaddi=219, +Tzaddi=399, morpheme
    ("ORO",          [240, 140, 340, 239, 241]),       # 19: -Qoph=140, +Qoph=340
    ("TALEJ",        [346, 146, 546, 23, 173]),        # 20: -Resh=146, +Resh=546, 173=prime
    ("GEAREIJ",      [134, 67, 534, 31, 133]),         # 21: 2x67, -Tav=134-400 wraps, AL=31
    ("RIRAJLA",      [252, 90, 31, 42, 210]),          # G: +Anael=342=MAAT, AL=31, AJLA=42
]


def prime_to_freq(p, fundamental=FUNDAMENTAL):
    """Map prime to overtone frequency, normalized to 100-800 Hz."""
    f = p * fundamental
    while f > 800:
        f /= 2
    while f < 100:
        f *= 2
    return f


PRIME_FREQS = {p: prime_to_freq(p) for p in PRIMES}


def factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def active_primes_single(value):
    factors = factorize(value)
    return [p for p in PRIMES if p in factors]


def active_primes(values):
    """Which tracked primes appear across ALL values for a card?"""
    found = set()
    for v in values:
        found.update(active_primes_single(v))
    return sorted(p for p in PRIMES if p in found)


def make_prime_tone(freq, duration, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), False)
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    sig += 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    vib_rate = (freq % 7) + 3
    sig *= 1.0 + 0.004 * np.sin(2 * np.pi * vib_rate * t)
    return sig


def generate_harmony(duration_s=140.0, sr=SR):
    total = int(sr * duration_s)
    t = np.linspace(0, duration_s, total, False)

    track_L = np.zeros(total)
    track_R = np.zeros(total)

    # Layer 1: Bass drone on SAMMA fundamental
    bass = 0.25 * np.sin(2 * np.pi * (FUNDAMENTAL / 4) * t)
    bass += 0.15 * np.sin(2 * np.pi * (FUNDAMENTAL / 2) * t)
    pulse = 0.85 + 0.15 * np.sin(2 * np.pi * PULSE_FREQ * t)
    bass *= pulse
    track_L += bass
    track_R += bass

    # Layer 2: Prime voice pads driven by card factorizations
    chord_duration = 6.0
    crossfade_s = 2.0

    for card_idx in range(len(CARD_VALUES)):
        name, values = CARD_VALUES[card_idx]
        start_s = card_idx * (chord_duration - crossfade_s)
        if start_s >= duration_s:
            break

        active = active_primes(values)
        all_voice_freqs = []
        for p in PRIMES:
            if p in active:
                all_voice_freqs.append((PRIME_FREQS[p], 1.0))
            else:
                all_voice_freqs.append((PRIME_FREQS[p], 0.08))

        progress = card_idx / len(CARD_VALUES)
        enrichment = 1.0 + (AMP_RATIO / FIEDLER_RATIO - 1.0) * progress

        this_dur = min(chord_duration, duration_s - start_s)
        start_idx = int(start_s * sr)
        dur_samples = int(this_dur * sr)

        env = np.ones(dur_samples)
        attack = min(int(sr * 1.5), dur_samples // 3)
        release = min(int(sr * 2.0), dur_samples // 3)
        env[:attack] = np.linspace(0, 1, attack) ** 0.7
        env[-release:] = np.linspace(1, 0, release) ** 0.7

        for voice_idx, (freq, amplitude) in enumerate(all_voice_freqs):
            prime = PRIMES[voice_idx]
            if prime > 31:
                amp = amplitude * enrichment * 0.4
            else:
                amp = amplitude * 0.5

            tone = make_prime_tone(freq, this_dur, sr)
            tone *= env * amp

            pan = (voice_idx / (len(PRIMES) - 1)) * 0.6 + 0.2
            end_idx = min(start_idx + dur_samples, total)
            actual_len = end_idx - start_idx
            track_L[start_idx:end_idx] += tone[:actual_len] * (1 - pan)
            track_R[start_idx:end_idx] += tone[:actual_len] * pan

    # Layer 3: 67-thread shimmer (crystalline high overtone)
    shimmer_freq = PRIME_FREQS[67] * 4
    while shimmer_freq > 6000:
        shimmer_freq /= 2
    shimmer = 0.02 * np.sin(2 * np.pi * shimmer_freq * t)
    shimmer *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    shimmer[:int(sr * 30)] *= np.linspace(0, 1, int(sr * 30))
    track_L += shimmer * 0.6
    track_R += shimmer * 0.4

    # Layer 4: Anael sub-harmonic (90 Hz — the angel of the rose)
    anael = 0.08 * np.sin(2 * np.pi * 90 * t)
    anael *= pulse
    track_L += anael
    track_R += anael

    # Post-processing: warm low-pass (corporate muzak is never harsh)
    b, a = butter(3, 3500 / (sr / 2), btype="low")
    track_L = filtfilt(b, a, track_L)
    track_R = filtfilt(b, a, track_R)

    # Normalize
    peak = max(np.max(np.abs(track_L)), np.max(np.abs(track_R)))
    track_L = track_L / peak * 0.55
    track_R = track_R / peak * 0.55

    # Fade in/out
    fade_in = int(sr * 4)
    fade_out = int(sr * 4)
    track_L[:fade_in] *= np.linspace(0, 1, fade_in)
    track_R[:fade_in] *= np.linspace(0, 1, fade_in)
    track_L[-fade_out:] *= np.linspace(1, 0, fade_out)
    track_R[-fade_out:] *= np.linspace(1, 0, fade_out)

    return np.column_stack([track_L, track_R])


if __name__ == "__main__":
    print("=== Harmony of the Spheres ===")
    print(f"Fundamental: SAMMA = {FUNDAMENTAL} Hz")
    print(f"Pulse: Fiedler {FIEDLER_RATIO}x -> {PULSE_FREQ:.4f} Hz")
    print(f"Enrichment: {FIEDLER_RATIO}x -> {AMP_RATIO}x over duration")
    print()

    print("Prime -> Frequency mapping:")
    for p, f in PRIME_FREQS.items():
        print(f"  {p:3d} -> {f:7.1f} Hz")
    print()

    for name, vals in CARD_VALUES:
        active = active_primes(vals)
        marker = " <<<" if active else ""
        print(f"  {name:15s}  vals={len(vals):2d}  threads: {active}{marker}")
    print()

    stereo = generate_harmony(140.0)
    wav_data = (stereo * 32767).astype(np.int16)
    wavfile.write(str(OUT / "harmony_of_spheres.wav"), SR, wav_data)
    fsize = (OUT / "harmony_of_spheres.wav").stat().st_size
    print(f"\n-> harmony_of_spheres.wav: 140.0s stereo, {fsize // 1024}KB")
