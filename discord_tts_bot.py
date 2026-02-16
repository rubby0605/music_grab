#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Bot：
  1. 上傳 PDF/DOCX → AI 摘要 → MP3 語音
  2. 貼 YouTube URL → 鋼琴樂譜 PDF + MIDI
"""

import os
import re
import tempfile
import asyncio
import functools
import shutil
import subprocess
import discord
from openai import OpenAI
from newslib import read_docx, read_pdf, chunk_text
import pretty_midi
import numpy as np
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate as PT_SR, load_audio

DISCORD_TOKEN = os.environ.get('DISCORD_BOT_TOKEN', '')
SUPPORTED_EXT = ('.pdf', '.docx')
DISCORD_MAX_BYTES = 8 * 1024 * 1024  # 未加成伺服器上限 8MB
YT_URL_PATTERN = re.compile(
    r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+'
)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# ── TTS 功能 ──────────────────────────────

def summarize_text(text):
    openai_client = OpenAI()
    max_input = 12000
    system_msg = (
        '你是一位專業的文件摘要助理。'
        '請將以下文件內容消化整理成適合語音播報的文稿。'
        '要求：保留重要資訊和關鍵數據、去除格式雜訊、口語化、繁體中文、不加前言。'
    )
    if len(text) > max_input:
        segments = [text[i:i+max_input] for i in range(0, len(text), max_input)]
        summaries = []
        for i, seg in enumerate(segments):
            resp = openai_client.chat.completions.create(
                model='gpt-4o-mini', temperature=0.3,
                messages=[
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user', 'content': f'第 {i+1}/{len(segments)} 部分：\n\n{seg}'}
                ])
            summaries.append(resp.choices[0].message.content)
        return '\n\n'.join(summaries)
    else:
        resp = openai_client.chat.completions.create(
            model='gpt-4o-mini', temperature=0.3,
            messages=[{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': text}])
        return resp.choices[0].message.content


def tts_single_chunk(text, output_path, voice='alloy', model='tts-1'):
    openai_client = OpenAI()
    response = openai_client.audio.speech.create(model=model, voice=voice, input=text)
    response.stream_to_file(output_path)


def convert_all_chunks(chunks, tmpdir, voice='alloy', model='tts-1'):
    part_paths = []
    for i, chunk in enumerate(chunks):
        part_path = os.path.join(tmpdir, f'part_{i}.mp3')
        tts_single_chunk(chunk, part_path, voice=voice, model=model)
        part_paths.append(part_path)
    merged_path = os.path.join(tmpdir, 'merged.mp3')
    with open(merged_path, 'wb') as out:
        for p in part_paths:
            with open(p, 'rb') as f:
                out.write(f.read())
    return merged_path, part_paths


# ── 樂譜功能（OpenAI GPT-4o Audio）────────

def yt_get_title(url):
    try:
        r = subprocess.run(['yt-dlp', '--get-title', '--no-playlist', url],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip()[:60] or 'Untitled'
    except Exception:
        return 'Untitled'


def yt_download_video(url, tmpdir):
    """YouTube → MP4 影片"""
    output_path = os.path.join(tmpdir, 'video.%(ext)s')
    r = subprocess.run(
        ['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
         '--merge-output-format', 'mp4',
         '-o', output_path, '--no-playlist', url],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'yt-dlp 失敗: {r.stderr[-300:]}')
    for f in os.listdir(tmpdir):
        if f.startswith('video.'):
            return os.path.join(tmpdir, f)
    raise RuntimeError('下載失敗：找不到影片檔')


def yt_download_mp3(url, tmpdir):
    """YouTube → MP3（不分離音軌，128kbps）"""
    output_path = os.path.join(tmpdir, 'raw.%(ext)s')
    r = subprocess.run(
        ['yt-dlp', '-x', '--audio-format', 'wav',
         '-o', output_path, '--no-playlist', url],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f'yt-dlp 失敗: {r.stderr[-300:]}')
    raw_path = None
    for f in os.listdir(tmpdir):
        if f.startswith('raw.'):
            raw_path = os.path.join(tmpdir, f)
            break
    if not raw_path:
        raise RuntimeError('下載失敗：找不到音頻檔')

    mp3_path = os.path.join(tmpdir, 'audio.mp3')
    subprocess.run(
        ['ffmpeg', '-y', '-i', raw_path, '-b:a', '128k', mp3_path],
        capture_output=True, timeout=120)
    if not os.path.exists(mp3_path):
        raise RuntimeError('ffmpeg 轉檔失敗')
    return mp3_path


GH_REPO = 'rubby0605/music_grab'
GH_RELEASE_TAG = 'mp3-files'


def upload_to_github(filepath, filename):
    """上傳檔案到 GitHub Release，回傳下載連結"""
    gh_token = os.environ.get('GH_TOKEN', '')
    if not gh_token:
        raise RuntimeError('未設定 GH_TOKEN')

    # 複製檔案並重新命名
    upload_path = os.path.join(os.path.dirname(filepath), filename)
    if upload_path != filepath:
        shutil.copy2(filepath, upload_path)

    # 上傳（--clobber 覆蓋同名檔案）
    r = subprocess.run(
        ['gh', 'release', 'upload', GH_RELEASE_TAG, upload_path,
         '--repo', GH_REPO, '--clobber'],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, 'GH_TOKEN': gh_token})
    if r.returncode != 0:
        raise RuntimeError(f'GitHub 上傳失敗: {r.stderr[-200:]}')

    from urllib.parse import quote
    safe_fn = quote(filename)
    return f'https://github.com/{GH_REPO}/releases/download/{GH_RELEASE_TAG}/{safe_fn}'


def transcribe_to_midi(audio_path, output_midi):
    """用 ByteDance Piano Transcription 轉 MIDI"""
    print('  Loading audio for transcription...')
    audio, _ = load_audio(audio_path, sr=PT_SR, mono=True)
    print(f'  Audio: {len(audio)/PT_SR:.1f}s')
    print('  Running Piano Transcription model...')
    transcriptor = PianoTranscription(device='cpu')
    transcriptor.transcribe(audio, output_midi)
    print(f'  MIDI saved: {output_midi}')
    return output_midi


def midi_pipeline(url, tmpdir):
    """YouTube → demucs → Piano Transcription → MIDI (vocals + other)"""
    title = yt_get_title(url)
    print(f'  Title: {title}')

    # 1. Download
    wav = yt_download_audio(url, tmpdir)

    # 2. Demucs separate
    stems = separate_stems(wav, tmpdir)

    other_path = stems.get('other')
    vocals_path = stems.get('vocals')
    if not other_path:
        raise RuntimeError('demucs 未產出 other 軌')

    # 3. Transcribe other (instruments)
    print('  Transcribing instruments...')
    other_midi = os.path.join(tmpdir, 'other.mid')
    transcribe_to_midi(other_path, other_midi)

    # 4. Transcribe vocals
    vocals_midi = None
    if vocals_path:
        print('  Transcribing vocals...')
        vocals_midi = os.path.join(tmpdir, 'vocals.mid')
        transcribe_to_midi(vocals_path, vocals_midi)

    return other_midi, vocals_midi, title


def merge_midi(other_midi, vocals_midi, output_path):
    """合併人聲和伴奏 MIDI，人聲用獨立 track"""
    other = pretty_midi.PrettyMIDI(other_midi)

    # 重新命名 other 的 instrument
    for inst in other.instruments:
        if not inst.is_drum:
            inst.name = 'Piano'

    if vocals_midi:
        vocals = pretty_midi.PrettyMIDI(vocals_midi)
        for inst in vocals.instruments:
            if not inst.is_drum:
                inst.name = 'Vocals'
                inst.program = 73  # Flute (近似人聲的音色)
                other.instruments.append(inst)

    other.write(output_path)
    return output_path


def pdf_pipeline(url, tmpdir):
    """YouTube → demucs → Piano Transcription → MIDI → LilyPond → PDF"""
    other_midi, vocals_midi, title = midi_pipeline(url, tmpdir)

    # 合併 MIDI
    merged_midi = os.path.join(tmpdir, 'merged.mid')
    merge_midi(other_midi, vocals_midi, merged_midi)

    # MIDI → LilyPond（含人聲譜）
    ly_path = os.path.join(tmpdir, 'score.ly')
    print('  Converting MIDI to LilyPond...')
    midi_to_lilypond_full(other_midi, vocals_midi, ly_path, title=title)
    print('  LilyPond file created, compiling PDF...')

    # LilyPond → PDF
    pdf_path = lilypond_to_pdf(ly_path, tmpdir)
    print(f'  PDF done: {pdf_path}')
    return pdf_path, merged_midi, title


def yt_download_audio(url, tmpdir):
    output_path = os.path.join(tmpdir, 'audio.%(ext)s')
    r = subprocess.run(
        ['yt-dlp', '-x', '--audio-format', 'wav', '--audio-quality', '0',
         '-o', output_path, '--no-playlist', url],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f'yt-dlp 失敗: {r.stderr[-300:]}')
    for f in os.listdir(tmpdir):
        if f.startswith('audio.'):
            return os.path.join(tmpdir, f)
    raise RuntimeError('下載失敗：找不到音頻檔')


def separate_stems(wav_path, tmpdir):
    """用 demucs 分離 4 軌：vocals, drums, bass, other"""
    print('  Separating stems with demucs (4 stems)...')
    r = subprocess.run(
        ['python', '-m', 'demucs', '-o', tmpdir, '--mp3', wav_path],
        capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(f'  demucs warning: {r.stderr[-200:]}')

    base = os.path.splitext(os.path.basename(wav_path))[0]
    stems = {}
    stem_names = ['vocals', 'drums', 'bass', 'other']

    # 找 demucs 輸出目錄
    for model_dir in ['htdemucs', 'htdemucs_ft', 'mdx_extra']:
        stem_dir = os.path.join(tmpdir, model_dir, base)
        if os.path.isdir(stem_dir):
            for name in stem_names:
                for ext in ['.mp3', '.wav']:
                    p = os.path.join(stem_dir, name + ext)
                    if os.path.exists(p):
                        stems[name] = p
                        print(f'    {name}: {os.path.getsize(p)/1024:.0f} KB')
            break

    if not stems:
        raise RuntimeError('demucs 未產出分離音軌')

    print(f'  分離完成，共 {len(stems)} 軌')
    return stems


def detect_key(y, sr):
    """用 chroma + Krumhansl-Kessler 偵測調性"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    best_corr, best_key, best_mode = -2, 0, 'major'

    for shift in range(12):
        rolled = np.roll(chroma_avg, -shift)
        corr_maj = np.corrcoef(rolled, major_profile)[0, 1]
        corr_min = np.corrcoef(rolled, minor_profile)[0, 1]
        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, shift, 'major'
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, shift, 'minor'

    return f'{NAMES[best_key]} {best_mode}'


def detect_bpm(y, sr):
    """偵測 BPM"""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = tempo[0]
    return int(round(tempo))


def audio_to_midi(audio_path, tmpdir):
    """用 librosa CQT + onset detection 從乾淨音軌轉 MIDI"""
    print('  Loading audio for transcription...')
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # 偵測調性和 BPM
    key_str = detect_key(y, sr)
    bpm = detect_bpm(y, sr)
    print(f'  Key: {key_str}, BPM: {bpm}')

    hop = 512
    # onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    print(f'  偵測到 {len(onset_times)} 個 onset')

    # CQT 頻譜（C1~C8, 84 bins）
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop,
                            fmin=librosa.note_to_hz('C1'), n_bins=84))
    times = librosa.times_like(C, sr=sr, hop_length=hop)

    # 動態門檻
    threshold = np.percentile(C, 92)

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name='Piano')

    for idx in range(len(onset_times)):
        t_start = onset_times[idx]
        t_end = onset_times[idx + 1] if idx + 1 < len(onset_times) else times[-1]

        f_start = np.searchsorted(times, t_start)
        f_end = np.searchsorted(times, t_end)
        if f_end <= f_start:
            f_end = f_start + 1
        if f_start >= C.shape[1]:
            continue

        segment = C[:, f_start:min(f_end, C.shape[1])]
        avg_energy = segment.mean(axis=1)

        active_bins = np.where(avg_energy > threshold)[0]
        if len(active_bins) == 0:
            continue

        # 最多 6 個同時音
        if len(active_bins) > 6:
            top_idx = np.argsort(avg_energy[active_bins])[-6:]
            active_bins = active_bins[top_idx]

        for b in active_bins:
            midi_pitch = b + 24  # C1 = MIDI 24
            midi_pitch = max(21, min(108, midi_pitch))
            dur = t_end - t_start
            if dur < 0.03:
                continue
            vel = int(min(127, max(30, 40 + avg_energy[b] / threshold * 40)))
            note = pretty_midi.Note(velocity=vel, pitch=midi_pitch,
                                     start=t_start, end=t_end)
            piano.notes.append(note)

    if not piano.notes:
        raise RuntimeError('無法偵測到音符')

    midi_data.instruments.append(piano)
    midi_path = os.path.join(tmpdir, 'transcribed.mid')
    midi_data.write(midi_path)
    print(f'  MIDI: {len(piano.notes)} 個音符')

    meta = {'key': key_str, 'bpm': bpm}
    return midi_path, meta


def midi_to_lilypond(midi_path, ly_path, title='Piano', key_info=None):
    """MIDI → LilyPond 樂譜"""
    midi = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)
    if not all_notes:
        raise RuntimeError('MIDI 中沒有音符')

    all_notes.sort(key=lambda n: n.start)

    tempo_times, tempos = midi.get_tempo_changes()
    bpm = int(tempos[0]) if len(tempos) > 0 else 120

    # 解析調性
    key_str = (key_info or {}).get('key', 'C major')
    use_flats = any(x in key_str for x in ['b ', 'flat', 'Bb', 'Eb', 'Ab', 'Db', 'Gb',
                                             'F major', 'D minor', 'G minor', 'C minor',
                                             'F minor', 'Bb', 'Eb'])

    SHARP_NAMES = ['c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b']
    FLAT_NAMES = ['c', 'des', 'd', 'ees', 'e', 'f', 'ges', 'g', 'aes', 'a', 'bes', 'b']
    note_names = FLAT_NAMES if use_flats else SHARP_NAMES

    # LilyPond 調名
    KEY_MAP = {
        'C major': 'c \\major', 'G major': 'g \\major', 'D major': 'd \\major',
        'A major': 'a \\major', 'E major': 'e \\major', 'B major': 'b \\major',
        'F major': 'f \\major', 'Bb major': 'bes \\major', 'Eb major': 'ees \\major',
        'Ab major': 'aes \\major', 'Db major': 'des \\major', 'Gb major': 'ges \\major',
        'F# major': 'fis \\major', 'C# major': 'cis \\major',
        'A minor': 'a \\minor', 'E minor': 'e \\minor', 'B minor': 'b \\minor',
        'F# minor': 'fis \\minor', 'C# minor': 'cis \\minor', 'G# minor': 'gis \\minor',
        'D minor': 'd \\minor', 'G minor': 'g \\minor', 'C minor': 'c \\minor',
        'F minor': 'f \\minor', 'Bb minor': 'bes \\minor', 'Eb minor': 'ees \\minor',
    }
    ly_key = KEY_MAP.get(key_str, 'c \\major')

    # 拍號
    ts = (key_info or {}).get('time_signature', [4, 4])
    time_sig = f'{ts[0]}/{ts[1]}' if isinstance(ts, list) and len(ts) == 2 else '4/4'

    def pitch_to_lily(p):
        octave = (p // 12) - 1
        name = note_names[p % 12]
        diff = octave - 3
        if diff > 0:
            name += "'" * diff
        elif diff < 0:
            name += "," * abs(diff)
        return name

    def dur_to_lily(dur_sec):
        beat = 60.0 / bpm
        beats = dur_sec / beat
        if beats >= 3.5: return '1'
        elif beats >= 1.75: return '2'
        elif beats >= 0.875: return '4'
        elif beats >= 0.4375: return '8'
        elif beats >= 0.21875: return '16'
        else: return '32'

    right = [n for n in all_notes if n.pitch >= 60]
    left = [n for n in all_notes if n.pitch < 60]

    def to_str(nl):
        if not nl: return 'r1 r1 r1 r1'
        return ' '.join(f'{pitch_to_lily(n.pitch)}{dur_to_lily(n.end - n.start)}' for n in nl)

    safe_title = re.sub(r'[\\\"{}]', '', title)
    ly = f'''\\version "2.24.0"
\\header {{
  title = "{safe_title}"
  subtitle = "AI-transcribed from YouTube — {key_str}"
  tagline = "Generated by Discord Sheet Music Bot"
}}
\\score {{
  \\new PianoStaff <<
    \\new Staff = "right" {{
      \\clef treble
      \\key {ly_key}
      \\time {time_sig}
      \\tempo 4 = {bpm}
      {{ {to_str(right)} }}
    }}
    \\new Staff = "left" {{
      \\clef bass
      \\key {ly_key}
      \\time {time_sig}
      {{ {to_str(left)} }}
    }}
  >>
  \\layout {{ }}
}}
'''
    with open(ly_path, 'w') as f:
        f.write(ly)


def midi_to_lilypond_full(other_midi, vocals_midi, ly_path, title='Score'):
    """MIDI → LilyPond 總譜（人聲 + 鋼琴左右手）"""

    def get_notes(midi_path):
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        return notes, midi

    other_notes, other_pm = get_notes(other_midi)

    # BPM
    tempo_times, tempos = other_pm.get_tempo_changes()
    bpm = int(tempos[0]) if len(tempos) > 0 else 120

    SHARP_NAMES = ['c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b']

    def pitch_to_lily(p):
        octave = (p // 12) - 1
        name = SHARP_NAMES[p % 12]
        diff = octave - 3
        if diff > 0:
            name += "'" * diff
        elif diff < 0:
            name += "," * abs(diff)
        return name

    def dur_to_lily(dur_sec):
        beat = 60.0 / bpm
        beats = dur_sec / beat
        if beats >= 3.5: return '1'
        elif beats >= 1.75: return '2'
        elif beats >= 0.875: return '4'
        elif beats >= 0.4375: return '8'
        elif beats >= 0.21875: return '16'
        else: return '32'

    def to_str(nl):
        if not nl:
            return 'r1 r1 r1 r1'
        return ' '.join(f'{pitch_to_lily(n.pitch)}{dur_to_lily(n.end - n.start)}' for n in nl)

    # Piano: split by middle C
    right = [n for n in other_notes if n.pitch >= 60]
    left = [n for n in other_notes if n.pitch < 60]

    # Vocals
    vocal_str = 'r1 r1 r1 r1'
    if vocals_midi:
        vocal_notes, _ = get_notes(vocals_midi)
        if vocal_notes:
            vocal_str = to_str(vocal_notes)

    safe_title = re.sub(r'[\\\"{}]', '', title)
    ly = f'''\\version "2.24.0"
\\header {{
  title = "{safe_title}"
  subtitle = "AI-transcribed from YouTube"
  tagline = "Generated by Music Grab Bot"
}}
\\score {{
  <<
    \\new Staff = "vocals" \\with {{ instrumentName = "Vocals" }} {{
      \\clef treble
      \\tempo 4 = {bpm}
      {{ {vocal_str} }}
    }}
    \\new PianoStaff \\with {{ instrumentName = "Piano" }} <<
      \\new Staff = "right" {{
        \\clef treble
        {{ {to_str(right)} }}
      }}
      \\new Staff = "left" {{
        \\clef bass
        {{ {to_str(left)} }}
      }}
    >>
  >>
  \\layout {{ }}
}}
'''
    with open(ly_path, 'w') as f:
        f.write(ly)
    print(f'  LilyPond: vocals={len(vocal_str.split())}, right={len(right)}, left={len(left)} notes')


def lilypond_to_pdf(ly_path, tmpdir):
    r = subprocess.run(
        ['lilypond', '-dno-point-and-click', '-o', os.path.join(tmpdir, 'score'), ly_path],
        capture_output=True, text=True, timeout=600)
    pdf_path = os.path.join(tmpdir, 'score.pdf')
    if not os.path.exists(pdf_path):
        raise RuntimeError(f'LilyPond 編譯失敗: {r.stderr[-500:]}')
    return pdf_path


def sheet_pipeline(url, tmpdir):
    title = yt_get_title(url)
    print(f'  Title: {title}')

    # 1. YouTube → WAV
    wav = yt_download_audio(url, tmpdir)
    print(f'  Audio: {wav}')

    # 2. demucs 分離 4 軌
    stems = separate_stems(wav, tmpdir)

    return stems, title


# ── Discord Events ────────────────────────

@client.event
async def on_ready():
    print(f'Bot 已上線: {client.user}')
    print('  功能: PDF/DOCX→語音, YouTube→樂譜')


@client.event
async def on_message(message):
    if message.author == client.user or message.author.bot:
        return

    content = message.content.strip()
    loop = asyncio.get_event_loop()
    print(f'  [MSG] {message.author}: {content[:80]}')

    # ── !pdf → demucs + Piano Transcription + LilyPond ──
    # ── !help → 功能說明 ──
    if content.lower() in ('!help', '!指令', '!commands'):
        help_text = """**🎵 Music Grab Bot 指令說明**

📥 **下載**
`YouTube URL` → 直接下載 MP3
`!video <URL>` → 下載影片 MP4

🎼 **音樂處理**
`!dep <URL>` → Demucs 分離四軌（人聲/鼓/貝斯/其他）
`!midi <URL>` → 轉 MIDI（人聲+伴奏）
`!pdf <URL>` → 轉樂譜 PDF

🔊 **文件轉語音**
上傳 PDF 或 DOCX → AI 摘要 → MP3 語音

💡 所有 YouTube 指令支援 youtube.com 和 youtu.be 連結"""
        await message.channel.send(help_text)
        return

    # ── 指令判斷 ──
    is_video_cmd = content.lower().startswith('!video')
    is_dep_cmd = content.lower().startswith('!dep')
    is_pdf_cmd = content.lower().startswith('!pdf')
    is_midi_cmd = content.lower().startswith('!midi')
    yt_match = YT_URL_PATTERN.search(content)

    if is_video_cmd and yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        await message.channel.send('下載影片中...')
        tmpdir = tempfile.mkdtemp()
        try:
            title = await loop.run_in_executor(None, yt_get_title, url)
            video_path = await loop.run_in_executor(None, yt_download_video, url, tmpdir)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'video'

            fsize = os.path.getsize(video_path)
            print(f'  Video: {fsize/1024/1024:.1f} MB')
            if fsize <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'**{title}**',
                    file=discord.File(video_path, filename=f'{safe_name}.mp4'))
            else:
                await message.channel.send(f'影片 {fsize/1024/1024:.1f} MB，上傳到 GitHub...')
                import time
                ts_name = f'video_{int(time.time())}.mp4'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, video_path, ts_name)
                await message.channel.send(f'**{title}**\n{dl_url}')
        except Exception as e:
            await message.channel.send(f'下載失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    if is_pdf_cmd and yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        await message.channel.send('開始轉樂譜（下載 → 分離音軌 → 轉譜 → PDF），需要幾分鐘...')
        tmpdir = tempfile.mkdtemp()
        try:
            pdf_path, midi_path, title = await loop.run_in_executor(
                None, pdf_pipeline, url, tmpdir)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'score'
            await message.channel.send(
                content=f'**{title}** 樂譜：',
                file=discord.File(pdf_path, filename=f'{safe_name}.pdf'))
            await message.channel.send(
                content=f'MIDI：',
                file=discord.File(midi_path, filename=f'{safe_name}.mid'))
        except Exception as e:
            await message.channel.send(f'轉譜失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    if is_midi_cmd and yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        await message.channel.send('開始轉 MIDI（下載 → 分離音軌 → 轉譜），需要幾分鐘...')
        tmpdir = tempfile.mkdtemp()
        try:
            other_midi, vocals_midi, title = await loop.run_in_executor(
                None, midi_pipeline, url, tmpdir)
            merged = os.path.join(tmpdir, 'merged.mid')
            merge_midi(other_midi, vocals_midi, merged)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'midi'
            await message.channel.send(
                content=f'**{title}** MIDI（人聲+伴奏）：',
                file=discord.File(merged, filename=f'{safe_name}.mid'))
        except Exception as e:
            await message.channel.send(f'轉譜失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # ── !dep → demucs 分離四軌 ──
    if is_dep_cmd and yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        await message.channel.send('收到 YouTube 連結，開始分離音軌（需要幾分鐘）...')
        tmpdir = tempfile.mkdtemp()
        try:
            stems, title = await loop.run_in_executor(
                None, sheet_pipeline, url, tmpdir)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'audio'

            stem_labels = {
                'vocals': '🎤 人聲', 'drums': '🥁 鼓',
                'bass': '🎸 貝斯', 'other': '🎹 其他樂器'
            }
            for stem_name, stem_path in stems.items():
                label = stem_labels.get(stem_name, stem_name)
                await message.channel.send(
                    content=f'**{title}** — {label}：',
                    file=discord.File(stem_path, filename=f'{safe_name}_{stem_name}.mp3'))
            await message.channel.send('分離完成！')
        except Exception as e:
            await message.channel.send(f'分離失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # ── 直接貼 URL → 下載 MP3 ──
    if yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        await message.channel.send('下載 MP3 中...')
        tmpdir = tempfile.mkdtemp()
        try:
            title = await loop.run_in_executor(None, yt_get_title, url)
            mp3_path = await loop.run_in_executor(None, yt_download_mp3, url, tmpdir)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'audio'
            filename = f'{safe_name}.mp3'

            fsize = os.path.getsize(mp3_path)
            print(f'  MP3: {fsize/1024/1024:.1f} MB')
            if fsize <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'**{title}**',
                    file=discord.File(mp3_path, filename=filename))
            else:
                await message.channel.send(f'檔案 {fsize/1024/1024:.1f} MB，上傳到 GitHub...')
                import time
                ts_name = f'mp3_{int(time.time())}.mp3'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, mp3_path, ts_name)
                await message.channel.send(f'**{title}**\n{dl_url}')
        except Exception as e:
            await message.channel.send(f'下載失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # ── PDF/DOCX → 語音 ──
    for attachment in message.attachments:
        ext = os.path.splitext(attachment.filename)[1].lower()
        if ext not in SUPPORTED_EXT:
            continue

        await message.channel.send(f'收到 `{attachment.filename}`，正在處理中...')
        tmpdir = tempfile.mkdtemp()
        input_path = os.path.join(tmpdir, attachment.filename)
        await attachment.save(input_path)

        try:
            if ext == '.docx':
                text = await loop.run_in_executor(None, read_docx, input_path)
            else:
                text = await loop.run_in_executor(None, read_pdf, input_path)

            if not text.strip():
                await message.channel.send('提取不到文字（可能是掃描檔）。')
                continue

            await message.channel.send('AI 正在消化文件內容...')
            summary = await loop.run_in_executor(None, summarize_text, text)

            chunks = chunk_text(summary)
            base_name = os.path.splitext(attachment.filename)[0]
            await message.channel.send(f'摘要完成（{len(summary)} 字），轉換語音中...')

            merged_path, part_paths = await loop.run_in_executor(
                None, functools.partial(convert_all_chunks, chunks, tmpdir))

            merged_size = os.path.getsize(merged_path)
            if merged_size <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'`{attachment.filename}` 的語音摘要：',
                    file=discord.File(merged_path, filename=f'{base_name}.mp3'))
            else:
                await message.channel.send(f'合併檔太大，分段上傳...')
                for i, pp in enumerate(part_paths):
                    await message.channel.send(
                        content=f'語音 ({i+1}/{len(part_paths)})',
                        file=discord.File(pp, filename=f'{base_name}_part{i+1}.mp3'))

            await message.channel.send('轉換完成！')

        except Exception as e:
            await message.channel.send(f'轉換失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        print('請設定環境變數 DISCORD_BOT_TOKEN')
        exit(1)
    client.run(DISCORD_TOKEN)
