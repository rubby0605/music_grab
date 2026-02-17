#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Bot：
  1. 上傳 PDF/DOCX → AI 摘要 → MP3 語音
  2. 貼 YouTube URL → 鋼琴樂譜 PDF + MIDI
"""

import os
import re
import time
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
VIDEO_EXT = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
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


def vocals_to_midi_pyin(audio_path, output_midi):
    """用 librosa pyin 將單音人聲轉成 MIDI（比 Piano Transcription 準確）"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'), sr=sr)
    times = librosa.times_like(f0, sr=sr)

    midi_data = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(program=0, name='Vocals')

    in_note = False
    note_start = 0.0
    note_pitch = 0

    for i in range(len(f0)):
        if voiced_flag[i] and f0[i] is not None and not np.isnan(f0[i]):
            midi_pitch = int(round(librosa.hz_to_midi(f0[i])))
            midi_pitch = max(21, min(108, midi_pitch))
            if not in_note:
                in_note = True
                note_start = times[i]
                note_pitch = midi_pitch
            elif abs(midi_pitch - note_pitch) > 1:
                dur = times[i] - note_start
                if dur > 0.05:
                    voice.notes.append(pretty_midi.Note(
                        velocity=80, pitch=note_pitch,
                        start=note_start, end=times[i]))
                note_start = times[i]
                note_pitch = midi_pitch
        else:
            if in_note:
                dur = times[i] - note_start
                if dur > 0.05:
                    voice.notes.append(pretty_midi.Note(
                        velocity=80, pitch=note_pitch,
                        start=note_start, end=times[i]))
                in_note = False

    if in_note and len(times) > 0:
        dur = times[-1] - note_start
        if dur > 0.05:
            voice.notes.append(pretty_midi.Note(
                velocity=80, pitch=note_pitch,
                start=note_start, end=times[-1]))

    midi_data.instruments.append(voice)
    midi_data.write(output_midi)
    print(f'  pyin: {len(voice.notes)} vocal notes')
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

    # 4. Transcribe vocals (pyin for monophonic)
    vocals_midi = None
    if vocals_path:
        print('  Transcribing vocals (pyin)...')
        vocals_midi = os.path.join(tmpdir, 'vocals.mid')
        vocals_to_midi_pyin(vocals_path, vocals_midi)

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


def pdf_pipeline(url, tmpdir, progress_cb=None):
    """YouTube → demucs → Piano Transcription → Whisper → LilyPond → PDF"""
    def update(msg):
        if progress_cb:
            progress_cb(msg)
        print(f'  {msg}')

    update('⬇️ 下載音頻中...')
    title = yt_get_title(url)
    wav = yt_download_audio(url, tmpdir)

    update('🎛️ Demucs 分離音軌中...')
    stems = separate_stems(wav, tmpdir)

    other_path = stems.get('other')
    vocals_path = stems.get('vocals')

    if not other_path:
        raise RuntimeError('demucs 未產出 other 軌')

    update('🎹 轉譜：伴奏軌...')
    other_midi = os.path.join(tmpdir, 'other.mid')
    transcribe_to_midi(other_path, other_midi)

    vocals_midi = None
    if vocals_path:
        update('🎤 轉譜：人聲軌 (pyin)...')
        vocals_midi = os.path.join(tmpdir, 'vocals.mid')
        vocals_to_midi_pyin(vocals_path, vocals_midi)

    # Whisper 辨識歌詞
    whisper_result = None
    if vocals_path:
        update('📝 Whisper 辨識歌詞中...')
        try:
            whisper_result = whisper_transcribe(vocals_path, tmpdir)
            lyrics_text = whisper_result.text if hasattr(whisper_result, 'text') else ''
            print(f'  Lyrics: {lyrics_text[:80]}...')
        except Exception as e:
            print(f'  Whisper error: {e}')

    # 合併 MIDI
    merged_midi = os.path.join(tmpdir, 'merged.mid')
    merge_midi(other_midi, vocals_midi, merged_midi)

    update('📄 產生樂譜 PDF...')
    ly_path = os.path.join(tmpdir, 'score.ly')
    midi_to_lilypond_full(other_midi, vocals_midi, ly_path, title=title,
                          whisper_result=whisper_result, audio_path=other_path)

    pdf_path = lilypond_to_pdf(ly_path, tmpdir)
    update('✅ 完成！')
    return pdf_path, merged_midi, title


def jianpu_pipeline(url, tmpdir, progress_cb=None):
    """YouTube → demucs → pyin + chords + Whisper → 簡譜 PDF"""
    def update(msg):
        if progress_cb:
            progress_cb(msg)
        print(f'  {msg}')

    update('⬇️ 下載音頻中...')
    title = yt_get_title(url)
    wav = yt_download_audio(url, tmpdir)

    update('🎛️ Demucs 分離音軌中...')
    stems = separate_stems(wav, tmpdir)
    other_path = stems.get('other')
    vocals_path = stems.get('vocals')
    if not vocals_path:
        raise RuntimeError('demucs 未產出 vocals 軌')

    update('🎤 偵測人聲旋律 (pyin)...')
    vocals_midi = os.path.join(tmpdir, 'vocals_jianpu.mid')
    vocals_to_midi_pyin(vocals_path, vocals_midi)

    vm = pretty_midi.PrettyMIDI(vocals_midi)
    vocal_notes = []
    for inst in vm.instruments:
        if not inst.is_drum:
            vocal_notes.extend(inst.notes)
    vocal_notes.sort(key=lambda n: n.start)

    # 偵測調性和 BPM
    ref_audio = other_path or vocals_path
    y_ref, sr_ref = librosa.load(ref_audio, sr=22050, mono=True)
    update('🔍 偵測調性和 BPM...')
    key_str = detect_key(y_ref, sr_ref)
    bpm_val = detect_bpm(y_ref, sr_ref)
    print(f'  Key: {key_str}, BPM: {bpm_val}')

    update('🎸 偵測和弦...')
    chords = detect_chords(ref_audio, bpm_val)

    whisper_result = None
    update('📝 Whisper 辨識歌詞中...')
    try:
        whisper_result = whisper_transcribe(vocals_path, tmpdir)
    except Exception as e:
        print(f'  Whisper error: {e}')

    update('🎼 產生簡譜 PDF...')
    jianpu_bars = format_jianpu(vocal_notes, chords, key_str, bpm_val, whisper_result)
    pdf_path = os.path.join(tmpdir, 'jianpu.pdf')
    render_jianpu_pdf(jianpu_bars, title, key_str, bpm_val, pdf_path)

    update('✅ 完成！')
    return pdf_path, title


def whisper_transcribe(audio_path, tmpdir, language=None):
    """用 OpenAI Whisper API 辨識語音，回傳時間戳。language=None 自動偵測"""
    openai_client = OpenAI()
    compressed = os.path.join(tmpdir, 'whisper_input.mp3')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', '16000', '-b:a', '64k', compressed],
        capture_output=True, timeout=60)
    kwargs = dict(model='whisper-1', response_format='verbose_json',
                  timestamp_granularities=['word', 'segment'])
    if language:
        kwargs['language'] = language
    with open(compressed, 'rb') as f:
        result = openai_client.audio.transcriptions.create(file=f, **kwargs)
    return result


def whisper_to_srt(audio_path, output_srt, tmpdir, language='zh'):
    """音頻 → Whisper → SRT 字幕檔（預設中文）"""
    result = whisper_transcribe(audio_path, tmpdir, language=language)
    segments = result.segments if hasattr(result, 'segments') else []
    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            start = seg['start'] if isinstance(seg, dict) else seg.start
            end = seg['end'] if isinstance(seg, dict) else seg.end
            text = seg['text'] if isinstance(seg, dict) else seg.text
            f.write(f"{i+1}\n")
            f.write(f"{_srt_time(start)} --> {_srt_time(end)}\n")
            f.write(f"{text.strip()}\n\n")
    return output_srt


def _srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def video_add_subtitles(video_path, srt_path, output_path):
    """把 SRT 字幕燒進影片"""
    r = subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-vf', f"subtitles={srt_path}",
         '-c:a', 'copy', output_path],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'ffmpeg 字幕合成失敗: {r.stderr[-200:]}')
    return output_path


def yt_download_audio(url, tmpdir):
    output_path = os.path.join(tmpdir, 'audio.%(ext)s')
    r = subprocess.run(
        ['yt-dlp', '-x', '--audio-format', 'wav', '--audio-quality', '0',
         '-o', output_path, '--no-playlist', url],
        capture_output=True, text=True, timeout=300)
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


def _tokenize(text):
    """CJK 逐字、英文保持整字"""
    tokens = []
    word = ''
    for ch in text:
        if ('\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff'
                or '\u31f0' <= ch <= '\u31ff' or '\uff00' <= ch <= '\uffef'
                or '\u3400' <= ch <= '\u4dbf' or '\uac00' <= ch <= '\ud7af'):
            if word.strip():
                tokens.append(word.strip())
                word = ''
            tokens.append(ch)
        elif ch in (' ', '\n', '\t'):
            if word.strip():
                tokens.append(word.strip())
                word = ''
        else:
            word += ch
    if word.strip():
        tokens.append(word.strip())
    return tokens


def detect_chords(audio_path, bpm):
    """用 chroma 偵測和弦（每小節一個）"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4
    times = librosa.times_like(chroma, sr=sr)
    if len(times) == 0:
        return []
    total_dur = times[-1]

    NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    major_t = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minor_t = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

    chords = []
    t = 0.0
    while t < total_dur:
        t_end = min(t + bar_sec, total_dur)
        mask = (times >= t) & (times < t_end)
        if mask.sum() == 0:
            t = t_end
            continue
        avg = chroma[:, mask].mean(axis=1)
        best, name = -1, 'C'
        for s in range(12):
            for tmpl, suffix in [(major_t, ''), (minor_t, 'm')]:
                rolled = np.roll(tmpl, s)
                denom = np.linalg.norm(avg) * np.linalg.norm(rolled)
                score = np.dot(avg, rolled) / denom if denom > 0 else 0
                if score > best:
                    best, name = score, NAMES[s] + suffix
        chords.append((t, t_end, name))
        t = t_end

    print(f'  Detected {len(chords)} chords')
    return chords


def format_jianpu(vocal_notes, chords, key_str, bpm, whisper_result):
    """人聲音符+和弦+歌詞 → 簡譜格式 (bars list)"""
    root_name = key_str.split()[0]
    ROOT_MAP = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    root_pc = ROOT_MAP.get(root_name, 0)

    is_minor = 'minor' in key_str.lower()
    DEGREE_MAJOR = ['1', '#1', '2', '#2', '3', '4', '#4', '5', '#5', '6', '#6', '7']
    DEGREE_MINOR = ['1', '#1', '2', 'b3', '3', '4', '#4', '5', 'b6', '6', 'b7', '7']
    DEGREE = DEGREE_MINOR if is_minor else DEGREE_MAJOR

    # 主音八度（取人聲中位數所在八度）
    if vocal_notes:
        med = sorted(n.pitch for n in vocal_notes)[len(vocal_notes) // 2]
        home_oct = med // 12
    else:
        home_oct = 5

    def pitch_to_deg(midi_pitch):
        sem = (midi_pitch % 12 - root_pc) % 12
        deg = DEGREE[sem]
        home_base = home_oct * 12 + root_pc
        oct_shift = (midi_pitch - home_base) // 12
        return deg, oct_shift

    beat_sec = 60.0 / bpm
    eighth = beat_sec / 2
    bar_sec = beat_sec * 4

    total_dur = max(
        (max(n.end for n in vocal_notes) if vocal_notes else 0),
        (max(c[1] for c in chords) if chords else 0),
        bar_sec)
    total_bars = int(total_dur / bar_sec) + 1
    slots_total = total_bars * 8

    # 建立 grid
    grid = [{'note': '0', 'oct': 0, 'onset': False, 'lyric': ''}
            for _ in range(slots_total)]

    for note in vocal_notes:
        s_start = int(round(note.start / eighth))
        s_end = int(round(note.end / eighth))
        s_start = max(0, min(s_start, slots_total - 1))
        s_end = max(s_start + 1, min(s_end, slots_total))

        deg, oct_diff = pitch_to_deg(note.pitch)
        grid[s_start] = {'note': deg, 'oct': oct_diff, 'onset': True, 'lyric': ''}
        for s in range(s_start + 1, s_end):
            if s < slots_total and grid[s]['note'] == '0':
                grid[s] = {'note': '-', 'oct': 0, 'onset': False, 'lyric': ''}

    # 歌詞對齊
    if whisper_result:
        segments = []
        raw_segs = getattr(whisper_result, 'segments', [])
        for seg in raw_segs:
            if isinstance(seg, dict):
                segments.append((seg['start'], seg['end'], seg['text'].strip()))
            else:
                segments.append((seg.start, seg.end, seg.text.strip()))

        timed_tokens = []
        for seg_s, seg_e, text in segments:
            tokens = _tokenize(text)
            if not tokens:
                continue
            dt = (seg_e - seg_s) / max(len(tokens), 1)
            for i, tok in enumerate(tokens):
                timed_tokens.append((seg_s + i * dt, tok))

        ci = 0
        for s in range(slots_total):
            if not grid[s]['onset']:
                continue
            t = s * eighth
            while ci < len(timed_tokens) and timed_tokens[ci][0] < t - 0.8:
                ci += 1
            if ci < len(timed_tokens) and abs(timed_tokens[ci][0] - t) < 1.5:
                grid[s]['lyric'] = timed_tokens[ci][1]
                ci += 1

    # 和弦對齊到小節
    bar_chords = [''] * total_bars
    for start, end, name in chords:
        idx = int(start / bar_sec)
        if 0 <= idx < total_bars:
            bar_chords[idx] = name

    # 組裝 bars
    bars = []
    for b in range(total_bars):
        slots = grid[b * 8: b * 8 + 8]
        bars.append({'chord': bar_chords[b], 'slots': slots})

    return bars


def render_jianpu_pdf(bars, title, key_str, bpm, output_pdf):
    """簡譜資料 → PDF（matplotlib）"""
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = [
        'PingFang TC', 'Hiragino Sans', 'Arial Unicode MS',
        'Noto Sans CJK TC', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root_name = key_str.split()[0]
    BARS_PER_LINE = 2

    # A4 page (inches)
    pw, ph = 8.27, 11.69
    ml, mr, mt, mb = 0.8, 0.6, 0.5, 0.5
    usable_w = pw - ml - mr

    bar_w = usable_w / BARS_PER_LINE
    slot_w = bar_w / 8

    chord_h = 0.22
    note_h = 0.30
    lyric_h = 0.25
    group_gap = 0.18
    group_h = chord_h + note_h + lyric_h + group_gap

    title_block_h = 0.9
    lines_p1 = int((ph - mt - mb - title_block_h) / group_h)
    lines_pn = int((ph - mt - mb) / group_h)

    # 去掉尾部空白小節
    while len(bars) > 1 and all(s['note'] == '0' for s in bars[-1]['slots']):
        bars.pop()
    total_lines = (len(bars) + BARS_PER_LINE - 1) // BARS_PER_LINE

    with PdfPages(output_pdf) as pdf:
        li = 0
        pg = 0
        while li < total_lines:
            fig = plt.figure(figsize=(pw, ph))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, pw)
            ax.set_ylim(0, ph)
            ax.axis('off')

            if pg == 0:
                y = ph - mt - 0.15
                ax.text(pw / 2, y, title, ha='center', va='top',
                        fontsize=18, fontweight='bold')
                y -= 0.4
                ax.text(pw / 2, y, f'1={root_name}    {bpm} BPM    4/4',
                        ha='center', va='top', fontsize=12, color='#555')
                y -= 0.45
                max_lines = lines_p1
            else:
                y = ph - mt
                max_lines = lines_pn

            for _ in range(max_lines):
                if li >= total_lines:
                    break

                bs = li * BARS_PER_LINE
                be = min(bs + BARS_PER_LINE, len(bars))
                n_bars = be - bs

                yc = y
                yn = y - chord_h
                yl = y - chord_h - note_h

                # 小節線
                for i in range(n_bars + 1):
                    x = ml + i * bar_w
                    lw = 1.5 if (bs + i == len(bars)) else 0.8
                    ax.plot([x, x], [yc + 0.1, yl - 0.1], 'k-', lw=lw)

                for bi in range(bs, be):
                    bar = bars[bi]
                    bx = ml + (bi - bs) * bar_w

                    if bar['chord']:
                        ax.text(bx + 0.08, yc, bar['chord'],
                                ha='left', va='top', fontsize=10,
                                fontweight='bold', style='italic', color='#333')

                    for si, slot in enumerate(bar['slots']):
                        sx = bx + si * slot_w + slot_w / 2
                        note = slot['note']
                        if note not in ('0', '-'):
                            ax.text(sx, yn, note, ha='center', va='top',
                                    fontsize=14, fontweight='bold')
                            oc = slot['oct']
                            if oc > 0:
                                for d in range(oc):
                                    ax.plot(sx, yn + 0.08 + d * 0.07,
                                            'ko', markersize=3)
                            elif oc < 0:
                                for d in range(-oc):
                                    ax.plot(sx, yn - 0.25 - d * 0.07,
                                            'ko', markersize=3)
                        elif note == '-':
                            ax.text(sx, yn, '-', ha='center', va='top',
                                    fontsize=14, color='#666')
                        else:
                            ax.text(sx, yn, '0', ha='center', va='top',
                                    fontsize=14, color='#ccc')

                        if slot.get('lyric'):
                            ax.text(sx, yl, slot['lyric'],
                                    ha='center', va='top', fontsize=10)

                y -= group_h
                li += 1

            pdf.savefig(fig)
            plt.close(fig)
            pg += 1

    print(f'  Jianpu PDF: {total_lines} lines, {pg} pages')
    return output_pdf


def align_lyrics_to_notes(whisper_result, vocal_notes):
    """用 Whisper 時間戳把歌詞對齊到人聲音符（兩指針掃描）"""
    if not whisper_result or not vocal_notes:
        return []

    segments = []
    raw_segments = getattr(whisper_result, 'segments', [])
    for seg in raw_segments:
        if isinstance(seg, dict):
            segments.append((seg['start'], seg['end'], seg['text'].strip()))
        else:
            segments.append((seg.start, seg.end, seg.text.strip()))

    if not segments:
        return []

    timed_chars = []
    for seg_start, seg_end, text in segments:
        tokens = _tokenize(text)
        if not tokens:
            continue
        tok_dur = (seg_end - seg_start) / max(len(tokens), 1)
        for i, tok in enumerate(tokens):
            timed_chars.append((seg_start + i * tok_dur, tok))

    if not timed_chars:
        return []

    # 兩指針對齊：每個音符找最近的未使用字元
    result = []
    ci = 0
    for note in vocal_notes:
        # 跳過時間上太早的字元
        while ci < len(timed_chars) and timed_chars[ci][0] < note.start - 0.8:
            ci += 1

        if ci < len(timed_chars) and abs(timed_chars[ci][0] - note.start) < 1.5:
            tok = timed_chars[ci][1]
            safe_tok = tok.replace('"', '').replace('\\', '').replace('{', '').replace('}', '')
            result.append(f'"{safe_tok}"' if safe_tok else '_')
            ci += 1
        else:
            result.append('_')

    return result


def midi_to_lilypond_full(other_midi, vocals_midi, ly_path, title='Score',
                          whisper_result=None, audio_path=None):
    """MIDI → LilyPond 總譜（人聲+歌詞 + 鋼琴左右手，含和弦+休止符+調性）"""

    def get_notes(midi_path):
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: (n.start, n.pitch))
        return notes, midi

    other_notes, other_pm = get_notes(other_midi)

    tempo_times, tempos = other_pm.get_tempo_changes()
    bpm = int(tempos[0]) if len(tempos) > 0 else 120
    beat_sec = 60.0 / bpm

    # 調性偵測
    key_str = 'C major'
    if audio_path:
        try:
            y_key, sr_key = librosa.load(audio_path, sr=22050, mono=True)
            key_str = detect_key(y_key, sr_key)
            print(f'  Detected key: {key_str}')
        except Exception as e:
            print(f'  Key detection failed: {e}')

    FLAT_KEYS = {'F major', 'Bb major', 'Eb major', 'Ab major', 'Db major', 'Gb major',
                 'D minor', 'G minor', 'C minor', 'F minor', 'Bb minor', 'Eb minor'}
    use_flats = key_str in FLAT_KEYS

    SHARP_NAMES = ['c', 'cis', 'd', 'dis', 'e', 'f', 'fis', 'g', 'gis', 'a', 'ais', 'b']
    FLAT_NAMES = ['c', 'des', 'd', 'ees', 'e', 'f', 'ges', 'g', 'aes', 'a', 'bes', 'b']
    note_names = FLAT_NAMES if use_flats else SHARP_NAMES

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

    def pitch_to_lily(p):
        octave = (p // 12) - 1
        name = note_names[p % 12]
        diff = octave - 3
        if diff > 0:
            name += "'" * diff
        elif diff < 0:
            name += "," * abs(diff)
        return name

    def beats_to_lily_dur(beats):
        if beats >= 3.5: return '1'
        elif beats >= 1.75: return '2'
        elif beats >= 1.2: return '4.'
        elif beats >= 0.875: return '4'
        elif beats >= 0.625: return '8.'
        elif beats >= 0.4375: return '8'
        elif beats >= 0.21875: return '16'
        else: return '32'

    def fill_rests(beats):
        if beats <= 0.1:
            return ''
        parts = []
        remaining = beats
        for val, name in [(4.0, 'r1'), (2.0, 'r2'), (1.0, 'r4'), (0.5, 'r8'), (0.25, 'r16')]:
            while remaining >= val - 0.05:
                parts.append(name)
                remaining -= val
        return ' '.join(parts)

    def notes_to_str(note_list):
        """音符列表 → LilyPond 字串（含和弦分組+休止符+小節線）"""
        if not note_list:
            return 'R1*4'

        # 按起始時間分組（同時間的音組成和弦）
        CHORD_TOL = 0.05  # 秒
        sorted_notes = sorted(note_list, key=lambda n: (n.start, n.pitch))
        groups = []
        i = 0
        while i < len(sorted_notes):
            group = [sorted_notes[i]]
            j = i + 1
            while j < len(sorted_notes) and abs(sorted_notes[j].start - sorted_notes[i].start) < CHORD_TOL:
                group.append(sorted_notes[j])
                j += 1
            groups.append(group)
            i = j

        bar_dur = 4 * beat_sec  # 一小節的秒數
        next_bar = bar_dur      # 下一個小節線的時間
        parts = ['\\cadenzaOn']
        cursor = 0.0

        for group in groups:
            gap = group[0].start - cursor
            if gap > beat_sec * 0.2:
                gap_beats = gap / beat_sec
                rest = fill_rests(gap_beats)
                if rest:
                    parts.append(rest)

            min_dur = min(n.end - n.start for n in group)
            dur_beats = min_dur / beat_sec
            dur_str = beats_to_lily_dur(dur_beats)

            if len(group) == 1:
                parts.append(f'{pitch_to_lily(group[0].pitch)}{dur_str}')
            else:
                pitches = ' '.join(pitch_to_lily(n.pitch)
                                   for n in sorted(group, key=lambda n: n.pitch))
                parts.append(f'<{pitches}>{dur_str}')

            cursor = max(n.end for n in group)

            # 每隔一小節插入小節線（讓 LilyPond 換行）
            while cursor >= next_bar - 0.05:
                parts.append('\\bar "|"')
                next_bar += bar_dur

        parts.append('\\bar "|."')
        return ' '.join(parts)

    # 用中央 C (MIDI 60) 分左右手
    right = [n for n in other_notes if n.pitch >= 60]
    left = [n for n in other_notes if n.pitch < 60]

    # Vocals
    vocal_notes = []
    if vocals_midi:
        vocal_notes, _ = get_notes(vocals_midi)

    # 歌詞（用 Whisper 時間戳對齊）
    lyrics_block = ''
    if whisper_result and vocal_notes:
        ly_words = align_lyrics_to_notes(whisper_result, vocal_notes)
        # 去掉尾部連續的 _
        while ly_words and ly_words[-1] == '_':
            ly_words.pop()
        if ly_words:
            lyrics_block = '\\new Lyrics \\lyricsto "vox" { ' + ' '.join(ly_words) + ' }'

    safe_title = re.sub(r'[\\\"{}]', '', title)
    ly = f'''\\version "2.24.0"
\\header {{
  title = "{safe_title}"
  subtitle = "AI-transcribed from YouTube — {key_str}"
  tagline = "Generated by Music Grab Bot"
}}
\\paper {{
  #(set-paper-size "a4")
  top-margin = 10
  bottom-margin = 10
  left-margin = 12
  right-margin = 12
}}
\\score {{
  <<
    \\new Staff = "vocals" \\with {{ instrumentName = "Vocals" }} {{
      \\clef treble
      \\key {ly_key}
      \\time 4/4
      \\tempo 4 = {bpm}
      \\new Voice = "vox" {{ {notes_to_str(vocal_notes)} }}
    }}
    {lyrics_block}
    \\new PianoStaff \\with {{ instrumentName = "Piano" }} <<
      \\new Staff = "right" {{
        \\clef treble
        \\key {ly_key}
        \\time 4/4
        {{ {notes_to_str(right)} }}
      }}
      \\new Staff = "left" {{
        \\clef bass
        \\key {ly_key}
        \\time 4/4
        {{ {notes_to_str(left)} }}
      }}
    >>
  >>
  \\layout {{
    indent = 15\\mm
    short-indent = 5\\mm
  }}
}}
'''
    with open(ly_path, 'w') as f:
        f.write(ly)
    n_lyrics = len(getattr(whisper_result, 'text', '')) if whisper_result else 0
    print(f'  LilyPond: vocals={len(vocal_notes)}, RH={len(right)}, LH={len(left)}, lyrics={n_lyrics} chars')


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
`!pdf <URL>` → 轉樂譜 PDF（含歌詞）
`!jianpu <URL>` → 轉簡譜 PDF（數字譜+和弦+歌詞）

📝 **字幕**
上傳 MP4/MKV/AVI → Whisper 自動辨識 → SRT 字幕檔

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
    is_jianpu_cmd = content.lower().startswith('!jianpu')
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

        progress_msg = await message.channel.send('🎵 開始轉樂譜...')
        tmpdir = tempfile.mkdtemp()
        try:
            # 用 progress callback 更新 Discord 訊息
            progress_state = {'msg': progress_msg, 'loop': loop}

            def _update_progress(text):
                asyncio.run_coroutine_threadsafe(
                    progress_state['msg'].edit(content=text),
                    progress_state['loop'])

            pdf_path, midi_path, title = await loop.run_in_executor(
                None, pdf_pipeline, url, tmpdir, _update_progress)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'score'
            # PDF
            pdf_size = os.path.getsize(pdf_path)
            if pdf_size <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'**{title}** 樂譜：',
                    file=discord.File(pdf_path, filename=f'{safe_name}.pdf'))
            else:
                ts_name = f'score_{int(time.time())}.pdf'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, pdf_path, ts_name)
                await message.channel.send(f'**{title}** 樂譜：\n{dl_url}')
            # MIDI
            midi_size = os.path.getsize(midi_path)
            if midi_size <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'MIDI：',
                    file=discord.File(midi_path, filename=f'{safe_name}.mid'))
            else:
                ts_name = f'midi_{int(time.time())}.mid'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, midi_path, ts_name)
                await message.channel.send(f'MIDI：\n{dl_url}')
        except Exception as e:
            await message.channel.send(f'轉譜失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    if is_jianpu_cmd and yt_match:
        url = yt_match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url

        progress_msg = await message.channel.send('🎵 開始轉簡譜...')
        tmpdir = tempfile.mkdtemp()
        try:
            progress_state = {'msg': progress_msg, 'loop': loop}

            def _update_jianpu(text):
                asyncio.run_coroutine_threadsafe(
                    progress_state['msg'].edit(content=text),
                    progress_state['loop'])

            pdf_path, title = await loop.run_in_executor(
                None, jianpu_pipeline, url, tmpdir, _update_jianpu)
            safe_name = re.sub(r'[^\w\s\-]', '', title)[:40] or 'jianpu'

            pdf_size = os.path.getsize(pdf_path)
            if pdf_size <= DISCORD_MAX_BYTES:
                await message.channel.send(
                    content=f'**{title}** 簡譜：',
                    file=discord.File(pdf_path, filename=f'{safe_name}_jianpu.pdf'))
            else:
                ts_name = f'jianpu_{int(time.time())}.pdf'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, pdf_path, ts_name)
                await message.channel.send(f'**{title}** 簡譜：\n{dl_url}')
        except Exception as e:
            await message.channel.send(f'簡譜轉譜失敗：{e}')
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
                ts_name = f'mp3_{int(time.time())}.mp3'
                dl_url = await loop.run_in_executor(
                    None, upload_to_github, mp3_path, ts_name)
                await message.channel.send(f'**{title}**\n{dl_url}')
        except Exception as e:
            await message.channel.send(f'下載失敗：{e}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # ── 上傳 MP4 → Whisper 字幕 ──
    for attachment in message.attachments:
        ext = os.path.splitext(attachment.filename)[1].lower()
        if ext not in VIDEO_EXT:
            continue

        progress_msg = await message.channel.send(f'收到 `{attachment.filename}`，Whisper 字幕產生中...')
        tmpdir = tempfile.mkdtemp()
        video_path = os.path.join(tmpdir, attachment.filename)
        await attachment.save(video_path)

        try:
            # 提取音頻
            audio_path = os.path.join(tmpdir, 'audio.wav')
            await loop.run_in_executor(None, lambda: subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', '16000', audio_path],
                capture_output=True, timeout=120))

            # Whisper
            srt_path = os.path.join(tmpdir, 'subtitles.srt')
            await loop.run_in_executor(None, whisper_to_srt, audio_path, srt_path, tmpdir)

            base_name = os.path.splitext(attachment.filename)[0]
            await message.channel.send(
                content=f'`{attachment.filename}` 字幕：',
                file=discord.File(srt_path, filename=f'{base_name}.srt'))

            await progress_msg.edit(content='✅ 字幕產生完成！')
        except Exception as e:
            await message.channel.send(f'字幕失敗：{e}')
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
