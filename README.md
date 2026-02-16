# Music Grab

Discord bot with two main features:

## Features

### 1. YouTube Audio Stem Separation
Paste a YouTube URL in Discord, and the bot will:
- Download the audio via `yt-dlp`
- Separate it into 4 stems using [Demucs](https://github.com/facebookresearch/demucs) (htdemucs model):
  - Vocals
  - Drums
  - Bass
  - Other instruments
- Send back 4 MP3 files

### 2. Document to Speech (PDF/DOCX)
Upload a `.pdf` or `.docx` file, and the bot will:
- Extract text (`python-docx` / `pdfplumber`)
- Summarize with GPT-4o-mini
- Convert to speech with OpenAI TTS
- Send back an MP3

## Setup

### Dependencies
```bash
pip install discord.py openai python-docx pdfplumber yt-dlp demucs librosa pretty_midi numpy
```

[LilyPond](https://lilypond.org/) is optional (not currently used in output).

### Environment Variables
Set these in your shell profile (e.g. `~/.zshrc`):
```bash
export DISCORD_BOT_TOKEN='your-discord-bot-token'
export OPENAI_API_KEY='your-openai-api-key'
```

### Run
```bash
python discord_tts_bot.py
```

## File Structure
- `discord_tts_bot.py` — Main bot logic
- `newslib.py` — Text extraction and chunking helpers
