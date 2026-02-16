# Music Grab

Discord bot for YouTube audio download, stem separation, and document-to-speech.

## Commands

### `!mp3 <YouTube URL>`
Download YouTube audio as MP3 (128kbps).
- Small files sent directly in Discord
- Large files (>8MB) uploaded to [GitHub Releases](https://github.com/rubby0605/music_grab/releases/tag/mp3-files) with download link

### YouTube URL (no prefix)
Paste a YouTube URL to separate audio into 4 stems using [Demucs](https://github.com/facebookresearch/demucs):
- Vocals
- Drums
- Bass
- Other instruments

Each stem is sent as an MP3 file.

### Upload PDF / DOCX
Upload a `.pdf` or `.docx` file to get an AI-narrated audio summary:
1. Text extraction via `python-docx` / `pdfplumber`
2. Summarization via GPT-4o-mini
3. Text-to-speech via OpenAI TTS
4. MP3 sent back in Discord

## Setup

### Dependencies
```bash
pip install discord.py openai python-docx pdfplumber yt-dlp demucs librosa pretty_midi numpy
```

### Environment Variables
```bash
export DISCORD_BOT_TOKEN='...'
export OPENAI_API_KEY='...'
export GH_TOKEN='...'   # GitHub PAT for release uploads
```

### Run
```bash
python discord_tts_bot.py
```

## Files
- `discord_tts_bot.py` — Bot main logic
- `newslib.py` — Text extraction and chunking helpers
