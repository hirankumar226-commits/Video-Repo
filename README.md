# 🎬 FaceStudio — AI Video Pipeline

Full pipeline to create AI character videos:
**Face Swap → Voice Clone → Expression Sync**

## Tech Stack
- **Backend**: Python + Flask
- **Face Swap**: Replicate API (inswapper_128)
- **Voice**: ElevenLabs API
- **Expression Sync**: Runway Act-Two API (Pipeline A) / Wav2Lip on Replicate (Pipeline B)
- **Hosting**: Render.com

---

## 🚀 Deploy to Render (1-click)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` → click **Deploy**
5. Get your live URL: `https://facestudio.onrender.com`

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## 🔑 API Keys Needed

| Service | Get it at | Used for |
|---------|-----------|----------|
| Replicate | replicate.com/account/api-tokens | Face swap + Wav2Lip |
| ElevenLabs | elevenlabs.io → Settings → API Keys | Voice clone + TTS |
| Runway | app.runwayml.com → Settings | Expression sync (Pipeline A) |

Keys are entered in the UI — never hardcoded.

---

## 📁 Project Structure

```
facestudio/
├── app.py              ← Flask backend (all API logic here)
├── templates/
│   └── index.html      ← Full UI (all steps, both pipelines)
├── requirements.txt
├── render.yaml         ← Render deployment config
├── Procfile            ← Gunicorn start command
└── .gitignore
```

---

## 🔧 Customizing Models

Edit these lines in `app.py`:

| What | Line to find |
|------|-------------|
| Face swap model | `codeplugtech/face-swap:278a...` |
| Runway model | `"model": "gen4_act_two"` |
| Wav2Lip model | `devxpy/cog-wav2lip:15d2...` |
| Voice emotions | `emotion_settings` dict |
