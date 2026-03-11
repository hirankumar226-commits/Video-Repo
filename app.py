"""
FaceStudio - AI Video Pipeline
================================
Step 1: Wan 2.2 Animate Replace (Replicate) — replaces entire character in video,
        copying body motion, expressions, gestures + scene relighting
Step 2: ElevenLabs — voice clone + speech generation
Step 3A: Runway Act-Two (Pipeline A — webcam-driven)
Step 3B: Wav2Lip on Replicate (Pipeline B — fully automated)

API KEYS NEEDED:
  - Replicate  → replicate.com/account/api-tokens  (Steps 1 + Pipeline B)
  - ElevenLabs → elevenlabs.io Settings → API Keys  (Step 2)
  - Runway     → app.runwayml.com/settings           (Pipeline A only)
"""

from flask import Flask, request, jsonify, send_file, render_template
import os, uuid, threading, time, requests, base64, subprocess
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

IS_RENDER  = os.environ.get('RENDER', False)
BASE_DIR   = Path("/tmp/facestudio") if IS_RENDER else Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Public base URL — Render sets this automatically, e.g. https://video-repo.onrender.com
# Used to serve files to models that need public URLs with proper extensions (Wav2Lip)
BASE_URL = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000').rstrip('/')

jobs = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def save_upload(file_obj, suffix=''):
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    file_obj.save(str(path))
    return path

def to_b64_uri(path, mime):
    with open(path, 'rb') as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

def update_job(job_id, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)

def download_to_output(url, filename):
    path = OUTPUT_DIR / filename
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

def merge_audio_video(video_path, audio_path, out_name):
    out = OUTPUT_DIR / out_name
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(video_path), '-i', str(audio_path),
         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(out)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"FFmpeg: {result.stderr}")
    return out


# ── UI routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files/<filename>')
def serve_file(filename):
    for d in [OUTPUT_DIR, UPLOAD_DIR]:
        p = d / filename
        if p.exists():
            return send_file(str(p))
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/job/<job_id>')
def get_job(job_id):
    return jsonify(jobs.get(job_id, {'status': 'not_found'}))


# ── Replicate helpers (used by Step 1 and Pipeline B) ────────────────────────

def _replicate_upload(api_key, file_path, mime):
    name = Path(file_path).name
    data = open(file_path, 'rb').read()
    r = requests.post(
        'https://api.replicate.com/v1/files',
        headers={'Authorization': f'Bearer {api_key}'},
        files={'content': (name, data, mime)},
        timeout=120
    )
    if not r.ok:
        raise Exception(f"Replicate upload {r.status_code}: {r.text}")
    return r.json()['urls']['get']


def _replicate_run(api_key, model, inputs, timeout_mins=15):
    """
    Run a Replicate model. Handles two API styles:
      - New models (no public versions): POST /v1/models/{owner}/{name}/predictions
      - Legacy models (versioned):       POST /v1/predictions  with version hash
    Tries the new style first; falls back to versioned if needed.
    Automatically retries on 429 rate limit with exponential backoff.
    """
    auth    = {'Authorization': f'Bearer {api_key}'}
    headers = {**auth, 'Content-Type': 'application/json'}

    # ── Submit with retry on 429 ──────────────────────────────────────────────
    def submit_with_retry(url, body):
        for attempt in range(6):
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code == 429:
                retry_after = int(r.json().get('retry_after', 15))
                wait = max(retry_after, 10) * (attempt + 1)
                time.sleep(wait)
                continue
            return r
        raise Exception(f"Replicate still rate-limiting after 6 retries. Add credit at replicate.com/account/billing to raise your limit.")

    # ── Try new-style deployment endpoint first ───────────────────────────────
    pr = submit_with_retry(
        f'https://api.replicate.com/v1/models/{model}/predictions',
        {'input': inputs}
    )

    # ── Fall back to versioned endpoint if new-style isn't supported ──────────
    if pr.status_code == 404:
        vr = requests.get(
            f'https://api.replicate.com/v1/models/{model}/versions',
            headers=auth, timeout=30
        )
        if not vr.ok:
            raise Exception(f"Replicate versions {vr.status_code}: {vr.text}")
        vs = vr.json().get('results', [])
        if not vs:
            raise Exception(f"No versions found for {model}")
        version = vs[0]['id']
        pr = submit_with_retry(
            'https://api.replicate.com/v1/predictions',
            {'version': version, 'input': inputs}
        )

    if not pr.ok:
        raise Exception(f"Replicate submit {pr.status_code}: {pr.text}")

    pred_id = pr.json()['id']

    # ── Poll until done ───────────────────────────────────────────────────────
    for _ in range(timeout_mins * 12):
        time.sleep(5)
        poll = requests.get(
            f'https://api.replicate.com/v1/predictions/{pred_id}',
            headers=auth, timeout=30
        )
        poll.raise_for_status()
        d = poll.json()
        if d['status'] == 'succeeded':
            return d['output']
        if d['status'] in ('failed', 'canceled'):
            raise Exception(
                f"Replicate failed: {d.get('error','?')} | "
                f"logs: {str(d.get('logs',''))[-400:]}"
            )
    raise Exception(f'Replicate timeout after {timeout_mins}m')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CHARACTER REPLACE  (Wan 2.2 Animate Replace on Replicate)
# replicate.com/wan-video/wan-2.2-animate-replace
#
# Replaces the ENTIRE character in the source video with your AI character
# image. Copies body skeleton, facial expressions, hand gestures.
# Relighting LoRA adjusts scene lighting — looks natural, not pasted-on.
#
# Inputs: image (character photo) · video (source reel) · resolution
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/faceswap', methods=['POST'])
def start_faceswap():
    key        = request.form.get('replicate_key', '').strip()
    face_file  = request.files.get('face_image')
    video_file = request.files.get('video')
    resolution = request.form.get('resolution', '720p')
    prompt     = request.form.get('prompt', '').strip()

    if not key:
        return jsonify({'error': 'Replicate API key required'}), 400
    if not face_file or not video_file:
        return jsonify({'error': 'Character image and source video are required'}), 400

    face_path  = save_upload(face_file, '.jpg')
    video_path = save_upload(video_file, '.mp4')
    job_id     = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}

    threading.Thread(
        target=_run_wan_replace,
        args=(job_id, key, face_path, video_path, resolution, prompt),
        daemon=True
    ).start()

    return jsonify({'job_id': job_id})


def _run_wan_replace(job_id, key, face_path, video_path, resolution, prompt):
    import traceback
    try:
        update_job(job_id, status='running', progress=10, step='Uploading character image...')
        face_url = _replicate_upload(key, face_path, 'image/jpeg')

        update_job(job_id, progress=25, step='Uploading source video...')
        video_url = _replicate_upload(key, video_path, 'video/mp4')

        update_job(job_id, progress=40, step='Running Wan 2.2 Animate Replace...')
        res_clean = resolution.replace('p', '')  # '720p' -> '720'
        inputs = {'character_image': face_url, 'video': video_url, 'resolution': res_clean}
        if prompt:
            inputs['prompt'] = prompt

        output = _replicate_run(key, 'wan-video/wan-2.2-animate-replace', inputs, timeout_mins=15)
        out_url = str(output) if not isinstance(output, list) else str(output[0])

        update_job(job_id, progress=90, step='Downloading result...')
        out_file = f"wan_replace_{job_id}.mp4"
        download_to_output(out_url, out_file)

        update_job(job_id, status='done', progress=100, step='Character replace complete! ✅',
                   result_url=f'/files/{out_file}', external_url=out_url)

    except Exception as e:
        update_job(job_id, status='error', error=f"{e} | {traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VOICE CLONE + SPEECH  (ElevenLabs)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/voice/clone', methods=['POST'])
def clone_voice():
    key         = request.form.get('elevenlabs_key', '').strip()
    voice_name  = request.form.get('voice_name', 'AI Character').strip()
    audio_files = request.files.getlist('voice_samples')
    if not key:
        return jsonify({'error': 'ElevenLabs key required'}), 400
    if not audio_files:
        return jsonify({'error': 'At least one voice sample required'}), 400
    try:
        r = requests.post(
            'https://api.elevenlabs.io/v1/voices/add',
            headers={'xi-api-key': key},
            data={'name': voice_name, 'description': 'AI character voice'},
            files=[('files', (f.filename, f.read(), f.mimetype)) for f in audio_files],
            timeout=60
        )
        if r.status_code != 200:
            return jsonify({'error': f'ElevenLabs: {r.text}'}), 400
        return jsonify({'success': True, 'voice_id': r.json()['voice_id'], 'voice_name': voice_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/speak', methods=['POST'])
def generate_speech():
    data     = request.get_json()
    key      = data.get('elevenlabs_key', '').strip()
    voice_id = data.get('voice_id', '').strip()
    script   = data.get('script', '').strip()
    emotion  = data.get('emotion', 'neutral')
    if not all([key, voice_id, script]):
        return jsonify({'error': 'key, voice_id, script required'}), 400

    presets = {
        'calm':      {'stability': 0.85, 'similarity_boost': 0.90, 'style': 0.10},
        'neutral':   {'stability': 0.65, 'similarity_boost': 0.80, 'style': 0.20},
        'excited':   {'stability': 0.35, 'similarity_boost': 0.70, 'style': 0.55},
        'emotional': {'stability': 0.20, 'similarity_boost': 0.65, 'style': 0.70},
    }
    vs = presets.get(emotion, presets['neutral'])
    try:
        r = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers={'xi-api-key': key, 'Content-Type': 'application/json'},
            json={'text': script, 'model_id': 'eleven_multilingual_v2',
                  'voice_settings': {**vs, 'use_speaker_boost': True}},
            timeout=60
        )
        if r.status_code != 200:
            return jsonify({'error': f'ElevenLabs: {r.text}'}), 400
        out = f"speech_{uuid.uuid4().hex}.mp3"
        (OUTPUT_DIR / out).write_bytes(r.content)
        return jsonify({'success': True, 'audio_url': f'/files/{out}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3A — PIPELINE A: RUNWAY ACT-TWO
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/sync/runway', methods=['POST'])
def start_runway_sync():
    key     = request.form.get('runway_key', '').strip()
    char_v  = request.files.get('character_video')
    perf_v  = request.files.get('performance_video')
    audio_f = request.files.get('audio')
    if not key:
        return jsonify({'error': 'Runway key required'}), 400
    if not char_v or not perf_v:
        return jsonify({'error': 'character_video and performance_video required'}), 400

    cp = save_upload(char_v, '.mp4')
    pp = save_upload(perf_v, '.mp4')
    ap = save_upload(audio_f, '.mp3') if audio_f else None

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}
    threading.Thread(target=_run_runway, args=(job_id, key, cp, pp, ap), daemon=True).start()
    return jsonify({'job_id': job_id})


def _run_runway(job_id, key, char_path, perf_path, audio_path):
    try:
        update_job(job_id, status='running', progress=15, step='Encoding videos...')
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                   'X-Runway-Version': '2024-11-06'}
        payload = {"model": "gen4_act_two",
                   "promptVideo": to_b64_uri(perf_path, 'video/mp4'),
                   "promptImage": to_b64_uri(char_path, 'video/mp4'),
                   "duration": 10, "ratio": "720:1280"}

        update_job(job_id, progress=30, step='Submitting to Runway Act-Two...')
        r = requests.post('https://api.dev.runwayml.com/v1/image_to_video',
                          headers=headers, json=payload, timeout=60)
        if r.status_code not in (200, 201):
            raise Exception(f'Runway {r.status_code}: {r.text}')

        task_id = r.json()['id']
        for i in range(120):
            time.sleep(5)
            poll = requests.get(f'https://api.dev.runwayml.com/v1/tasks/{task_id}',
                                headers=headers, timeout=30).json()
            st = poll.get('status', 'PENDING')
            update_job(job_id, progress=min(40 + i, 85), step=f'Runway: {st}...')
            if st == 'SUCCEEDED':
                video_url = poll['output'][0]; break
            if st in ('FAILED', 'CANCELLED'):
                raise Exception(f'Runway failed: {poll.get("failure")}')
        else:
            raise Exception('Runway timed out')

        raw = f"runway_raw_{job_id}.mp4"
        update_job(job_id, progress=87, step='Downloading...')
        download_to_output(video_url, raw)

        if audio_path:
            update_job(job_id, progress=93, step='Merging audio...')
            final = f"pipeline_a_{job_id}.mp4"
            merge_audio_video(OUTPUT_DIR / raw, audio_path, final)
        else:
            final = raw

        update_job(job_id, status='done', progress=100, step='Pipeline A complete! ✅',
                   result_url=f'/files/{final}')
    except Exception as e:
        update_job(job_id, status='error', error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3B — PIPELINE B: AUTOMATED LIP SYNC  (Wav2Lip on Replicate)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/sync/auto', methods=['POST'])
def start_auto_sync():
    key   = request.form.get('replicate_key', '').strip()
    vid_f = request.files.get('video')
    aud_f = request.files.get('audio')
    if not key:
        return jsonify({'error': 'Replicate key required'}), 400
    if not vid_f or not aud_f:
        return jsonify({'error': 'video and audio required'}), 400

    vp = save_upload(vid_f, '.mp4')
    ap = save_upload(aud_f, '.mp3')
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}
    threading.Thread(target=_run_wav2lip, args=(job_id, key, vp, ap), daemon=True).start()
    return jsonify({'job_id': job_id})


def _run_wav2lip(job_id, key, video_path, audio_path):
    import traceback, shutil
    try:
        # Wav2Lip's cog model checks file extension via args.face.split('.')[1]
        # Replicate upload URLs have no extension → IndexError.
        # Fix: serve files from our own public URL (which has .mp4/.mp3 in the path).
        update_job(job_id, status='running', progress=20, step='Preparing files...')
        vid_name = f"wav2lip_face_{job_id}.mp4"
        aud_name = f"wav2lip_audio_{job_id}.mp3"
        shutil.copy(str(video_path), str(UPLOAD_DIR / vid_name))
        shutil.copy(str(audio_path), str(UPLOAD_DIR / aud_name))
        vu = f"{BASE_URL}/files/{vid_name}"
        au = f"{BASE_URL}/files/{aud_name}"
        update_job(job_id, progress=50, step='Running Wav2Lip...')

        out = _replicate_run(key, 'devxpy/cog-wav2lip',
                             {'face': vu, 'audio': au, 'pads': '0 10 0 0',
                              'smooth': True, 'resize_factor': 1, 'nosmooth': False})
        out_url = str(out) if not isinstance(out, list) else str(out[0])

        update_job(job_id, progress=85, step='Downloading result...')
        out_file = f"pipeline_b_{job_id}.mp4"
        download_to_output(out_url, out_file)
        update_job(job_id, status='done', progress=100, step='Pipeline B complete! ✅',
                   result_url=f'/files/{out_file}', external_url=out_url)

    except Exception as e:
        update_job(job_id, status='error', error=f"{e} | {traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE C — OMNIHUMAN 1.5  (Replicate: bytedance/omni-human-1.5)
# https://replicate.com/bytedance/omni-human-1.5
#
# Replaces Steps 1 + 3 entirely. One call does everything:
#   • character image  → identity anchor
#   • audio (ElevenLabs output) → drives lip sync, expressions, gestures
#   • optional prompt  → control camera, emotion, staging
#
# The model understands speech semantics — so if the character says something
# sad, it looks sad. If excited, it gestures excitedly. Not just beat-matching.
#
# Inputs:
#   image  → AI character photo (portrait / half-body / full-body)
#   audio  → ElevenLabs MP3 (max 35 seconds)
#   prompt → optional scene direction (camera, emotion, actions)
# Output: video URL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/omnihuman', methods=['POST'])
def start_omnihuman():
    replicate_key = request.form.get('replicate_key', '').strip()
    image_file    = request.files.get('image')
    audio_file    = request.files.get('audio')
    prompt        = request.form.get('prompt', '').strip()

    if not replicate_key:
        return jsonify({'error': 'Replicate API key required'}), 400
    if not image_file:
        return jsonify({'error': 'Character image required'}), 400
    if not audio_file:
        return jsonify({'error': 'Audio file required (use ElevenLabs output from Step 2)'}), 400

    image_path = save_upload(image_file, '.jpg')
    audio_path = save_upload(audio_file, '.mp3')

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}

    threading.Thread(
        target=_run_omnihuman,
        args=(job_id, replicate_key, image_path, audio_path, prompt),
        daemon=True
    ).start()

    return jsonify({'job_id': job_id})


def _run_omnihuman(job_id, api_key, image_path, audio_path, prompt):
    import traceback
    try:
        update_job(job_id, status='running', progress=10, step='Uploading character image...')
        image_url = _replicate_upload(api_key, image_path, 'image/jpeg')

        update_job(job_id, progress=25, step='Uploading audio...')
        audio_url = _replicate_upload(api_key, audio_path, 'audio/mpeg')

        update_job(job_id, progress=40, step='Running OmniHuman 1.5 — generating video...')

        # OmniHuman 1.5 input schema:
        #   image  → portrait URL
        #   audio  → speech audio URL (≤35s)
        #   prompt → optional scene direction string
        inputs = {
            'image': image_url,
            'audio': audio_url,
        }
        if prompt:
            inputs['prompt'] = prompt

        output = _replicate_run(
            api_key,
            'bytedance/omni-human-1.5',
            inputs,
            timeout_mins=10
        )

        # output is a URL string or FileOutput object
        video_url = str(output) if not isinstance(output, list) else str(output[0])

        update_job(job_id, progress=90, step='Downloading result...')
        out_file = f"omnihuman_{job_id}.mp4"
        download_to_output(video_url, out_file)

        update_job(job_id,
            status='done', progress=100, step='OmniHuman complete! ✅',
            result_url=f'/files/{out_file}',
            external_url=video_url
        )

    except Exception as e:
        update_job(job_id, status='error', error=f"{e} | {traceback.format_exc()}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "═"*50)
    print("  🎬  FaceStudio AI Pipeline")
    print("  👉  http://localhost:5000")
    print("═"*50 + "\n")
    app.run(debug=True, port=5000, threaded=True)
