"""
FaceStudio - AI Video Pipeline
================================
Full pipeline: Face Swap → Voice Clone → Expression Sync

Pipeline A: You drive it (Runway Act-Two)
Pipeline B: Fully automated (Wav2Lip on Replicate)

SETUP:
  pip install flask replicate requests
  python app.py
  Open: http://localhost:5000

API KEYS NEEDED:
  - Replicate  → replicate.com/account/api-tokens
  - ElevenLabs → elevenlabs.io (Settings → API Key)
  - Runway     → app.runwayml.com/settings (Pipeline A only)
"""

from flask import Flask, request, jsonify, send_file, render_template
import os, uuid, threading, time, requests, base64, subprocess, json
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# ── Directories ──────────────────────────────────────────────────────────────
# On Render: uses /data (persistent disk). Locally: uses uploads/ and outputs/
IS_RENDER = os.environ.get('RENDER', False)
BASE_DIR   = Path("/tmp/facestudio") if IS_RENDER else Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# Each job: { status, progress, step, result_url, error }
jobs = {}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_upload(file_obj, suffix=''):
    """Save an uploaded file to the uploads dir, return its Path."""
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / name
    file_obj.save(str(path))
    return path

def to_b64_uri(path, mime):
    """Convert a local file to a base64 data URI (for APIs that accept it)."""
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def update_job(job_id, **kwargs):
    """Update job fields in the store."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)

def download_to_output(url, filename):
    """Download a remote URL and save to outputs dir."""
    path = OUTPUT_DIR / filename
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        f.write(resp.content)
    return path

def merge_audio_video(video_path, audio_path, output_filename):
    """Use ffmpeg to replace video audio with new audio."""
    out_path = OUTPUT_DIR / output_filename
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — UI
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files/<filename>')
def serve_file(filename):
    """Serve a file from outputs or uploads dir."""
    for directory in [OUTPUT_DIR, UPLOAD_DIR]:
        path = directory / filename
        if path.exists():
            return send_file(str(path))
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/job/<job_id>')
def get_job(job_id):
    """Poll job status."""
    return jsonify(jobs.get(job_id, {'status': 'not_found'}))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FACE SWAP (Replicate)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/faceswap', methods=['POST'])
def start_faceswap():
    replicate_key = request.form.get('replicate_key', '').strip()
    face_file     = request.files.get('face_image')
    video_file    = request.files.get('video')

    if not replicate_key:
        return jsonify({'error': 'Replicate API key required'}), 400
    if not face_file or not video_file:
        return jsonify({'error': 'Face image and video are required'}), 400

    face_path  = save_upload(face_file, '.jpg')
    video_path = save_upload(video_file, '.mp4')

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}

    t = threading.Thread(target=_run_faceswap, args=(job_id, replicate_key, face_path, video_path))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


def _replicate_upload_file(api_key, file_path, mime_type):
    """Upload a file to Replicate file storage via multipart, return the file URL."""
    filename = Path(file_path).name
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    resp = requests.post(
        'https://api.replicate.com/v1/files',
        headers={'Authorization': f'Bearer {api_key}'},
        files={'content': (filename, file_bytes, mime_type)},
        timeout=120
    )
    if not resp.ok:
        raise Exception(f"Replicate file upload failed {resp.status_code}: {resp.text}")
    return resp.json()['urls']['get']


def _replicate_get_version(api_key, model_owner_name):
    """Fetch the latest version hash for a community model dynamically."""
    headers = {'Authorization': f'Bearer {api_key}'}
    # GET /v1/models/{owner}/{name}/versions — returns list sorted newest first
    resp = requests.get(
        f'https://api.replicate.com/v1/models/{model_owner_name}/versions',
        headers=headers,
        timeout=30
    )
    if not resp.ok:
        raise Exception(f"Could not fetch versions for {model_owner_name}: {resp.status_code} {resp.text}")
    results = resp.json().get('results', [])
    if not results:
        raise Exception(f"No versions found for model {model_owner_name}")
    return results[0]['id']  # latest version


def _replicate_run(api_key, model_owner_name, input_data):
    """
    Run a community Replicate model via REST API.
    Dynamically fetches the latest version hash — never hardcoded, never stale.
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    # Step 1: get latest version hash
    version_id = _replicate_get_version(api_key, model_owner_name)

    # Step 2: create prediction with version hash
    resp = requests.post(
        'https://api.replicate.com/v1/predictions',
        headers=headers,
        json={'version': version_id, 'input': input_data},
        timeout=60
    )
    if not resp.ok:
        raise Exception(f"Replicate {resp.status_code}: {resp.text}")

    pred_id = resp.json()['id']

    # Step 3: poll until complete (max 10 min)
    for _ in range(120):
        time.sleep(5)
        poll = requests.get(
            f'https://api.replicate.com/v1/predictions/{pred_id}',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=30
        )
        poll.raise_for_status()
        data = poll.json()
        status = data.get('status')
        if status == 'succeeded':
            return data['output']
        elif status in ['failed', 'canceled']:
            raise Exception(f"Replicate failed: {data.get('error','unknown')} | logs: {str(data.get('logs',''))[-500:]}")

    raise Exception('Replicate timed out after 10 minutes')


def _run_faceswap(job_id, api_key, face_path, video_path):
    import traceback
    try:
        update_job(job_id, status='running', progress=15, step='Uploading face image...')
        face_url = _replicate_upload_file(api_key, face_path, 'image/jpeg')

        update_job(job_id, progress=30, step='Uploading source video...')
        video_url_input = _replicate_upload_file(api_key, video_path, 'video/mp4')

        update_job(job_id, progress=45, step='Running face swap model on Replicate...')

        # arabyai-replicate/roop_face_swap — video face swap (39K+ runs)
        # Inputs: swap_image (face photo), target_video (source video)
        # https://replicate.com/arabyai-replicate/roop_face_swap
        output = _replicate_run(
            api_key,
            'arabyai-replicate/roop_face_swap',
            {
                'swap_image':   face_url,
                'target_video': video_url_input,
            }
        )

        video_url = str(output) if not isinstance(output, list) else str(output[0])
        update_job(job_id, progress=85, step='Downloading face-swapped video...')

        out_file = f"faceswap_{job_id}.mp4"
        download_to_output(video_url, out_file)

        update_job(job_id,
            status='done', progress=100, step='Face swap complete! ✅',
            result_url=f'/files/{out_file}',
            external_url=video_url
        )

    except Exception as e:
        tb = traceback.format_exc()
        update_job(job_id, status='error', error=f"{str(e)} | Traceback: {tb}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VOICE CLONE + SPEECH GENERATION (ElevenLabs)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/voice/clone', methods=['POST'])
def clone_voice():
    """Clone a voice from uploaded audio samples."""
    api_key      = request.form.get('elevenlabs_key', '').strip()
    voice_name   = request.form.get('voice_name', 'AI Character').strip()
    audio_files  = request.files.getlist('voice_samples')

    if not api_key:
        return jsonify({'error': 'ElevenLabs API key required'}), 400
    if not audio_files:
        return jsonify({'error': 'At least one voice sample required'}), 400

    try:
        headers = {'xi-api-key': api_key}
        files   = [('files', (f.filename, f.read(), f.mimetype)) for f in audio_files]
        data    = {'name': voice_name, 'description': 'AI character voice clone'}

        resp = requests.post(
            'https://api.elevenlabs.io/v1/voices/add',
            headers=headers,
            data=data,
            files=files,
            timeout=60
        )

        if resp.status_code != 200:
            return jsonify({'error': f'ElevenLabs: {resp.text}'}), 400

        voice_id = resp.json()['voice_id']
        return jsonify({'success': True, 'voice_id': voice_id, 'voice_name': voice_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/speak', methods=['POST'])
def generate_speech():
    """Generate speech audio from text using a cloned voice."""
    data       = request.get_json()
    api_key    = data.get('elevenlabs_key', '').strip()
    voice_id   = data.get('voice_id', '').strip()
    script     = data.get('script', '').strip()
    emotion    = data.get('emotion', 'neutral')  # calm | neutral | excited | emotional

    if not all([api_key, voice_id, script]):
        return jsonify({'error': 'API key, voice ID, and script are required'}), 400

    # Emotion → voice settings mapping
    emotion_settings = {
        'calm':      {'stability': 0.85, 'similarity_boost': 0.90, 'style': 0.10},
        'neutral':   {'stability': 0.65, 'similarity_boost': 0.80, 'style': 0.20},
        'excited':   {'stability': 0.35, 'similarity_boost': 0.70, 'style': 0.55},
        'emotional': {'stability': 0.20, 'similarity_boost': 0.65, 'style': 0.70},
    }
    vs = emotion_settings.get(emotion, emotion_settings['neutral'])

    try:
        headers = {'xi-api-key': api_key, 'Content-Type': 'application/json'}
        body = {
            'text': script,
            'model_id': 'eleven_multilingual_v2',
            'voice_settings': {**vs, 'use_speaker_boost': True}
        }

        resp = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers=headers,
            json=body,
            timeout=60
        )

        if resp.status_code != 200:
            return jsonify({'error': f'ElevenLabs: {resp.text}'}), 400

        out_file = f"speech_{uuid.uuid4().hex}.mp3"
        out_path = OUTPUT_DIR / out_file
        with open(out_path, 'wb') as f:
            f.write(resp.content)

        return jsonify({'success': True, 'audio_url': f'/files/{out_file}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3A — PIPELINE A: RUNWAY ACT-TWO (You drive it with webcam)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/sync/runway', methods=['POST'])
def start_runway_sync():
    """
    Pipeline A: Transfer your webcam performance onto the AI character.
    - character_video: the face-swapped video from Step 1
    - performance_video: your webcam recording of you acting the script
    - audio: the ElevenLabs-generated speech from Step 2
    """
    runway_key         = request.form.get('runway_key', '').strip()
    character_video    = request.files.get('character_video')
    performance_video  = request.files.get('performance_video')
    audio_file         = request.files.get('audio')

    if not runway_key:
        return jsonify({'error': 'Runway API key required'}), 400
    if not character_video or not performance_video:
        return jsonify({'error': 'Character video and performance video required'}), 400

    char_path  = save_upload(character_video, '.mp4')
    perf_path  = save_upload(performance_video, '.mp4')
    audio_path = save_upload(audio_file, '.mp3') if audio_file else None

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}

    t = threading.Thread(target=_run_runway_acttwo,
                         args=(job_id, runway_key, char_path, perf_path, audio_path))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


def _run_runway_acttwo(job_id, api_key, char_path, perf_path, audio_path):
    try:
        update_job(job_id, status='running', progress=15, step='Encoding videos...')

        char_b64 = to_b64_uri(char_path, 'video/mp4')
        perf_b64 = to_b64_uri(perf_path, 'video/mp4')

        update_job(job_id, progress=30, step='Submitting to Runway Act-Two...')

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'X-Runway-Version': '2024-11-06'
        }

        # ── Runway Act-Two API ──────────────────────────────────────────────
        # Docs: https://docs.dev.runwayml.com
        # Model options: gen4_act_two (Act-Two), gen3a_turbo (Act-One)
        payload = {
            "model": "gen4_act_two",        # ← change if needed
            "promptVideo": perf_b64,         # your performance (driving video)
            "promptImage": char_b64,         # AI character (target)
            "duration": 10,
            "ratio": "720:1280",             # portrait for reels
        }

        resp = requests.post(
            'https://api.dev.runwayml.com/v1/image_to_video',
            headers=headers,
            json=payload,
            timeout=60
        )

        if resp.status_code not in [200, 201]:
            raise Exception(f'Runway API {resp.status_code}: {resp.text}')

        task_id = resp.json().get('id')
        update_job(job_id, progress=40, step=f'Runway processing (task {task_id})...')

        # Poll until done (max 10 min)
        for i in range(120):
            time.sleep(5)
            poll = requests.get(
                f'https://api.dev.runwayml.com/v1/tasks/{task_id}',
                headers=headers,
                timeout=30
            )
            task_data   = poll.json()
            task_status = task_data.get('status', 'PENDING')
            progress    = min(40 + i, 85)
            update_job(job_id, progress=progress, step=f'Runway: {task_status}...')

            if task_status == 'SUCCEEDED':
                video_url = task_data['output'][0]
                break
            elif task_status in ['FAILED', 'CANCELLED']:
                raise Exception(f'Runway failed: {task_data.get("failure", "Unknown")}')
        else:
            raise Exception('Runway task timed out after 10 minutes')

        update_job(job_id, progress=87, step='Downloading from Runway...')
        raw_file = f"runway_raw_{job_id}.mp4"
        download_to_output(video_url, raw_file)

        # Merge with ElevenLabs audio if provided
        if audio_path:
            update_job(job_id, progress=93, step='Merging voice audio...')
            final_file = f"pipeline_a_{job_id}.mp4"
            merge_audio_video(OUTPUT_DIR / raw_file, audio_path, final_file)
        else:
            final_file = raw_file

        update_job(job_id,
            status='done', progress=100, step='Pipeline A complete! ✅',
            result_url=f'/files/{final_file}'
        )

    except Exception as e:
        update_job(job_id, status='error', error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3B — PIPELINE B: AUTOMATED LIP SYNC (Wav2Lip on Replicate)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/sync/auto', methods=['POST'])
def start_auto_sync():
    """
    Pipeline B: Fully automated - just feed face-swapped video + generated audio.
    Uses Wav2Lip on Replicate for lip sync (no webcam needed).
    """
    replicate_key = request.form.get('replicate_key', '').strip()
    video_file    = request.files.get('video')
    audio_file    = request.files.get('audio')

    if not replicate_key:
        return jsonify({'error': 'Replicate API key required'}), 400
    if not video_file or not audio_file:
        return jsonify({'error': 'Video and audio are required'}), 400

    video_path = save_upload(video_file, '.mp4')
    audio_path = save_upload(audio_file, '.mp3')

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'step': 'Queued...'}

    t = threading.Thread(target=_run_auto_sync, args=(job_id, replicate_key, video_path, audio_path))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


def _run_auto_sync(job_id, api_key, video_path, audio_path):
    import traceback
    try:
        update_job(job_id, status='running', progress=20, step='Uploading video...')
        video_url_input = _replicate_upload_file(api_key, video_path, 'video/mp4')

        update_job(job_id, progress=35, step='Uploading audio...')
        audio_url_input = _replicate_upload_file(api_key, audio_path, 'audio/mpeg')

        update_job(job_id, progress=50, step='Running Wav2Lip lip sync on Replicate...')

        output = _replicate_run(
            api_key,
            'devxpy/cog-wav2lip',   # owner/name — no version hash needed
            {
                'face':          video_url_input,
                'audio':         audio_url_input,
                'pads':          '0 10 0 0',
                'smooth':        True,
                'resize_factor': 1,
                'nosmooth':      False,
            }
        )

        video_url = str(output) if not isinstance(output, list) else str(output[0])
        update_job(job_id, progress=85, step='Downloading result...')

        out_file = f"pipeline_b_{job_id}.mp4"
        download_to_output(video_url, out_file)

        update_job(job_id,
            status='done', progress=100, step='Pipeline B complete! ✅',
            result_url=f'/files/{out_file}',
            external_url=video_url
        )

    except Exception as e:
        tb = traceback.format_exc()
        update_job(job_id, status='error', error=f"{str(e)} | Traceback: {tb}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 50)
    print("  🎬  FaceStudio AI Pipeline")
    print("═" * 50)
    print("  👉  http://localhost:5000")
    print("═" * 50 + "\n")
    app.run(debug=True, port=5000, threaded=True)
