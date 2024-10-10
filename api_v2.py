# StyleTTS 2 HTTP Streaming API by @fakerybakery - Copyright (c) 2023
# mrfakename. All rights reserved.
# further API modifications by github.com/gilbertgong
# Docs: API_DOCS.md

import io
import os
import logging
import markdown
import subprocess
import time
from tortoise.utils.text import split_and_recombine_text
from flask import Flask, request, jsonify, Response, render_template_string, send_file
import scipy.io.wavfile as wavfile
import numpy as np
import msinference
from flask_cors import CORS
import nltk
from scipy import signal  # Import signal for resampling
from pydub import AudioSegment  # Import pydub for audio conversion
from gevent.lock import Semaphore

# Constants
URL_PREFIX = "/styletts2"
DEFAULT_FORMAT = "mp3"
DEFAULT_BITRATE = "64k"
ADDITIONAL_VOICE_DIR = os.environ.get('ADDITIONAL_VOICE_DIR', 'additional_voices')

logging.basicConfig(level=logging.INFO)
logging.info("Starting StyleTTS 2 API, logging level INFO")

# semaphore to ensure only one inference at a time
inference_lock = Semaphore(1)

# Load voices
# note these are random voices from the LibriTTS dataset
# https://huggingface.co/spaces/styletts2/styletts2/discussions/16
voicelist = [
    'f-us-1', 'f-us-2', 'f-us-3', 'f-us-4',
    'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4'
]
voices = {}
logging.info("Computing stock 8 voices")
for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')

def scan_additional_voice_dir():
    if ADDITIONAL_VOICE_DIR:
        logging.info(f"Scanning for additional voices from: {ADDITIONAL_VOICE_DIR}")
        new_voices_count = 0
        existing_voices_count = 0
        try:
            all_files = os.listdir(ADDITIONAL_VOICE_DIR)
            additional_voicelist = [os.path.splitext(f)[0] for f in all_files if f.endswith('.wav')]
            for v in additional_voicelist:
                if v not in voices:
                    voicelist.append(v)
                    wav_file_path = os.path.join(ADDITIONAL_VOICE_DIR, f"{v}.wav")
                    voices[v] = msinference.compute_style(wav_file_path)
                    logging.info(f"    Voice {v} computed")
                    new_voices_count += 1
                else:
                    existing_voices_count += 1
            logging.info(f"Added: {new_voices_count}, Previously Computed: {existing_voices_count}")
        except Exception as e:
            logging.error(f"An error occurred while loading additional voices: {e}")
    else:
        logging.info("No additional voice directory set.")


# Call the function initially to load the voices
scan_additional_voice_dir()

# We need to prime the app by initiating the punkt download
logging.info("Downloading punkt_tab")
nltk.download('punkt_tab')

logging.info('Generating "Hello world!" wav as a smoke test')
text = 'Hello world!'
wav = msinference.inference(
    text, voices['f-us-1'],
    alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1
)
logging.info("Generation done, if no errors were seen, we should be good")

logging.info("Starting Flask app")

app = Flask(__name__)
cors = CORS(app)

# explicitly close connection after each request
# to disable keepalive behavior
@app.after_request
def add_header(response):
    response.headers['Connection'] = 'close'
    return response

@app.route("/docs")
def docs():
    with open('API_DOCS.md', 'r') as f:
        content = f.read()
    html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
    style = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        table, th, td { border: 1px solid black; padding: 8px; }
        th { background-color: #f2f2f2; }
        pre { background-color: #f8f8f8; padding: 10px; border-radius: 5px; }
    </style>
    """
    return render_template_string(style + html_content)

@app.route("/docs.md")
def raw_md_docs():
    # Serve the raw markdown file with the correct MIME type
    return send_file('API_DOCS.md', mimetype='text/markdown')

@app.route("/")
@app.route("/health")
def health():
    return "OK"

@app.route("/v1/models")
def models():
    # check for new voices first
    scan_additional_voice_dir()
    response = {
        "model": "StyleTTS2",
        "voicelist": voicelist
    }
    return jsonify(response)

def generate_response(audios, format, bitrate=DEFAULT_BITRATE):
    if format == 'wav-full':
        concatenated_audio = np.concatenate(audios)
        # Model generates float32
        output_buffer = io.BytesIO()
        wavfile.write(output_buffer, 24000, concatenated_audio)
        response = Response(output_buffer.getvalue())
        response.headers["Content-Type"] = "audio/wav"
        return response

    elif format == 'wav':
        concatenated_audio = np.concatenate(audios)
        # Resample audio from 24kHz to 16kHz
        number_of_samples = round(len(concatenated_audio) * 16000 / 24000)
        resampled_audio = signal.resample(concatenated_audio, number_of_samples)
        # Convert to 16-bit PCM
        resampled_audio_int16 = np.int16(resampled_audio * 32767)
        output_buffer = io.BytesIO()
        wavfile.write(output_buffer, 16000, resampled_audio_int16)
        response = Response(output_buffer.getvalue())
        response.headers["Content-Type"] = "audio/wav"
        return response

    # bitrate paramter only applies to mp3
    elif format == 'mp3':
        logging.info("Converting to mp3 with bitrate: " + bitrate)
        concatenated_audio = np.concatenate(audios)
        temp_wav_path = 'temp_audio.wav'
        wavfile.write(temp_wav_path, 24000, concatenated_audio)
        output_buffer = io.BytesIO()
        temp_mp3_path = 'temp_audio.mp3'
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_wav_path,
            '-b:a', bitrate,  # Set bitrate to 192k
            temp_mp3_path
        ], check=True)
        with open(temp_mp3_path, 'rb') as f:
            output_buffer.write(f.read())
        response = Response(output_buffer.getvalue())
        response.headers["Content-Type"] = "audio/mpeg"
        return response

    elif format == 'opus':
        concatenated_audio = np.concatenate(audios)
        temp_wav_path = 'temp_audio.wav'
        wavfile.write(temp_wav_path, 24000, concatenated_audio)
        output_buffer = io.BytesIO()
        temp_opus_path = 'temp_audio.opus'
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_wav_path,
            '-c:a', 'libopus',
            '-b:a', bitrate,
            temp_opus_path
        ], check=True)
        with open(temp_opus_path, 'rb') as f:
            output_buffer.write(f.read())
        response = Response(output_buffer.getvalue())
        response.headers["Content-Type"] = "audio/ogg"
        return response

    else:
        # default to wav if format is not recognized
        # this avoids infinite loop if DEFAULT_FORMAT is misconfigured
        return generate_response(audios, format='wav')

def validate_input(form):
    if 'text' not in form or 'voice' not in form:
        return None

    text = form['text'].strip()
    voice = form['voice'].strip().lower()
    format = form.get('format', DEFAULT_FORMAT).strip().lower()
    bitrate = form.get('bitrate', DEFAULT_BITRATE).strip().lower()
    ref_text = form.get('ref_text')
    if ref_text:
        ref_text = ref_text.strip()

    # advanced parameters
    alpha = float(form.get('alpha', 0.3))
    beta = float(form.get('beta', 0.7))
    embedding_scale = form.get('embedding_scale', 1)  # Default to 1, no type conversion

    if voice not in voices:
        # rescan voice dir if voice not found, in case it's just been added
        logging.info(f"Voice {voice} not found, reloading voices...")
        scan_additional_voice_dir()
        if voice not in voices:
            # if still not found, generate error
            return None

    inputs = {
        'text': text,
        'voice': voice,
        'format': format,
        'bitrate': bitrate,
        'ref_text': ref_text,
        'alpha': alpha,
        'beta': beta,
        'embedding_scale': embedding_scale
    }
    return inputs

@app.route(URL_PREFIX + "/v2/inference", methods=['POST'])
def serve_inference():
    inputs = validate_input(request.form)
    if inputs is None:
        error_response = {
            'error': 'Missing or invalid fields. Please include "text" and "voice" in your request, and ensure the voice selected is valid.'
        }
        return jsonify(error_response), 400

    logging.info(f"Inputs received: {inputs}\n")

    v = voices[inputs['voice']]
    texts = split_and_recombine_text(inputs['text'])
    audios = []

    alpha = inputs['alpha']
    beta = inputs['beta']
    embedding_scale = inputs['embedding_scale']

    with inference_lock:
        start_inference_time = time.time()
        for t in texts:
            if inputs['ref_text']:
                audios.append(msinference.STinference(
                    t, v, inputs['ref_text'],
                    alpha=alpha, beta=beta, diffusion_steps=7, embedding_scale=embedding_scale
                ))
            else:
                audios.append(msinference.inference(
                    t, v,
                    alpha=alpha, beta=beta, diffusion_steps=7, embedding_scale=embedding_scale
                ))

        inference_duration = time.time() - start_inference_time

    start_response_time = time.time()
    response = generate_response(audios, format=inputs['format'], bitrate=inputs['bitrate'])
    response_duration = time.time() - start_response_time

    logging.info(f"inference time: {inference_duration:.4f}s, audio processing time: {response_duration:.4f}s.")

    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run("0.0.0.0", port=port, debug=False)
