
# API Documentation

## Table of Contents
- [POST /v2/inference](#post-v2inference)
- [GET /docs](#get-docs)
- [GET /docs.md](#get-docsmd)
- [GET /health](#get-health)
- [GET /v1/models](#get-v1models)

---

## POST /v2/inference

### Description
The `/v2/inference` endpoint is used for generating audio based on the provided text and voice model. Optionally, a reference text (`ref_text`) can be provided, which triggers the `STinference` function for style transfer. [Reference](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LibriTTS.ipynb)

### Request

**Method**: `POST`
**URL**: `/v2/inference`

### Description
The `/v2/inference` endpoint generates audio based on the provided text and voice model. Optionally, a reference text (`ref_text`) can be provided for style transfer using the `STinference` function. Additional advanced parameters such as `alpha`, `beta`, and `embedding_scale` can be adjusted for fine-tuning the output.

> **Note**: The `embedding_scale` parameter currently seems to be broken and should be left at its default value of `1`.

### Request

**Method**: `POST`
**URL**: `/v2/inference`

### Parameters

| Name            | Type    | Required | Description                                                                 |
|-----------------|---------|----------|-----------------------------------------------------------------------------|
| text            | string  | Yes      | The text input that will be converted into audio.                           |
| voice           | string  | Yes      | The voice model to be used for synthesis (e.g., `f-us-2`).                  |
| ref_text        | string  | No       | An optional reference text used for style transfer (`STinference`).         |
| format          | string  | No       | The audio format for the response. Defaults to `mp3`.                       |
| bitrate         | string  | No       | Audio bitrate for `mp3` or `opus` format. Defaults to `64k`.                |
| alpha           | float   | No       | Controls the degree of variability in the audio generation. Default: `0.3`. |
| beta            | float   | No       | Controls the stylistic emphasis. Default: `0.7`.                            |
| speed           | float   | No       | Allows speed adjustment (higher is faster). Default: `1.0`, recommended range: `0.9-1.25`  More extreme adjustments may affect quality. |
| embedding_scale | float   | No       | Adjusts the voice embedding. Default: `1`. **Currently recommended to keep at `1` due to issues.** |

Supported formats are:
- `mp3` *default* – MPEG audio with customizable bitrate.
- `opus` – Opus codec in Ogg container with customizable bitrate.
- `wav` – 16-bit 16kHz PCM in a WAV file.
- `wav-full` – 32-bit 24kHz PCM in a WAV file.

### Example Request

```bash
curl -X POST http://your-server-url/v2/inference      -F "text=Hello, world!"      -F "voice=f-us-2"      -F "format=wav"      --output output.wav
```

### Responses

- **Success** (200 OK):
  The response is the generated audio file in the specified format (e.g., WAV).

- **Error** (400 Bad Request):
  If the request is missing required fields or contains invalid values, a JSON error response is returned.

#### Example Error Response

```json
{
  "error": "Missing or invalid fields. Please include 'text' and 'voice' in your request, and ensure the voice selected is valid."
}
```

---

## GET /docs

### Description
The `/docs` endpoint serves a rendered HTML version of the API documentation.

### Request

**Method**: `GET`
**URL**: `/docs`

### Example Request

```bash
curl http://your-server-url/docs
```

### Response

- **Success** (200 OK):
  Renders the API documentation in HTML format.

---

## GET /docs.md

### Description
The `/docs.md` endpoint serves the raw Markdown file of the API documentation.

### Request

**Method**: `GET`
**URL**: `/docs.md`

### Example Request

```bash
curl http://your-server-url/docs.md
```

### Response

- **Success** (200 OK):
  Returns the raw Markdown file for the API documentation.

---

## GET /health

### Description
The `/health` endpoint is a simple health check that returns "OK" if the service is running properly.

### Request

**Method**: `GET`
**URL**: `/health`

### Example Request

```bash
curl http://your-server-url/health
```

### Response

- **Success** (200 OK):
  Returns a plain text response: `"OK"`.

---

## GET /v1/models

### Description
The `/v1/models` endpoint returns a list of available voice models.

### Request

**Method**: `GET`
**URL**: `/v1/models`

### Example Request

```bash
curl http://your-server-url/v1/models
```

### Response

- **Success** (200 OK):
  Returns the model name and voicelist.

#### Example Success Response

```json
{
    "model":"StyleTTS2",
    "voicelist":["f-us-1","f-us-2","f-us-3","f-us-4","m-us-1","m-us-2","m-us-3","m-us-4"]
}
```

---
