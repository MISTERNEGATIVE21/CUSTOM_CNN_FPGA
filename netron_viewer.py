#!/usr/bin/env python3
"""Serve local model files and generate shareable netron.app links via Flask."""

from __future__ import annotations

import shutil
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from flask import (
    Flask,
    flash,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

app = Flask(__name__)
app.secret_key = "netron-uploader"

UPLOAD_DIR = Path("netron_uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("models").resolve()
SUPPORTED_EXTENSIONS = {".onnx", ".pb", ".h5", ".keras", ".json", ".tflite", ".pt", ".pth"}

REGISTERED_MODELS: Dict[str, Path] = {}


def _persist_upload(storage) -> Path:
    filename = Path(storage.filename or "model")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "_".join(filter(None, filename.stem.split())) + filename.suffix
    target = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    storage.stream.seek(0)
    with target.open("wb") as f_out:
        shutil.copyfileobj(storage.stream, f_out)
    return target


def _discover_models() -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory not found at {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return discovered

    model_files = [p for p in MODELS_DIR.rglob("*") 
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not model_files:
        print(f"⚠️  No model files with extensions {SUPPORTED_EXTENSIONS} in {MODELS_DIR}")
    
    for path in model_files:
        rel = path.relative_to(MODELS_DIR)
        discovered[str(rel)] = path.resolve()
        print(f"✅ Discovered model: {rel} at {path}")
    
    return dict(sorted(discovered.items()))


def _register_model(path: Path) -> str:
    model_id = uuid.uuid4().hex
    REGISTERED_MODELS[model_id] = path
    return model_id


@app.route("/models/<model_id>")
def serve_model(model_id: str):
    path = REGISTERED_MODELS.get(model_id)
    if path is None or not path.exists():
        return ("Model not found", 404)

    response = send_file(path, as_attachment=False)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Netron Model Viewer</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; line-height: 1.6; }
      header { margin-bottom: 2rem; }
      form { padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
      label { font-weight: 600; }
      input[type=file] { margin: 0.75rem 0; }
      select { margin: 0.75rem 0; padding: 0.4rem 0.8rem; min-width: 280px; }
      button { padding: 0.6rem 1.2rem; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
      button:hover { background: #1d4ed8; }
      .message { margin-top: 1rem; padding: 0.75rem 1rem; border-radius: 6px; background: #ecfdf5; border: 1px solid #34d399; }
      iframe { margin-top: 2rem; width: 100%; height: 640px; border: 1px solid #ddd; border-radius: 6px; }
    </style>
  </head>
  <body>
    <header>
      <h1>🧠 Netron Model Viewer</h1>
      <p>Upload a neural network file and open it instantly on <a href="https://netron.app" target="_blank" rel="noopener">netron.app</a>. Supported formats include ONNX, TensorFlow, Keras, TFLite, and PyTorch checkpoints.</p>
    </header>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <div class="message">{{ message|safe }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form action="{{ url_for('index') }}" method="post" enctype="multipart/form-data">
      <input type="hidden" name="action" value="upload">
      <label for="model">Choose a model file:</label><br />
      <input id="model" name="model" type="file" accept=".onnx,.pb,.h5,.keras,.json,.tflite,.pt,.pth">
      <p style="font-size: 0.9rem; color: #555;">Files are copied to <code>netron_uploads/</code> and served locally so netron.app can fetch them.</p>
      <button type="submit">Upload &amp; Open on netron.app</button>
    </form>

    {% if existing_models %}
    <form action="{{ url_for('index') }}" method="post" style="margin-top: 2rem; padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; background: #f8fafc;">
      <input type="hidden" name="action" value="existing">
      <label for="existing_model">Or select a model already in <code>models/</code>:</label><br />
      <select id="existing_model" name="existing_model">
        <option value="">-- Choose an existing model --</option>
        {% for display in existing_models.keys() %}
          <option value="{{ display }}">{{ display }}</option>
        {% endfor %}
      </select>
      <p style="font-size: 0.9rem; color: #555;">Supported extensions: {{ extensions|join(', ') }}</p>
      <button type="submit">Open selected model</button>
    </form>
    {% endif %}

    {% if netron_url %}
      <div class="message" style="background:#eff6ff; border-color:#60a5fa;">
        ✅ Viewer ready: <a href="{{ netron_url }}" target="_blank" rel="noopener">open on netron.app</a><br />
        <small>The generated link allows netron.app to download the model from this Flask server. Ensure your machine is reachable if you plan to share the link externally.</small>
      </div>
      <iframe src="{{ netron_url }}" title="Netron Viewer"></iframe>
    {% endif %}
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    netron_url = None
    existing_models = _discover_models()

    if request.method == "POST":
        action = request.form.get("action", "upload")

        if action == "existing":
            chosen = request.form.get("existing_model", "")
            if not chosen:
                flash("⚠️ Please select a model from the dropdown before launching.")
                return redirect(url_for("index"))
            if chosen not in existing_models:
                flash("❌ Selected model could not be found on disk.")
                return redirect(url_for("index"))
            model_path = existing_models[chosen]
            display_name = chosen
        else:
            uploaded = request.files.get("model")
            if not uploaded or uploaded.filename == "":
                flash("⚠️ Please choose a file before submitting.")
                return redirect(url_for("index"))
            model_path = _persist_upload(uploaded)
            display_name = Path(uploaded.filename).name or model_path.name

        model_id = _register_model(model_path)
        file_url = url_for("serve_model", model_id=model_id, _external=True)
        netron_url = f"https://netron.app/?url={urllib.parse.quote(file_url, safe='')}"
        flash(
            "✅ Netron link ready for <strong>{name}</strong>. Use the button below or copy the URL to open directly on netron.app.".format(
                name=display_name
            )
        )

    return render_template_string(
        FORM_TEMPLATE,
        netron_url=netron_url,
        existing_models=existing_models,
        extensions=sorted(SUPPORTED_EXTENSIONS),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
