# ai-personal-asistant-from-zero

Run Qwen3 as a personal assistant.

Setup virtual environment.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install deps.

```bash
pip install --upgrade --quiet \
    transformers \
    accelerate \
    bitsandbytes \
    einops \
    ipywidgets \
    protobuf
```

Check your GPU.

```bash
python check-cuda.py 

Successfully imported bitsandbytes
Using device: cuda
```

Check model downloads and loads.

```bash
export HF_TOKEN = hf_your_hugging_face_token
python load-model.py
```

Chat CLI

```bash
python chat.py
```

Chat UI

```bash
pip install --upgrade --quiet \
    jupyter
```

Load `chat.py` into a notebook and run it. Select ChatUI.
