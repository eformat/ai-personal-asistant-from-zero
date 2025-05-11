# ai-personal-asistant-from-zero

Run Qwen3

```bash
python3.12 -m venv venv
source venv/bin/activate
```

```bash
pip install --upgrade --quiet \
    transformers \
    accelerate \
    bitsandbytes \
    einops \
    ipywidgets \
    protobuf
```

```bash
python check-cuda.py 

Successfully imported bitsandbytes
Using device: cuda
```

```bash
python load-model.py
```

Chat CLI

```bash
python chat.py
```

```bash
pip install --upgrade --quiet \
    jupyter
```

Load `chat.py` into a notebook and run it. Select ChatUI.
