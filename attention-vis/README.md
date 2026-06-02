# attention-vis

Interactive attention score heatmap visualization for HuggingFace transformer models.

Runs model inference, sums attention weights across all heads per layer, and generates a standalone HTML file with interactive Plotly heatmaps.

![alt text](image.png)

## Install

```bash
uv sync
```

## Usage

```bash
python app.py [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openbmb/MiniCPM5-1B` | HuggingFace model name |
| `--prompt` | `The quick brown fox jumps over the lazy dog.` | Input text to visualize |
| `--output` | `attention.html` | Output HTML file path |
| `--no-browser` | off | Don't open browser automatically |

### Examples

```bash
# Default model and prompt
python app.py

# Custom model and prompt
python app.py --model gpt2 --prompt "Hello, world!"

# Save without opening browser
python app.py --no-browser --output my_attention.html
```

## Controls

- **← Prev / Next →** — navigate between transformer layers
- **Arrow keys** — keyboard layer navigation (←/→/↑/↓)
- **Log Scale** — toggle log-transformed attention scores (`log(v+1) / log(base)`)
- **Base slider** — adjust logarithm base (2–10, default 2)