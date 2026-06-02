import argparse
import html
import json
import os
import webbrowser

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, device=None):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if device == "cpu":
        dtype = torch.float32
        device_map = None
    elif device == "cuda" or (device is None and torch.cuda.is_available()):
        dtype = torch.float16
        device_map = "auto"
    elif device == "mps" or (device is None and torch.backends.mps.is_available()):
        dtype = torch.float16
        device_map = None
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    if device:
        model = model.to(device)
    elif device_map is None and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    print(f"Model loaded. Device: {model.device}, Layers: {model.config.num_hidden_layers}")
    return tokenizer, model


def clean_token(tokenizer, token):
    text = tokenizer.convert_tokens_to_string([token])
    if text.strip() == "":
        return html.escape(token)
    return html.escape(text)


def get_attention(tokenizer, model, prompt, max_new_tokens=20):
    print(f"Running inference on: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

        # Generate continuation to verify model works (no cache to avoid Phi-3.5 compat issues)
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
        generated = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        print(f"Generated ({max_new_tokens} tokens): {generated!r}")

    tokens = [
        clean_token(tokenizer, t)
        for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    ]
    attentions = []
    for layer_attn in outputs.attentions:
        summed = layer_attn[0].sum(dim=0).cpu().numpy()
        attentions.append(summed)

    # Debug: attention statistics
    print(f"\nTokens ({len(tokens)}): {tokens}")
    print(f"Layers: {len(attentions)}")
    print("\nAttention stats per layer (summed over heads):")
    all_vals = []
    for i, attn in enumerate(attentions):
        flat = attn.flatten()
        all_vals.extend(flat.tolist())
        print(f"  Layer {i:2d}: min={flat.min():.4f}, max={flat.max():.4f}, mean={flat.mean():.4f}, std={flat.std():.4f}")
    all_arr = torch.tensor(all_vals)
    print(f"\nGlobal: min={all_arr.min():.4f}, max={all_arr.max():.4f}, mean={all_arr.mean():.4f}, std={all_arr.std():.4f}")
    return tokens, attentions


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Attention Visualization</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; }}
#nav {{
    display:flex; align-items:center; justify-content:center; gap:12px;
    padding:8px 16px; background:#1a1a2e; color:#eee; font-family:system-ui,sans-serif;
    flex-wrap:wrap;
    position:relative; z-index:100;
}}
#nav button {{
    background:#16213e; color:#eee; border:1px solid #0f3460;
    padding:6px 20px; border-radius:6px; cursor:pointer;
    font-size:15px; font-weight:600; transition:background .2s;
}}
#nav button:hover {{ background:#0f3460; }}
#nav button:disabled {{ opacity:.3; cursor:default; }}
#nav button.active {{ background:#e94560; border-color:#e94560; }}
#nav span {{ font-size:15px; min-width:140px; text-align:center; }}
.sep {{ width:1px; height:24px; background:#0f3460; }}
#log-controls {{ display:flex; align-items:center; gap:8px; }}
#log-controls label {{ font-size:13px; }}
#base-slider {{ width:80px; accent-color:#e94560; }}
#base-value {{ font-size:13px; min-width:24px; }}
#chart {{ width:100vw; height:calc(100vh - 50px); }}
</style>
</head>
<body>
<div id="nav">
  <button id="prev" onclick="go(-1)">&#8592; Prev</button>
  <span id="label">Layer 0 / {num_layers_minus1}</span>
  <button id="next" onclick="go(1)">Next &#8594;</button>
  <div class="sep"></div>
  <button id="logBtn" onclick="toggleLog()">Log Scale</button>
  <div id="log-controls">
    <label>Base:</label>
    <input id="base-slider" type="range" min="2" max="10" value="2" step="1" oninput="changeBase(this.value)">
    <span id="base-value">2</span>
  </div>
</div>
<div id="chart"></div>
<script>
var rawZ = {rawZ_json};
var numLayers = rawZ.length;
var current = 0;
var model = {model};
var logScale = false;
var logBase = 2;
var curZ = [];

function computeZ() {{
    curZ = [];
    for (var i = 0; i < numLayers; i++) {{
        if (logScale) {{
            curZ.push(rawZ[i].map(function(row) {{
                return row.map(function(v) {{ return Math.log(v + 1) / Math.log(logBase); }});
            }}));
        }} else {{
            curZ.push(rawZ[i].map(function(row) {{ return row.slice(); }}));
        }}
    }}
}}

computeZ();

var figData = [];
for (var i = 0; i < numLayers; i++) {{
    figData.push({{
        type: "heatmap",
        z: curZ[i],
        x: {tokens_json},
        y: {tokens_json},
        colorscale: [[0, '#ffffff'], [1, '#08306b']],
        visible: i === 0,
        name: "Layer " + i,
        hovertemplate: "From: %{{y}}<br>To: %{{x}}<br>Score: %{{z:.4f}}<extra></extra>",
        colorbar: {{title: {{text: "Score", side: "right"}}, thickness: 15, len: 0.8}}
    }});
}}

var figLayout = {{
    autosize: true,
    title: {{text: "Layer 0 / " + (numLayers - 1) + " — " + model + " (heads summed)", font: {{size: 16}}, x: 0.5, xanchor: "center"}},
    xaxis: {{title: "Key (attended to)", side: "bottom", tickangle: -45, automargin: true}},
    yaxis: {{title: "Query (attending from)", autorange: "reversed", automargin: false, tickfont: {{size: 10}}}},
    margin: {{l: 120, r: 80, t: 80, b: 140}}
}};

Plotly.newPlot("chart", figData, figLayout, {{responsive: true, scrollZoom: true}});
updateButtons();

getTitle = function() {{
    return "Layer " + current + " / " + (numLayers - 1) + " — " + model + " (heads summed" + (logScale ? ", log" + logBase : "") + ")";
}};

function refreshChart() {{
    var vis = [];
    var zs = [];
    for (var i = 0; i < numLayers; i++) {{
        vis.push(i === current);
        zs.push(curZ[i]);
    }}
    Plotly.restyle("chart", {{"visible": vis, "z": zs}});
    Plotly.relayout("chart", {{title: {{text: getTitle(), font: {{size: 16}}, x: 0.5, xanchor: "center"}}}});
}}

function toggleLog() {{
    logScale = !logScale;
    var btn = document.getElementById("logBtn");
    btn.classList.toggle("active", logScale);
    btn.textContent = logScale ? "Log Scale (base " + logBase + ")" : "Log Scale";
    computeZ();
    refreshChart();
}}

function changeBase(val) {{
    logBase = parseInt(val);
    document.getElementById("base-value").textContent = logBase;
    var btn = document.getElementById("logBtn");
    if (logScale) btn.textContent = "Log Scale (base " + logBase + ")";
    computeZ();
    refreshChart();
}}

function go(delta) {{
    var next = current + delta;
    if (next < 0 || next >= numLayers) return;
    current = next;
    refreshChart();
    updateButtons();
}}

function updateButtons() {{
    document.getElementById("prev").disabled = (current === 0);
    document.getElementById("next").disabled = (current === numLayers - 1);
    document.getElementById("label").textContent = "Layer " + current + " / " + (numLayers - 1);
}}

document.addEventListener("keydown", function(e) {{
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") go(-1);
    if (e.key === "ArrowRight" || e.key === "ArrowDown") go(1);
}});

window.addEventListener("resize", function() {{ Plotly.Plots.resize("chart"); }});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Attention Score Visualization")
    parser.add_argument(
        "--model", default="openbmb/MiniCPM5-1B", help="HuggingFace model name"
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog.",
        help="Input prompt",
    )
    parser.add_argument("--output", default="attention.html", help="Output HTML file")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max tokens to generate for prediction check"
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda", "mps"], help="Force device (default: auto)"
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, device=args.device)
    tokens, attentions = get_attention(tokenizer, model, args.prompt, max_new_tokens=args.max_tokens)

    num_layers = len(attentions)
    raw_z = [a.tolist() for a in attentions]
    page = HTML_TEMPLATE.format(
        num_layers=num_layers,
        num_layers_minus1=num_layers - 1,
        model=json.dumps(args.model),
        rawZ_json=json.dumps(raw_z),
        tokens_json=json.dumps(tokens),
    )

    outpath = os.path.abspath(args.output)
    with open(outpath, "w") as f:
        f.write(page)
    print(f"Saved to: {outpath}")

    if not args.no_browser:
        webbrowser.open(f"file://{outpath}")


if __name__ == "__main__":
    main()
