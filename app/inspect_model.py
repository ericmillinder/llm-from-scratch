import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
import torch

from model import GPT
from train import get_device

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect attention and activations during autoregressive generation.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", required=True, help="Prompt to inspect")
    parser.add_argument("--out_dir", required=True, help="Directory for plots and JSON trace")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--report_top_n", type=int, default=10,
                        help="Number of token predictions to display in the HTML report")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index for saved heatmaps; -1 means final layer")
    parser.add_argument("--head", type=int, default=0, help="Head index for saved heatmaps")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer_name = checkpoint.get("tokenizer", "gpt2")
    config = checkpoint["config"]

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoder = tiktoken.get_encoding(tokenizer_name)
    return model, encoder


def top_tokens(logits: torch.Tensor, enc, k: int = 10):
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.shape[-1]))
    results = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        results.append({
            "token_id": token_id,
            "token_text": enc.decode([token_id]),
            "prob": prob,
        })
    return results


def token_prediction_stats(logits: torch.Tensor, token_id: int) -> dict:
    probs = torch.softmax(logits, dim=-1)
    token_prob = probs[token_id]
    finite_probs = probs[torch.isfinite(logits)]
    rank = int((finite_probs > token_prob).sum().item()) + 1
    return {
        "rank": rank,
        "prob": float(token_prob.item()),
    }


def compute_hidden_norms(hidden_states: list[torch.Tensor]) -> list[list[float]]:
    norms = []
    for state in hidden_states:
        token_norms = state[0].norm(dim=-1).detach().cpu().tolist()
        norms.append(token_norms)
    return norms


def save_attention_heatmap(attn: np.ndarray, labels: list[str], out_path: Path, title: str):
    if plt is None:
        raise RuntimeError("matplotlib is required to save heatmaps. Install it with `uv sync` or `pip install matplotlib`.")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.45)))
    im = ax.imshow(attn, cmap="magma", interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Attended token")
    ax.set_ylabel("Query token")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_activation_heatmap(norms: np.ndarray, labels: list[str], out_path: Path):
    if plt is None:
        raise RuntimeError("matplotlib is required to save heatmaps. Install it with `uv sync` or `pip install matplotlib`.")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, norms.shape[0] * 0.6)))
    im = ax.imshow(norms, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(norms.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(norms.shape[0])], fontsize=8)
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")
    ax.set_title("Per-token hidden-state norm by layer")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_token_label(token: str) -> str:
    if token == "\n":
        return "\\n"
    if token == "\t":
        return "\\t"
    if token.strip() == "":
        return repr(token)[1:-1]
    return token


def build_story_html(trace: dict) -> str:
    parts = []
    prompt_len = len(trace["prompt_tokens"])

    for index, token in enumerate(trace["prompt_tokens"]):
        label = html.escape(format_token_label(token))
        parts.append(
            f'<span class="story-token prompt-token" title="Prompt token {index}">{label}</span>'
        )

    for step in trace["steps"]:
        step_id = step["step"]
        token = html.escape(format_token_label(step["selected_token_text"]))
        parts.append(
            f'<a class="story-token generated-token" href="#step-{step_id}" '
            f'title="Generated token {prompt_len + step_id} at step {step_id}">{token}</a>'
        )

    return "".join(parts)


def build_top_predictions_html(predictions: list[dict]) -> str:
    rows = []
    for rank, pred in enumerate(predictions, start=1):
        token = html.escape(format_token_label(pred["token_text"]))
        prob_pct = pred["prob"] * 100.0
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><code>{token}</code></td>"
            f"<td>{pred['token_id']}</td>"
            f"<td>{prob_pct:.2f}%</td>"
            f'<td><div class="bar-track"><div class="bar-fill" style="width: {prob_pct:.2f}%"></div></div></td>'
            "</tr>"
        )
    return "".join(rows)


def build_selected_prediction_html(step: dict) -> str:
    selected = step.get("selected_prediction", {})
    raw = selected.get("raw", {})
    sampling = selected.get("sampling", {})

    if not raw or not sampling:
        return ""

    raw_prob = raw["prob"] * 100.0
    sampling_prob = sampling["prob"] * 100.0
    return (
        '<p class="prediction-note">'
        f'Raw model rank: <strong>{raw["rank"]}</strong> '
        f'({raw_prob:.2f}% before top-k filtering). '
        f'Sampling-pool rank: <strong>{sampling["rank"]}</strong> '
        f'({sampling_prob:.2f}% after top-k renormalization).'
        '</p>'
    )


def attention_focus_for_step(step: dict, layer_idx: int, head_idx: int, top_n: int = 5) -> list[dict]:
    attn = np.array(step["attentions"][layer_idx][head_idx], dtype=float)
    if attn.size == 0:
        return []

    next_token_query = attn[-1]
    top_indices = np.argsort(-next_token_query)[:top_n]
    focus = []
    for index in top_indices:
        token_text = step["context_tokens"][int(index)]
        focus.append({
            "context_position": int(index),
            "token_id": step["context_token_ids"][int(index)],
            "token_text": token_text,
            "attention": float(next_token_query[int(index)]),
        })
    return focus


def build_attention_focus_chips(focus: list[dict]) -> str:
    chips = []
    for item in focus:
        token = html.escape(format_token_label(item["token_text"]))
        attention_pct = item["attention"] * 100.0
        chips.append(
            '<span class="focus-chip">'
            f'<code>{token}</code>'
            f'<span>{attention_pct:.1f}%</span>'
            f'<small>pos {item["context_position"]}</small>'
            '</span>'
        )
    return "".join(chips)


def build_attention_summary_html(trace: dict) -> str:
    rows = []
    prompt_len = len(trace["prompt_tokens"])
    for step in trace["steps"]:
        generated_token = html.escape(format_token_label(step["selected_token_text"]))
        focus_html = build_attention_focus_chips(step.get("attention_focus", []))
        rows.append(
            "<tr>"
            f'<td><a class="back-link" href="#step-{step["step"]}">Step {step["step"]}</a></td>'
            f"<td>{prompt_len + step['step']}</td>"
            f"<td><code>{generated_token}</code></td>"
            f'<td><div class="focus-list">{focus_html}</div></td>'
            "</tr>"
        )

    rows_html = "".join(rows) if rows else '<tr><td colspan="4">No generated steps.</td></tr>'
    return (
        '<section class="summary-panel" id="attention-summary">'
        '<div class="section-header"><h2>Attention Summary</h2><a class="back-link" href="#story">Story</a></div>'
        '<table><thead><tr><th>Step</th><th>Token #</th><th>Generated</th><th>Most-attended earlier tokens</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
        '</section>'
    )


def build_step_section_html(step: dict, layer_idx: int, head_idx: int) -> str:
    selected_token = html.escape(format_token_label(step["selected_token_text"]))
    context_preview = "".join(html.escape(format_token_label(token)) for token in step["context_tokens"][-24:])
    top_predictions_html = build_top_predictions_html(step["top_predictions"])
    selected_prediction_html = build_selected_prediction_html(step)
    attention_focus_html = build_attention_focus_chips(step.get("attention_focus", []))
    attention_file = f"step_{step['step']:03d}_layer_{layer_idx}_head_{head_idx}_attention.png"
    activation_file = f"step_{step['step']:03d}_activation_norms.png"

    return (
        f'<section class="step-card" id="step-{step["step"]}">'
        f'<div class="step-header"><h2>Step {step["step"]}</h2>'
        f'<a class="back-link" href="#story">Story</a></div>'
        f'<p class="step-summary">Next token: <code>{selected_token}</code> '
        f'(<span class="token-id">id {step["selected_token_id"]}</span>)</p>'
        f'<p class="context-preview">{context_preview}</p>'
        f'<div class="focus-block"><h3>Most-attended earlier tokens</h3><div class="focus-list">{attention_focus_html}</div></div>'
        '<div class="media-grid">'
        f'<figure><a href="{attention_file}"><img src="{attention_file}" alt="Attention heatmap for step {step["step"]}"></a>'
        f'<figcaption>Layer {layer_idx}, head {head_idx} attention</figcaption></figure>'
        f'<figure><a href="{activation_file}"><img src="{activation_file}" alt="Activation heatmap for step {step["step"]}"></a>'
        '<figcaption>Hidden-state norm map</figcaption></figure>'
        '</div>'
        '<div class="prediction-block">'
        '<h3>Displayed prediction distribution</h3>'
        f'{selected_prediction_html}'
        '<table><thead><tr><th>#</th><th>Token</th><th>ID</th><th>Prob.</th><th></th></tr></thead>'
        f'<tbody>{top_predictions_html}</tbody></table>'
        '</div>'
        '</section>'
    )


def save_html_report(trace: dict, out_dir: Path, layer_idx: int, head_idx: int):
    title = html.escape(trace["prompt"])
    story_html = build_story_html(trace)
    attention_summary_html = build_attention_summary_html(trace)
    generated_text = html.escape(trace["generated_text"])
    step_sections = "".join(build_step_section_html(step, layer_idx, head_idx) for step in trace["steps"])

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generation Trace Report</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --accent-2: #38bdf8;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .lede, .story-panel, .summary-panel, .step-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .lede {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .lede h1 {{
      margin: 0 0 10px;
      font-size: 1.5rem;
    }}
    .lede p {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .story-panel, .summary-panel {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .story-text {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.95rem;
    }}
    .story-token {{
      display: inline;
      padding: 1px 0;
      text-decoration: none;
      color: inherit;
    }}
    .prompt-token {{
      color: var(--muted);
    }}
    .generated-token {{
      color: var(--accent-2);
      border-bottom: 1px dotted var(--accent-2);
    }}
    .steps {{
      display: grid;
      gap: 16px;
    }}
    .step-card {{
      padding: 18px;
    }}
    .step-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .section-header h2, .step-header h2, .prediction-block h3, .focus-block h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    .back-link {{
      color: var(--accent-2);
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .step-summary, .context-preview, .prediction-note {{
      margin: 8px 0 0;
    }}
    .prediction-note {{
      color: var(--muted);
    }}
    .context-preview {{
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin: 16px 0;
    }}
    .focus-block {{
      margin-top: 14px;
    }}
    .focus-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }}
    .focus-chip {{
      display: inline-flex;
      align-items: baseline;
      gap: 7px;
      max-width: 100%;
      padding: 5px 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
    }}
    .focus-chip code {{
      max-width: 160px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .focus-chip span {{
      color: var(--accent);
      font-size: 0.86rem;
    }}
    .focus-chip small {{
      color: var(--muted);
      font-size: 0.78rem;
    }}
    figure {{
      margin: 0;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figcaption {{
      padding: 10px 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 8px 10px;
      border-top: 1px solid var(--border);
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .bar-track {{
      width: 100%;
      min-width: 120px;
      height: 8px;
      background: #0b1220;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    @media (max-width: 700px) {{
      main {{ padding: 14px; }}
      .media-grid {{ grid-template-columns: 1fr; }}
      th:nth-child(3), td:nth-child(3) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="lede">
      <h1>Generation Trace Report</h1>
      <p><strong>Prompt:</strong> {title}</p>
      <p><strong>Generated text:</strong> <span class="story-text">{generated_text}</span></p>
      <p><strong>Artifacts:</strong> <a class="back-link" href="trace.json">trace.json</a></p>
    </section>
    <section class="story-panel" id="story">
      <h2>Story View</h2>
      <p class="story-text">{story_html}</p>
    </section>
    {attention_summary_html}
    <section class="steps">
      {step_sections}
    </section>
  </main>
</body>
</html>
"""
    with open(out_dir / "report.html", "w") as f:
        f.write(report_html)


def resolve_trace_layer_head(trace: dict, layer_idx: int, head_idx: int) -> tuple[int, int]:
    if not trace["steps"]:
        return max(layer_idx, 0), max(head_idx, 0)

    layer_count = len(trace["steps"][0]["attentions"])
    resolved_layer = layer_idx if layer_idx >= 0 else layer_count - 1
    resolved_layer = max(0, min(resolved_layer, layer_count - 1))

    head_count = len(trace["steps"][0]["attentions"][resolved_layer])
    resolved_head = max(0, min(head_idx, head_count - 1))
    return resolved_layer, resolved_head


def add_attention_focus(trace: dict, layer_idx: int, head_idx: int):
    for step in trace["steps"]:
        step["attention_focus"] = attention_focus_for_step(step, layer_idx, head_idx)


@torch.no_grad()
def trace_generation(model: GPT, enc, prompt: str, max_new_tokens: int, temperature: float, top_k: int,
                     report_top_n: int, seed: int):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    prompt_tokens = enc.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    steps = []
    generated_token_ids = []

    for step_idx in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        outputs = model(idx_cond, output_attentions=True, output_hidden_states=True)
        raw_logits = outputs["logits"][:, -1, :] / temperature
        sampling_logits = raw_logits.clone()

        if top_k > 0:
            values, _ = torch.topk(sampling_logits, min(top_k, sampling_logits.shape[-1]))
            sampling_logits[sampling_logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(sampling_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()

        visible_token_ids = idx_cond[0].detach().cpu().tolist()
        visible_tokens = [enc.decode([token_id]) for token_id in visible_token_ids]
        hidden_norms = compute_hidden_norms(outputs["hidden_states"])

        attentions = []
        for layer_attn in outputs["attentions"]:
            attentions.append(layer_attn[0].detach().cpu().tolist())

        steps.append({
            "step": step_idx,
            "context_token_ids": visible_token_ids,
            "context_tokens": visible_tokens,
            "top_predictions": top_tokens(sampling_logits[0], enc, k=report_top_n),
            "selected_token_id": next_token_id,
            "selected_token_text": enc.decode([next_token_id]),
            "selected_prediction": {
                "raw": token_prediction_stats(raw_logits[0], next_token_id),
                "sampling": token_prediction_stats(sampling_logits[0], next_token_id),
            },
            "sampling_top_k": top_k,
            "report_top_n": report_top_n,
            "hidden_state_norms": hidden_norms,
            "attentions": attentions,
        })

        if next_token_id == enc.eot_token:
            break

        generated_token_ids.append(next_token_id)
        idx = torch.cat([idx, next_token], dim=1)

    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_tokens,
        "prompt_tokens": [enc.decode([token_id]) for token_id in prompt_tokens],
        "generated_token_ids": generated_token_ids,
        "generated_text": enc.decode(generated_token_ids),
        "steps": steps,
    }


def save_trace_artifacts(trace: dict, out_dir: Path, layer_idx: int, head_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_layer, resolved_head = resolve_trace_layer_head(trace, layer_idx, head_idx)
    add_attention_focus(trace, resolved_layer, resolved_head)

    for step in trace["steps"]:
        labels = [token.replace("\n", "\\n") for token in step["context_tokens"]]
        attn = np.array(step["attentions"][resolved_layer][resolved_head], dtype=float)
        save_attention_heatmap(
            attn,
            labels,
            out_dir / f"step_{step['step']:03d}_layer_{resolved_layer}_head_{resolved_head}_attention.png",
            title=f"Step {step['step']} | layer {resolved_layer} | head {resolved_head} | next={step['selected_token_text']!r}",
        )

        norms = np.array(step["hidden_state_norms"], dtype=float)
        save_activation_heatmap(
            norms,
            labels,
            out_dir / f"step_{step['step']:03d}_activation_norms.png",
        )

    with open(out_dir / "trace.json", "w") as f:
        json.dump(trace, f, indent=2)

    save_html_report(trace, out_dir, resolved_layer, resolved_head)


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    model, enc = load_checkpoint(args.checkpoint, device)

    trace = trace_generation(
        model,
        enc,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        report_top_n=args.report_top_n,
        seed=args.seed,
    )
    save_trace_artifacts(trace, out_dir, args.layer, args.head)

    print(f"Saved trace to {out_dir}")
    print(f"Open report: {out_dir / 'report.html'}")
    print(f"Generated text:\n{trace['generated_text']}")


if __name__ == "__main__":
    main()
