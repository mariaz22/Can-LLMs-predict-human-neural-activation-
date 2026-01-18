import time
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from brainscore.metrics.cka import CKACrossValidated
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models.implementations import load_model, model_layers


warnings.filterwarnings("ignore", category=FutureWarning)



LANGUAGE_BENCHMARKS = [
    "Pereira2018-encoding",
    "Pereira2018-language",
    "Blank2014fROI-encoding",
    "Fedorenko2016v3-encoding",
]


def run_language_benchmarks(model_identifier: str = "gpt2") -> pd.DataFrame:
    print(f"Loading model: {model_identifier}")
    model = load_model(model_identifier)
    print("Model loaded:", model.identifier)

    results: List[Dict] = []

    for benchmark_name in LANGUAGE_BENCHMARKS:
        if benchmark_name not in benchmark_pool:
            print(f"[WARN] Benchmark '{benchmark_name}' not found in benchmark_pool. Skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"Running benchmark: {benchmark_name}")
        benchmark = benchmark_pool[benchmark_name]

        start = time.time()
        score = benchmark(model)
        elapsed = time.time() - start

        try:
            center = float(score.sel(aggregation="center").values)
        except Exception:
            center = np.nan

        try:
            error = float(score.sel(aggregation="error").values)
        except Exception:
            error = np.nan

        ceiling_center = np.nan
        ceiling_error = np.nan
        ceiling = score.attrs.get("ceiling", None)
        if ceiling is not None:
            try:
                ceiling_center = float(ceiling.sel(aggregation="center").values)
            except Exception:
                pass
            try:
                ceiling_error = float(ceiling.sel(aggregation="error").values)
            except Exception:
                pass

        results.append(
            dict(
                benchmark=benchmark_name,
                model=model_identifier,
                score_center=center,
                score_error=error,
                ceiling_center=ceiling_center,
                ceiling_error=ceiling_error,
                runtime_seconds=elapsed,
            )
        )

        print(f"Finished {benchmark_name}")
        print(f"  score_center   = {center}")
        print(f"  score_error    = {error}")
        print(f"  ceiling_center = {ceiling_center}")
        print(f"  ceiling_error  = {ceiling_error}")
        print(f"  runtime [s]    = {elapsed:.1f}")

    df = pd.DataFrame(results)
    csv_path = f"gpt2_language_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print("All benchmark results saved to:", csv_path)
    print(df)
    return df



def compute_layer_cka(model_identifier: str = "gpt2"):
    print("\n" + "=" * 80)
    print("Computing CKA similarity between GPT-2 layers on a small sentence set")

    model = load_model(model_identifier)
    layers = model_layers[model_identifier]
    print("Available layers:", layers)

    preferred_layers = [
        "encoder.h.0",
        "encoder.h.3",
        "encoder.h.6",
        "encoder.h.9",
        "encoder.h.11",
    ]
    selected_layers = [l for l in layers if l in preferred_layers]
    if not selected_layers:
        selected_layers = layers[:5]

    print("Selected layers for CKA:", selected_layers)

    sentences = [
        "The cat sat on the mat.",
        "The dog chased the cat across the garden.",
        "A neural network processed the sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Language models can be compared to brain activity.",
        "Brainscore compares models to neural data.",
        "The story continued with an unexpected twist.",
        "Reading sentences engages multiple brain regions.",
        "GPT-2 is a transformer-based language model.",
        "Functional MRI measures brain responses to language.",
    ]

    activations = {}
    for layer in selected_layers:
        print(f"Extracting activations for layer: {layer}")
        acts = model(stimuli=sentences, layers=[layer])
        activations[layer] = acts.values
        print("  shape:", activations[layer].shape)

    cka = CKACrossValidated(split=2)

    n_layers = len(selected_layers)
    cka_matrix = np.zeros((n_layers, n_layers), dtype=np.float32)

    for i, li in enumerate(selected_layers):
        for j, lj in enumerate(selected_layers):
            if j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
                continue
            if li == lj:
                cka_matrix[i, j] = 1.0
                continue
            print(f"Computing CKA between {li} and {lj} ...")
            score = cka(activations[li], activations[lj])
            val = float(score.values)
            cka_matrix[i, j] = val
            cka_matrix[j, i] = val

    print("\nCKA similarity matrix (rows/cols = layers):")
    print(selected_layers)
    print(cka_matrix)

    if HAS_MPL:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cka_matrix,
            xticklabels=selected_layers,
            yticklabels=selected_layers,
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
        )
        plt.title(f"CKA similarity between layers ({model_identifier})")
        plt.tight_layout()
        png_path = f"gpt2_layer_cka.png"
        plt.savefig(png_path, dpi=150)
        print("Saved CKA heatmap to:", png_path)
    else:
        print("Matplotlib/Seaborn not available, skipping heatmap plot.")


if __name__ == "__main__":
    df = run_language_benchmarks("gpt2")
    compute_layer_cka("gpt2")
