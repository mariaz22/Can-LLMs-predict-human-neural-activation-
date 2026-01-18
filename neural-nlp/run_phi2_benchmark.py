#!/usr/bin/env python

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("Phi-2 vs GPT-2 on Pereira2018 Benchmark")
    print("=" * 60)

    print("\n[1] Testing Phi-2 model loading...")
    try:
        from neural_nlp.models.implementations import model_pool, model_layers

        if 'phi-2' in model_pool:
            print("✓ Phi-2 found in model pool")
            print(f"  Layers: {model_layers.get('phi-2', 'N/A')[:5]}... ({len(model_layers.get('phi-2', []))} total)")
        else:
            print("✗ Phi-2 NOT found in model pool")
            print(f"  Available models: {list(model_pool.keys())[:10]}...")
            return

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[2] Loading Pereira2018 benchmark...")
    try:
        from neural_nlp.benchmarks import benchmark_pool
        benchmark = benchmark_pool['Pereira2018-encoding']
        print(f"✓ Benchmark loaded")
    except Exception as e:
        print(f"✗ Error loading benchmark: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[3] Loading Phi-2 model (this may take a while, ~5GB download)...")
    start_time = time.time()
    try:
        phi2_model = model_pool['phi-2']
        print(f"✓ Phi-2 loaded in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"✗ Error loading Phi-2: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[4] Running benchmark with Phi-2...")
    print("    (This will take a long time - encoding 627 sentences through 32 layers)")

    try:
        from neural_nlp import score

        print(f"    Using default layers (33 total)")

        start_time = time.time()
        result = score(model='phi-2', benchmark='Pereira2018-encoding')
        elapsed = time.time() - start_time

        print(f"\n✓ Benchmark completed in {elapsed:.1f}s")
        print(f"\nResults:")
        print(result)

        import pandas as pd
        result_df = result.to_dataframe()
        result_df.to_csv('phi2_pereira_results.csv')
        print(f"\nResults saved to phi2_pereira_results.csv")

    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == '__main__':
    main()
