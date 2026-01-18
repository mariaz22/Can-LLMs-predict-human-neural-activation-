from neural_nlp.benchmarks import benchmark_pool
import pandas as pd
import matplotlib.pyplot as plt

print("\n=== Loading Pereira2018 benchmark ===")
benchmark = benchmark_pool['Pereira2018-encoding']
assembly = benchmark._target_assembly.load()
stimulus_set = assembly.attrs['stimulus_set']

# =========================
# 1. STIMULUS SET (TEXTE)
# =========================
print("\n=== STIMULUS SET INFO ===")
print(stimulus_set.head())
print("Număr propoziții:", len(stimulus_set))
print("Număr povești:", stimulus_set['story'].nunique())

# Lungimea propozițiilor în cuvinte
stimulus_set['length'] = stimulus_set['sentence'].apply(lambda x: len(str(x).split()))

plt.figure()
stimulus_set['length'].hist(bins=20)
plt.xlabel("Lungime propoziție (cuvinte)")
plt.ylabel("Frecvență")
plt.title("Distribuția lungimii propozițiilor – Pereira2018")
plt.tight_layout()
plt.savefig("pereira_sentence_length.png")
print("\nSalvat grafic: pereira_sentence_length.png")

# =========================
# 2. ASSEMBLY fMRI
# =========================
print("\n=== ASSEMBLY INFO (FMRI) ===")
print("Dims:", assembly.dims)
print("Shape:", assembly.shape)
print("Coords:", list(assembly.coords))

print("\nNumăr voxeli (neuroids):", assembly.sizes.get('neuroid', 'N/A'))

# =========================
# 3. COORDONATE PE NEUROID
# =========================
neuroid = assembly['neuroid']
neuroid_coord_names = [c for c in neuroid.coords if c != 'neuroid']
print("\n=== NEUROID COORDS ===")
print("Coordonate disponibile pe neuroid:", neuroid_coord_names)

# salvăm un mic sample din coordonatele de neuroid (dacă există ceva)
if neuroid_coord_names:
    neuroid_df = neuroid.to_dataframe().reset_index()
    neuroid_df.head(20).to_csv("pereira_neuroid_coords_sample.csv", index=False)
    print("Salvat sample coordonate neuroid: pereira_neuroid_coords_sample.csv")
else:
    print("Nu există coordonate suplimentare pe dimensiunea 'neuroid' (doar indexul).")

# dacă există o coordonată care arată ca o regiune (region/roi etc.), facem și un plot
candidate_names = [c for c in neuroid_coord_names if any(
    key in c.lower() for key in ['region', 'roi', 'frois', 'froi'])]
if candidate_names:
    region_coord = candidate_names[0]
    print(f"\nFolosesc coordonata '{region_coord}' ca regiune cerebrală.")
    import numpy as np
    regions = pd.Series(neuroid.coords[region_coord].values)
    region_counts = regions.value_counts()
    print("\n=== Voxeli pe regiune ===")
    print(region_counts)

    plt.figure()
    region_counts.plot(kind='bar')
    plt.title(f'Număr voxeli per regiune ({region_coord}) – Pereira2018')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("pereira_voxels_by_region.png")
    print("Salvat grafic: pereira_voxels_by_region.png")
else:
    print("\nNu am găsit nicio coordonată de tip regiune/ROI pe neuroid. Sari peste graficul pe regiuni.")

print("\n=== EDA TERMINAT CU SUCCES ✅ ===")
