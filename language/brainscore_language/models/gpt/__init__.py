from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry['openai-gpt'] = lambda: HuggingfaceSubject(model_id='openai-gpt', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['distilgpt2'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'})

model_registry['gpt2'] = lambda: HuggingfaceSubject(model_id='gpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt2-medium'] = lambda: HuggingfaceSubject(model_id='gpt2-medium', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.22'})

model_registry['gpt2-large'] = lambda: HuggingfaceSubject(model_id='gpt2-large', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.33'})

model_registry['gpt2-xl'] = lambda: HuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43'})

model_registry['gpt-neo-125m'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-125m', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt-neo-2.7B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-2.7B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.31'})

model_registry['gpt-neo-1.3B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-1.3B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.18'})

# SmolLM2 (CPU-friendly)
model_registry['smollm2-135m'] = lambda: HuggingfaceSubject(
    model_id='HuggingFaceTB/SmolLM2-135M',
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'model.layers.15'
    }
)


# --- SmolLM2 layer sweep helpers (auto-added) ---
model_registry['smollm2-135m_l2'] = lambda: HuggingfaceSubject(
    model_id='HuggingFaceTB/SmolLM2-135M',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.2'}
)
model_registry['smollm2-135m_l8'] = lambda: HuggingfaceSubject(
    model_id='HuggingFaceTB/SmolLM2-135M',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.8'}
)
model_registry['smollm2-135m_l15'] = lambda: HuggingfaceSubject(
    model_id='HuggingFaceTB/SmolLM2-135M',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.15'}
)
# --- end SmolLM2 layer sweep helpers ---


# --- Pythia (EleutherAI) ---
model_registry['pythia-70m_l2'] = lambda: HuggingfaceSubject(
    model_id='EleutherAI/pythia-70m-deduped',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'gpt_neox.layers.2'}
)
model_registry['pythia-70m_last'] = lambda: HuggingfaceSubject(
    model_id='EleutherAI/pythia-70m-deduped',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'gpt_neox.layers.5'}
)
# --- end Pythia ---


# --- Gemma 2B (google/gemma-2b) ---
# We expose layer 2 and last (17).

model_registry['gemma-2b_l2'] = lambda: HuggingfaceSubject(
    model_id='google/gemma-2b',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.2'}
)

model_registry['gemma-2b_last'] = lambda: HuggingfaceSubject(
    model_id='google/gemma-2b',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'model.layers.17'}
)
# --- end Gemma ---
