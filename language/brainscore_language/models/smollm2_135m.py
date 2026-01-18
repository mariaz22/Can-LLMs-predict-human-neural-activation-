from brainscore_language import model_registry
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

def smollm2_135m():
    return HuggingfaceSubject(
        model_id="HuggingFaceTB/SmolLM2-135M",
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: "model.layers.15"
        },
    )

model_registry["smollm2-135m"] = smollm2_135m
