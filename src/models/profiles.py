"""Explicit model contracts shared by inference, native training and reload."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    loader: str
    protocol: str
    eos: int
    pad: int
    template_marker: str


PROFILES = {
    "qwen3.5": ModelProfile(
        "Qwen3_5ForCausalLM", "qwen3.5-xml-single-v1", 248046, 248044, "<function="
    ),
    "lfm2.5": ModelProfile(
        "Lfm2ForCausalLM", "lfm2.5-python-single-v1", 7, 0, "<|tool_call_start|>"
    ),
}


def model_profile(config) -> ModelProfile:
    return PROFILES[config.profile]


def model_class(config):
    import transformers

    return getattr(transformers, model_profile(config).loader)


def validate_tokenizer(config, tokenizer) -> None:
    profile = model_profile(config)
    if (
        not isinstance(tokenizer.chat_template, str)
        or profile.template_marker not in tokenizer.chat_template
    ):
        raise ValueError(f"Tokenizer does not match model.profile={config.profile}")
    if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (profile.eos, profile.pad):
        raise ValueError("Tokenizer EOS/pad IDs differ from the explicit model profile")


def configure_chat_template(config, tokenizer) -> None:
    """Present LFM's documented flat tool definitions for wrapped API schemas."""
    if config.profile == "lfm2.5":
        original = "set tool = tool | tojson"
        replacement = "set tool = (tool.function if tool is mapping and tool.function is defined else tool) | tojson"
        if original not in tokenizer.chat_template and replacement not in tokenizer.chat_template:
            raise ValueError(
                "Unsupported LFM tool-definition template; inspect the pinned revision"
            )
        tokenizer.chat_template = tokenizer.chat_template.replace(original, replacement)
