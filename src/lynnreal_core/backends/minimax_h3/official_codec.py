"""Native H3 first-frame/HAND encoding and joint image/text conditions."""
from typing import Any
import numpy as np
from lynnreal_core.errors import DataContractError
H3_TEXT_ENCODER_LAYER=50

def encode_joint_prompt_condition(
    keyframe: Any, caption: str, *, processor: Any, tokenizer: Any, text_encoder: Any, device: Any
) -> tuple[Any, Any]:
    import torch
    from PIL import Image

    if not isinstance(keyframe, Image.Image):
        array = np.asarray(keyframe, dtype=np.uint8)
        keyframe = Image.fromarray(array, mode="RGB")
    vision = processor.image_processor(images=[keyframe], return_tensors="pt")
    grid = vision["image_grid_thw"]
    merge = processor.image_processor.merge_size**2
    image_tokens = int(grid[0].prod()) // merge
    label_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    vision_ids = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    caption_ids = tokenizer(str(caption), add_special_tokens=False)["input_ids"]
    token_ids = label_ids + vision_ids + caption_ids
    tags = [1] * len(label_ids) + [0] * len(vision_ids) + [1] * len(caption_ids)
    if not token_ids:
        raise DataContractError("H3 Qwen presentation produced no tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    token_types = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long,
        device=device,
    )
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=token_types,
        use_cache=False,
        output_hidden_states=True,
        pixel_values=vision["pixel_values"].to(device, text_encoder.dtype),
        image_grid_thw=grid.to(device),
    )
    if len(outputs.hidden_states) <= H3_TEXT_ENCODER_LAYER:
        raise DataContractError("H3 Qwen has no hidden state 50")
    hidden = outputs.hidden_states[H3_TEXT_ENCODER_LAYER][0].to(torch.bfloat16).contiguous().cpu()
    tags_tensor = torch.tensor(tags, dtype=torch.long)
    if tuple(hidden.shape) != (len(token_ids), 5120):
        raise DataContractError("H3 Qwen hidden state has the wrong shape")
    return hidden, tags_tensor

class OfficialH3Codec:
    def _video_latents(self, pixels: Any, *, seed: int) -> Any:
        import torch
        mean = torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=pixels.device).view(1, -1, 1, 1, 1)
        normalized = (pixels.float().div(255.0) - pixel_mean) / pixel_std
        posterior = self.video_vae.encode(normalized, return_dict=False)[0]
        latent = posterior.sample(generator=torch.Generator().manual_seed(int(seed)))
        latent = latent.to(torch.float16).float().cpu()
        return ((latent - mean) / std).to(torch.bfloat16).contiguous()
