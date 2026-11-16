"""Compose the real exact-kernel and streaming installers on a small H3 model."""
def test_exact_blocks_and_streaming_attention_can_be_installed_together():
    from lynnreal_hand.backends.h3.runtime import activate
    activate()
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
    from lynnreal_hand.models.h3 import _install_inference_kernels
    from exact_kernels.install import block_forward

    model = SolarMiniMaxH3Transformer3DModel.from_config(dict(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32,
        num_layers=50, num_refiner_layers=1, ffn_dim=64, in_channels=4,
        audio_in_channels=4, text_dim=16, freq_dim=8,
        time_embed_hidden_dim=32, time_embed_dim=16,
        lynnreal_variant='fast', anyflow_gate=.25))
    parameters = [(name, id(value), value.dtype) for name, value in model.named_parameters()]
    _install_inference_kernels(model, role='student')
    block = model.transformer_blocks[0]
    assert block.forward.__func__ is block_forward
    assert type(block.attn.processor).__module__ == 'inference_core'
    assert parameters == [(name, id(value), value.dtype) for name, value in model.named_parameters()]
