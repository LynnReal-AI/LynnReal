"""The startup gradient check must also work inside an inference pipeline.

CPU stand-ins exercise gradient context only; real FA3 kernels are checked on GPU.
"""
import torch


def test_startup_check_restores_gradients_inside_inference_mode(monkeypatch):
    from lynnreal_hand.backends.h3.runtime import activate
    activate()
    from diffusers.models import attention_dispatch as dispatch
    from fa3_autograd_policy import verify_gpu

    def native(q, k, v):
        return q + k + v, torch.zeros((1, 17, 2))

    def kernel(*, q, k, v, **kwargs):
        return q + k + v, torch.zeros((1, 2, 17))

    original_randn = torch.randn

    def cpu_randn(shape, *, device, dtype, requires_grad):
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        return original_randn(shape, dtype=dtype, requires_grad=requires_grad)

    monkeypatch.setattr(dispatch, '_wrapped_flash_attn_3', native)
    monkeypatch.setattr(dispatch, 'flash_attn_3_func', kernel)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch, 'randn', cpu_randn)
    with torch.inference_mode():
        verify_gpu(torch.device('cuda:0'))
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
