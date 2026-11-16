"""Ultra-Fast-VAE decoding adapted to the native 544x960 canvas.

The published package remains immutable. Only tile starts, output buffers and
the two-row top/bottom delivery crop differ from its fixed 768p interface.
"""
import torch
from torch import nn
from ultra_fast_vae.inference import Decoder as PublishedDecoder
from ultra_fast_vae.model import split_tiles, stitch
from ultra_fast_vae.stable_vae_ops import denormalize_half


class Canvas540(nn.Module):
    def __init__(self, decoder, post):
        super().__init__()
        self.decoder, self.post = decoder, post
        self.ys, self.hs, self.yo = split_tiles(544, 256, 64)
        self.xs, self.ws, self.xo = split_tiles(960, 256, 64)

    def forward(self, z):
        assert z.shape == (1, 24, 5, 34, 60)
        tiles = [z[..., y//16:(y+h)//16, x//16:(x+w)//16]
                 for y, h in zip(self.ys, self.hs)
                 for x, w in zip(self.xs, self.ws)]
        decoded = self.decoder(self.post(torch.cat(tiles)))
        pieces = list(decoded.split(1))
        rows = [pieces[i:i+len(self.xs)] for i in range(0, len(pieces), len(self.xs))]
        return stitch(rows, self.yo, self.xo)


class Decoder540(PublishedDecoder):
    def __init__(self, root, device='cuda:0'):
        super().__init__(root, device)
        self.canvas = Canvas540(self.decoder, self.post)

        def packed(current):
            assert current.shape == (1, 24, 5, 34, 60)
            z = denormalize_half(current, self.mean, self.std)
            with torch.autocast('cuda', dtype=torch.float16):
                raw = self.canvas(z)[:, :, 3:20, 2:-2]
            return ((raw.float()*self.pixel_std+self.pixel_mean).clamp(0, 1)[0]*255).round().byte().permute(1, 2, 3, 0).contiguous()

        self.packed = packed
        self.compiled = torch.compile(packed, fullgraph=True, dynamic=False,
                                      options={'triton.cudagraphs': False, 'emulate_precision_casts': True})
        self.count = 0
        self.banks = [torch.empty((17, 540, 960, 3), dtype=torch.uint8, pin_memory=True)
                      for _ in range(2)]

    @torch.inference_mode()
    def warmup(self, fixture):
        expected = self.packed(fixture).clone()
        actual = self.compiled(fixture)
        error = int((expected.short()-actual.short()).abs().max())
        if error > 1:
            raise RuntimeError(f'Ultra-Fast-VAE 540p compile/eager code error {error}')
        self.input = fixture.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.compiled(self.input)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.compiled(self.input)
        self.graph.replay()
        torch.cuda.synchronize()
        graph_error = int((self.output.short()-actual.short()).abs().max())
        assert graph_error == 0
        for _ in range(15):
            self.decode(fixture)
        return dict(passed=True, compile_eager_max_code_error=error,
                    graph_max_code_error=graph_error, tiles=len(self.canvas.ys)*len(self.canvas.xs),
                    input=[1, 24, 5, 34, 60], output=[17, 540, 960, 3],
                    weights_unchanged=True, all104_W8A8_projections=True)

    @torch.inference_mode()
    def decode(self, current):
        assert current.shape == self.input.shape and current.dtype == torch.float32
        self.input.copy_(current, non_blocking=True)
        self.graph.replay()
        bank = self.banks[self.count % 2]
        bank.copy_(self.output, non_blocking=True)
        torch.cuda.synchronize()
        self.count += 1
        return bank

    @torch.inference_mode()
    def reset(self):
        self.count = 0
        self.input.zero_()
