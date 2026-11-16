"""Schedule specialization preserves modulation values and rejects other times."""
import copy

import pytest
import torch
from torch import nn

from lynnreal_hand.inference.adaln import schedule_rows, select_adaln, specialize_adaln


class Modulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 6 * 8 * 3)

    def forward(self, temb):
        return self.linear(torch.nn.functional.silu(temb).to(self.linear.weight.dtype)).view(-1, 6 * 8).chunk(6, dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedder = nn.Linear(1, 64)
        self.transformer_blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for block in self.transformer_blocks:
            block.adaln_proj = Modulation()

    def _time_condition(self, times, targets=None, *, parameter_dtype):
        value = self.time_embedder(times[:, None].to(parameter_dtype(self.time_embedder)))
        return value if targets is None else .75 * value + .25 * self.time_embedder(targets[:, None])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('stage,sampling', [
    ('stage1', dict(schedule_points=30, shift=3.)),
    ('stage3', dict(steps=3, shift=3.)),
])
def test_all_scheduled_modulations_match_original_without_projection_parameters(stage, sampling, dtype):
    original = Model().eval().requires_grad_(False)
    for block in original.transformer_blocks:
        block.adaln_proj.to(dtype=dtype)
    specialized = copy.deepcopy(original)
    config = dict(stage=stage, sampling=sampling)
    result = specialize_adaln(specialized, config, 'cpu')
    assert result['schedule_entries'] == (29 if stage == 'stage1' else 5)
    assert not any('adaln_proj' in name for name, _ in specialized.named_parameters())
    assert not any('adaln_proj' in name for name in specialized.state_dict())
    # Reuse the schedule twice, as in repeated clips and positive/negative CFG.
    with torch.no_grad():
        for _ in range(2):
            for times, targets in schedule_rows(config, 'cpu'):
                t = torch.tensor(times)
                r = torch.tensor(targets) if targets is not None else None
                select_adaln(specialized, t, r)
                embedding = original._time_condition(t, r, parameter_dtype=lambda m: next(m.parameters()).dtype)
                for before, after in zip(original.transformer_blocks, specialized.transformer_blocks):
                    for expected, actual in zip(before.adaln_proj(embedding), after.adaln_proj(embedding)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    specialized.to('cpu')
    select_adaln(specialized, *schedule_rows(config, 'cpu')[0])


def test_unknown_time_pairs_and_unselected_or_wrong_row_tables_fail():
    model = Model().eval().requires_grad_(False)
    specialize_adaln(model, dict(stage='stage3', sampling=dict(steps=3, shift=3.)), 'cpu')
    with pytest.raises(RuntimeError, match='Select'):
        model.transformer_blocks[0].adaln_proj(torch.zeros(1, 64))
    with pytest.raises(ValueError, match='outside'):
        select_adaln(model, [.123], [.456])
    select_adaln(model, [.999, 1.], [.999, 1.])
    with pytest.raises(ValueError, match='time rows'):
        model.transformer_blocks[0].adaln_proj(torch.zeros(1, 64))
    with pytest.raises(ValueError, match='already'):
        specialize_adaln(model, dict(stage='stage3', sampling=dict(steps=3, shift=3.)), 'cpu')
