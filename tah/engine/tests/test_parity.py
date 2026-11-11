"""Revised G-TF arithmetic, independent of models and GPU availability."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tah.engine.parity import assert_gtf, compare_output, derive_decision_limits, gtf_summary
from tah.engine.reference_capture import summarize


def comparison():
    logits = torch.zeros(6, 3)
    reference_visited = torch.ones(3, 6, dtype=torch.bool)
    visited = reference_visited.clone()
    visited[1:, 0] = False
    visited[2, 1] = False
    reference_decision = reference_visited.clone()
    reference_decision[2] = False
    decision = reference_decision & visited
    decision[0, 0] = False
    decision[1, 1] = False
    probability = torch.full((3, 6), .51)
    probability[0, 0] = .49
    probability[1, 1] = .49
    output = SimpleNamespace(logits=logits, iter_count=visited.sum(0), visited=visited,
        decision=decision, continue_probability=probability, decision_logits=probability,
        decider_hidden=torch.empty(3, 0, 0, 1))
    template = dict(visited=reference_visited, decision=reference_decision,
        continue_probability=torch.full((3, 6), .51), pass_count=reference_visited.sum(0),
        final_topk_values=logits, final_topk_indices=torch.arange(3).expand(6, 3),
        logit_positions=torch.arange(6), final_logits_sample=logits)
    row = compare_output(template, output, logits, .5)
    noise = summarize([row])
    noise["reference_max_flip_margin"] = max(d["max_flip_margin"] for d in row["per_depth"])
    return row, noise


def test_common_visited_flips_exclude_cascades_and_count_tokens():
    row, noise = comparison()
    stats = assert_gtf([row], noise)
    assert [d["common_visited_compared"] for d in stats["per_depth"]] == [6, 5, 4]
    assert [d["common_decision_mismatches"] for d in stats["per_depth"]] == [1, 1, 0]
    assert stats["visited_mismatches"] == 3
    assert stats["pass_count_mismatch_rate"] == 2 / 6
    limits = derive_decision_limits(noise)
    assert [d["flip_rate_limit"] for d in limits["per_depth"]] == [1.5 / 6, 1.5 / 5, 0.]


def test_zero_reference_flip_rate_requires_exact_match():
    row, noise = comparison()
    noise["per_depth"][0]["common_decision_mismatches"] = 0
    with pytest.raises(AssertionError):
        assert_gtf([row], noise)


def test_pass_count_rate_is_a_separate_gate():
    row, noise = comparison()
    noise["pass_count_mismatches"] = 1
    with pytest.raises(AssertionError):
        assert_gtf([row], noise)


def test_margin_is_measured_and_gated_even_when_rate_passes():
    row, noise = comparison()
    noise["reference_max_flip_margin"] = .02
    assert derive_decision_limits(noise)["flip_margin_limit"] == .03
    row["per_depth"][0]["max_flip_margin"] = .031
    with pytest.raises(AssertionError):
        assert_gtf([row], noise)
    del noise["reference_max_flip_margin"]
    with pytest.raises(AssertionError, match="missing measured"):
        derive_decision_limits(noise)


def test_terminal_visitation_difference_is_not_a_terminal_flip():
    row, noise = comparison()
    stats = gtf_summary([row])
    assert stats["per_depth"][2]["decision_mismatches"] == 2  # legacy union diagnostic
    assert stats["per_depth"][2]["common_flip_rate"] == 0
    assert_gtf([row], noise)


def test_no_logit_epsilon_floor():
    row, noise = comparison()
    altered = deepcopy(row)
    altered["logit_abs_sum"] = .1
    with pytest.raises(AssertionError):
        assert_gtf([altered], noise)


def test_generated_mlp_count_floor_preserves_structural_exactness():
    from tah.engine.parity import derive_decision_limits
    noise = dict(per_depth=[dict(depth=d, decision_compared=100, visited_mismatches=0,
                                common_decision_mismatches=f) for d, f in [(1,0),(2,1),(3,0)]],
                 reference_max_flip_margin=.02, pass_count_mismatches=1, top1_compared=100,
                 mlp_count_floor=2, learned_depths=[1,2])
    limits = derive_decision_limits(noise)
    assert [d['flip_rate_limit'] for d in limits['per_depth']] == [.03,.03,0]
    assert limits['pass_count_mismatch_rate_limit'] == .03
    noise['learned_depths'] = []
    noise['pass_count_mismatches'] = 0
    for depth in noise['per_depth']:
        depth['common_decision_mismatches'] = 0
    limits = derive_decision_limits(noise)
    assert all(d['flip_rate_limit'] == 0 for d in limits['per_depth'])
    assert limits['pass_count_mismatch_rate_limit'] == 0
