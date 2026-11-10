"""Run: python -m unittest research.bandit_router.test_core -v."""
import unittest
from types import SimpleNamespace

import torch

from .core import bandit_loss, calibrate_budget, features, fork_cache, paired_step, problem_split


class CounterfactualModel:
    """History-sensitive fixture: the current action writes to its own cache."""
    def __init__(self, decider, imitate_continue):
        self.decider = decider
        self.imitate_continue = imitate_continue
        self.histories = []

    def __call__(self, input_ids, past_key_values, use_cache):
        history = past_key_values["depths"]
        self.histories.append(history.copy())
        self.decider.state = torch.tensor([[float(sum(history)), float(input_ids.item())]])
        self.decider.imitation_decision = torch.tensor([self.imitate_continue])
        depth = 2 if self.decider.policy == "continue" else 1
        history.append(depth)
        # Native depth-2 accumulation is represented by final branch logits.
        logits = torch.tensor([[[0., 2. if depth == 1 else -2.]]])
        return SimpleNamespace(logits=logits, iter_count=torch.tensor([[depth]]), past_key_values=past_key_values)


class PrototypeTests(unittest.TestCase):
    def test_both_branches_keep_fixed_history_and_commit_only_imitation(self):
        for imitate_continue in (False, True):
            decider = SimpleNamespace(policy="imitation")
            model = CounterfactualModel(decider, imitate_continue)
            parent = {"depths": [2, 1, 2]}
            _, losses, depth, committed = paired_step(model, decider, parent, torch.tensor([[3]]), torch.tensor(1))
            self.assertEqual(model.histories, [[2, 1, 2], [2, 1, 2]])
            self.assertEqual(parent["depths"], [2, 1, 2])
            self.assertEqual(committed["depths"], [2, 1, 2, 2 if imitate_continue else 1])
            self.assertEqual(depth, 2 if imitate_continue else 1)
            self.assertGreater(float(losses[1]), float(losses[0]))
            self.assertEqual(decider.policy, "imitation")

    def test_cache_tensor_storage_is_independent(self):
        parent = {"kv": torch.ones(2), "metadata": {"depths": [1]}}
        branch = fork_cache(parent)
        branch["kv"].zero_()
        branch["metadata"]["depths"].append(2)
        self.assertTrue(torch.equal(parent["kv"], torch.ones(2)))
        self.assertEqual(parent["metadata"]["depths"], [1])

    def test_reward_gradient_encourages_gain_and_discourages_overthinking(self):
        logits = torch.zeros(2, requires_grad=True)
        losses = torch.tensor([[2., 1.], [1., 2.]])
        bandit_loss(logits, losses, 0.1).backward()
        self.assertLess(float(logits.grad[0]), 0)
        self.assertGreater(float(logits.grad[1]), 0)
        with self.assertRaises(ValueError):
            bandit_loss(logits, losses, -1)

    def test_features_use_transformer_layers_and_probability_statistics(self):
        hidden = torch.arange(8.).reshape(1, 4, 2)
        state = features(torch.zeros(1, 2), hidden, (1, 2, 3))
        self.assertTrue(torch.equal(state[0, :6], torch.arange(2., 8.)))
        self.assertAlmostEqual(float(state[0, -1]), 0.5)
        self.assertAlmostEqual(float(state[0, -2]), float(torch.log(torch.tensor(2.))))

    def test_duplicate_problem_text_cannot_cross_splits(self):
        rows = [{"question": "A question"}, {"question": " a   QUESTION "}, {"question": "different"}]
        train, validation = problem_split(rows)
        self.assertEqual(bool(train[0]), bool(train[1]))
        self.assertTrue(bool((train ^ validation).all()))
        self.assertTrue(bool(train.any()) and bool(validation.any()))

    def test_budget_calibration_respects_ties_and_extreme_budgets(self):
        scores = torch.tensor([3., 2., 2., 1.])
        losses = torch.tensor([[2., 1.], [1., 2.], [1., 2.], [1., 2.]])
        threshold, chosen = calibrate_budget(scores, losses, 0.5)
        self.assertEqual(int(chosen.sum()), 1)
        self.assertTrue(torch.equal(chosen, scores > threshold))
        self.assertEqual(int(calibrate_budget(scores, losses, 0.)[1].sum()), 0)
        self.assertEqual(int(calibrate_budget(scores, losses, 1.)[1].sum()), 4)

    def test_logit_calibration_preserves_rank_after_sigmoid_saturation(self):
        scores = torch.tensor([24., 25., 26.])
        self.assertTrue(torch.equal(scores.sigmoid(), torch.ones(3)))
        threshold, chosen = calibrate_budget(scores, torch.ones(3, 2), 1 / 3)
        self.assertEqual(int(chosen.sum()), 1)
        self.assertTrue(bool(chosen[-1]))

    def test_benchmark_screen_matches_normalized_and_near_duplicate_questions(self):
        from .prepare_math import OverlapScreen
        screen = OverlapScreen(["A farmer buys twelve red apples and gives three red apples to his neighbor today."])
        self.assertTrue(screen.overlaps("A FARMER buys twelve red apples, and gives three red apples to his neighbor today!"))
        self.assertTrue(screen.overlaps("A farmer buys twelve red apples and gives three red apples to his neighbor today. Please solve."))
        self.assertFalse(screen.overlaps("Find the square root of sixteen."))


if __name__ == "__main__":
    unittest.main()
