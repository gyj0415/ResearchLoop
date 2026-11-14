import unittest
from researchloop.ollama import Budget, BudgetError


class BudgetTests(unittest.TestCase):
    def test_preflight_prevents_call(self):
        budget = Budget(10, 30)
        with self.assertRaises(BudgetError):
            budget.reserve(11)

    def test_all_purposes_and_unknown_usage_consume_budget(self):
        budget = Budget(100, 30)
        budget.record({"purpose": "planner", "input_tokens": 10, "output_tokens": 5})
        budget.record({"purpose": "retry", "input_tokens": 10, "output_tokens": 5})
        budget.record({"purpose": "reflection", "unknown_reserved": 20})
        self.assertEqual(budget.used, 50)
        self.assertEqual(budget.snapshot()["model_calls"], 3)
