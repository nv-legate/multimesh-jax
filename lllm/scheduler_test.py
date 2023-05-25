import unittest

from lllm.scheduler import Scheduler


class SchedulerTest(unittest.TestCase):
    def ideal_time(
        self, num_layers, num_microbatch, num_gpus, forward_cost, backward_cost
    ):
        return (
            num_layers
            * num_microbatch
            * (forward_cost + backward_cost)
            / num_gpus
        )

    def bubble_fraction(self, num_layers, num_microbatch, num_gpus):
        interleave = num_layers // num_gpus
        return (num_layers - 1) / num_microbatch / interleave

    def pipeline_time(
        self, num_layers, num_microbatch, num_gpus, forward_cost, backward_cost
    ):
        ideal_time = self.ideal_time(
            num_layers, num_microbatch, num_gpus, forward_cost, backward_cost
        )
        bubble_fraction = self.bubble_fraction(
            num_layers, num_microbatch, num_gpus
        )
        return ideal_time * (1 + bubble_fraction)

    def test_ideal_matches_no_interleave(self):
        num_microbatch = 4
        num_gpus = 4
        forward_cost = 1
        backward_cost = 2
        scheduler = Scheduler(
            num_microbatches=num_microbatch,
            num_layers=num_gpus,
            num_gpus=num_gpus,
            forward_cost=forward_cost,
            backward_cost=backward_cost,
            max_breadth=100,
        )

        computed_time, order = scheduler.compute()
        correct_time = self.pipeline_time(
            num_gpus, num_microbatch, num_gpus, forward_cost, backward_cost
        )
        self.assertEqual(computed_time, correct_time)

        # Each GPU should have fwd/bkwd microbatches
        self.assertEqual(len(order), num_microbatch * 2 * num_gpus)

    def test_max_breadth(self):
        num_microbatch = 32
        num_gpus = 4
        num_layers = 8
        num_interleave = num_layers // num_gpus
        forward_cost = 1
        backward_cost = 3
        max_breadth = 12
        scheduler = Scheduler(
            num_microbatches=num_microbatch,
            num_layers=num_layers,
            num_gpus=num_gpus,
            forward_cost=forward_cost,
            backward_cost=backward_cost,
            max_breadth=max_breadth,
        )

        computed_time, total_order = scheduler.compute()
        min_time = self.pipeline_time(
            num_gpus, num_microbatch, num_gpus, forward_cost, backward_cost
        )
        # this should not have enough parallelism to reach the min time
        self.assertGreater(computed_time, min_time)

        self.assertEqual(
            len(total_order), num_gpus * num_microbatch * 2 * num_interleave
        )


if __name__ == "__main__":
    unittest.main()
