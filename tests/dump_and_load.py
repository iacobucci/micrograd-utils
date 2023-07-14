import unittest
import sys

sys.path.append("../src/micrograd_utils")
from micrograd_utils.utils import *


class TestDumpAndLoad(unittest.TestCase):
    def test_dump(self):
        d = {
            (0,): 0,
            (1,): 1,
        }

        dataset = [list(k) for k, v in d.items()]
        desired = [v for k, v in d.items()]

        m = MLP(1, [16, 16, 1])

        for k in range(100):
            # forward pass with mean squared error loss
            ypred = [m(x) for x in dataset]
            loss = sum([(yout - ygt)**2 for ygt, yout in zip(desired, ypred)])
            # zero gradients
            m.zero_grad()
            # backward pass
            loss.backward()
            # update weights
            for p in m.parameters():
                p.data -= p.grad * 0.05

            print(k, loss)

        dump_parameters(m)

    def test_load(self):
        m = load_parameters()

        new_data = [
            [0],
            [1],
        ]

        for x in new_data:
            result = m(x)
            print("{", x[0], ":", result.data, "}")
            self.assertAlmostEqual(result.data, x[0], delta=0.01)

