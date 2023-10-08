from micrograd.nn import *
from micrograd.engine import *
import pytest
import sys
import os

# Ottieni il percorso completo alla directory src/micrograd_utils
src_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../src/micrograd_utils'))

# Aggiungi src al percorso di ricerca dei moduli
sys.path.insert(0, src_path)


import utils


def test_dump():
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

    utils.dump(m)


def test_load():
    src_path = os.path.abspath(os.path.join( os.path.dirname(__file__), '.'))

    sys.path.insert(0, src_path)

    import params

    m = params.model()

    new_data = [
        [0],
        [1],
    ]

    for x in new_data:
        result = m(x)
        print("{", x[0], ":", result.data, "}")

        assert abs(result.data - x[0]) < 0.01


if __name__ == "__main__":
    pytest.main()
