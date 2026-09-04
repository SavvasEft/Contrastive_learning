import cle


def test_package_imports():
    assert cle.__version__


def test_env_describe():
    env = cle.describe_env()
    assert env["torch"]
    assert isinstance(env["cuda_available"], bool)


def test_torch_forward():
    import torch

    x = torch.randn(4, 3)
    layer = torch.nn.Linear(3, 2)
    y = layer(x)
    y.sum().backward()
    assert y.shape == (4, 2)
    assert layer.weight.grad is not None