import numpy as np
import pytest

from lsy_drone_racing.sim.noise import GaussianNoise, Noise, NoiseList, UniformNoise


@pytest.mark.unit
def test_base_noise():
    noise = Noise(dim=3)
    assert noise.dim == 3
    assert np.all(noise.mask == np.ones(3, dtype=bool))
    target = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(noise.apply(target), target)
    noise.step()
    assert noise._step == 1
    noise.reset()
    assert noise._step == 0


@pytest.mark.unit
def test_uniform_noise():
    noise = UniformNoise(dim=3, low=-1.0, high=1.0)
    target = np.zeros(3)

    for _ in range(100):
        noisy = noise.apply(target)
        assert np.all(noisy >= -1.0) and np.all(noisy <= 1.0)


@pytest.mark.unit
def test_gaussian_noise():
    noise = GaussianNoise(dim=3, std=1.0)
    target = np.zeros(3)

    samples = [noise.apply(target) for _ in range(10_000)]
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    assert np.allclose(mean, np.zeros(3), atol=0.1), f"Mean: {mean}"
    assert np.allclose(std, np.ones(3), atol=0.1)


@pytest.mark.unit
def test_noise_with_mask():
    mask = np.array([True, False, True])
    noise = UniformNoise(dim=3, low=-1.0, high=1.0, mask=mask)
    target = np.zeros(3)

    for _ in range(100):
        noisy = noise.apply(target)
        assert noisy[1] == 0
        assert -1.0 <= noisy[0] <= 1.0
        assert -1.0 <= noisy[2] <= 1.0


@pytest.mark.unit
def test_noise_list():
    noise1 = UniformNoise(dim=3, low=-1.0, high=1.0)
    noise2 = GaussianNoise(dim=3, std=0.5)
    noise_list = NoiseList([noise1, noise2])

    target = np.zeros(3)
    noisy = noise_list.apply(target)
    assert noisy.shape == (3,)

    noise_list.reset()
    assert noise1._step == 0
    assert noise2._step == 0


@pytest.mark.unit
def test_noise_list_from_specs():
    specs = [
        {"type": "UniformNoise", "dim": 3, "low": -1.0, "high": 1.0},
        {"type": "GaussianNoise", "dim": 3, "std": 0.5},
    ]
    noise_list = NoiseList.from_specs(specs)

    assert len(noise_list) == 2
    assert isinstance(noise_list[0], UniformNoise)
    assert isinstance(noise_list[1], GaussianNoise)


@pytest.mark.unit
def test_noise_seeding():
    noise = UniformNoise(dim=3, low=-1.0, high=1.0)
    noise.seed(42)

    target = np.zeros(3)
    result1 = noise.apply(target)
    noise.seed(42)
    result2 = noise.apply(target)

    assert np.array_equal(result1, result2)
