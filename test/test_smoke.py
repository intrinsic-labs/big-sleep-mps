"""End-to-end smoke test: two optimisation steps at 128 px on whatever device
is available (CPU in CI), writing a PNG. Downloads ~565 MB of weights on first
run (BigGAN-deep-128 + CLIP ViT-B/32); CI caches them."""
import sys

from PIL import Image


def test_two_steps_write_an_image(tmp_path):
    sys.argv = sys.argv[:1]  # the library's SIGINT/fire plumbing reads argv
    from big_sleep import Imagine

    dream = Imagine(
        text="a pyramid made of ice",
        image_size=128,
        num_cutouts=2,
        epochs=1,
        iterations=2,
        save_every=1,
        open_folder=False,
        seed=1,
        output_dir=str(tmp_path),
    )
    dream()

    assert dream.filename.exists(), f"no image written at {dream.filename}"
    with Image.open(dream.filename) as im:
        assert im.size == (128, 128)


def test_seed_is_reproducible(tmp_path):
    """--seed must reproduce the loss: torch AND Python `random` (used for
    cutout offsets) are both seeded."""
    sys.argv = sys.argv[:1]
    from big_sleep import Imagine

    def first_loss():
        dream = Imagine(text="a pyramid made of ice", image_size=128, num_cutouts=2,
                        epochs=1, iterations=1, save_every=10**9, open_folder=False,
                        seed=7, output_dir=str(tmp_path))
        _, loss = dream.train_step(0, 0)
        return loss.item()

    assert first_loss() == first_loss()
