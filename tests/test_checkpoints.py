"""Checkpoint discovery."""

from gatle_ignite.utils.checkpoints import find_best_checkpoint, find_latest_checkpoint, score_of


def _touch(directory, name):
    path = directory / name
    path.write_bytes(b"x")
    return path


def test_score_is_read_from_the_filename(tmp_path):
    assert score_of("valid_best_result_checkpoint_15_0.9258.pt") == 0.9258
    assert score_of("valid_best_result_checkpoint_3_-0.25.pt") == -0.25
    assert score_of("latest_epoch_checkpoint_3.pt") == 3.0  # trailing int is the step
    assert score_of("no_numbers_here.pt") is None


def test_best_is_the_highest_score_not_the_newest_file(tmp_path):
    _touch(tmp_path, "valid_best_result_checkpoint_1_0.9000.pt")
    _touch(tmp_path, "valid_best_result_checkpoint_2_0.5000.pt")
    newest = _touch(tmp_path, "valid_best_result_checkpoint_3_0.6000.pt")

    best = find_best_checkpoint(tmp_path, prefix="valid_")
    assert best.name == "valid_best_result_checkpoint_1_0.9000.pt"
    assert best != newest


def test_negative_scores_are_handled(tmp_path):
    """score_factor=-1 (a loss or CER) makes every score negative."""
    # Scores below -100, so a finite 'no score yet' sentinel would win.
    _touch(tmp_path, "valid_best_result_checkpoint_1_-250.5.pt")
    _touch(tmp_path, "valid_best_result_checkpoint_2_-300.0.pt")
    best = find_best_checkpoint(tmp_path, prefix="valid_")
    assert best.name == "valid_best_result_checkpoint_1_-250.5.pt"


def test_missing_dir_and_empty_dir_return_none(tmp_path):
    assert find_best_checkpoint(tmp_path / "nope") is None
    assert find_latest_checkpoint(tmp_path) is None


def test_prefixes_do_not_collide(tmp_path):
    _touch(tmp_path, "valid_best_result_checkpoint_1_0.9.pt")
    _touch(tmp_path, "test_best_result_checkpoint_1_0.1.pt")
    assert "valid" in find_best_checkpoint(tmp_path, prefix="valid_").name
    assert "test" in find_best_checkpoint(tmp_path, prefix="test_").name
