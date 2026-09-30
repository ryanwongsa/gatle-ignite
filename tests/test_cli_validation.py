"""CLI misuse must fail loudly rather than quietly do something else."""


def test_eval_refuses_a_comma_list_rather_than_scoring_only_the_first(capsys):
    """`train` fans a comma list out to subprocesses; `eval` has no such path."""
    from gatle_ignite.cli.main import main

    # main() prints a ConfigError and exits 2 rather than raising.
    assert main(["eval", "--config=a.py,b.py"]) == 2
    assert capsys.readouterr().err.count("eval takes one --config") == 1
