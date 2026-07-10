from clinical_trial_outcome.condition_normalization.cli import build_parser


def test_condition_cli_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(["asthma"])
    assert args.execute is False


def test_condition_cli_requires_explicit_execute_flag() -> None:
    args = build_parser().parse_args(["asthma", "--execute"])
    assert args.execute is True
