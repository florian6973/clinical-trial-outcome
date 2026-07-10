from clinical_trial_outcome.classification.cli import build_parser


def test_classification_cli_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(["overall survival"])
    assert args.execute is False


def test_classification_cli_requires_explicit_execute_flag() -> None:
    args = build_parser().parse_args(["overall survival", "--execute"])
    assert args.execute is True
