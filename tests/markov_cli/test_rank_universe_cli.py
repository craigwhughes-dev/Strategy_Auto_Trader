from __future__ import annotations

import json
from unittest import mock

from Strategy_Auto_Trader.markov_cli.rank_universe_cli import main


def test_writes_scores_json_to_output_path(tmp_path):
    output_path = tmp_path / "scores.json"

    with mock.patch(
        "Strategy_Auto_Trader.markov_cli.rank_universe_cli.full_scan.load_sp_ftse_universe",
        return_value=["AAPL", "MSFT"],
    ):
        with mock.patch(
            "Strategy_Auto_Trader.markov_cli.rank_universe_cli.rank_universe",
            return_value={"AAPL": 0.8, "MSFT": 0.6},
        ) as mock_rank:
            rc = main(["--output", str(output_path)])

    assert rc == 0
    assert json.loads(output_path.read_text()) == {"AAPL": 0.8, "MSFT": 0.6}
    mock_rank.assert_called_once()


def test_args_passed_through_to_rank_universe(tmp_path):
    output_path = tmp_path / "scores.json"

    with mock.patch(
        "Strategy_Auto_Trader.markov_cli.rank_universe_cli.full_scan.load_sp_ftse_universe",
        return_value=["AAPL"],
    ):
        with mock.patch(
            "Strategy_Auto_Trader.markov_cli.rank_universe_cli.rank_universe",
            return_value={},
        ) as mock_rank:
            main([
                "--strategy", "conservative",
                "--vol-weight", "0.5", "--win-rate-weight", "0.5",
                "--lookback-days", "30", "--workers", "2",
                "--output", str(output_path),
            ])

    mock_rank.assert_called_once_with(
        ["AAPL"], "conservative",
        vol_weight=0.5, win_rate_weight=0.5, lookback_days=30, workers=2,
        use_seasonal_volume=True,
    )


def test_creates_output_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "scores.json"

    with mock.patch(
        "Strategy_Auto_Trader.markov_cli.rank_universe_cli.full_scan.load_sp_ftse_universe",
        return_value=[],
    ):
        with mock.patch(
            "Strategy_Auto_Trader.markov_cli.rank_universe_cli.rank_universe",
            return_value={},
        ):
            main(["--output", str(output_path)])

    assert output_path.exists()
