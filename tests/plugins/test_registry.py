from __future__ import annotations

import numpy as np
import pytest


class TestRegistry:

    # -- resolve() basics -------------------------------------------------------

    def test_resolve_kelly_sizer(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.kelly_sizer import KellySizer
        sizer = resolve("sizer", "kelly")
        assert isinstance(sizer, KellySizer)

    def test_resolve_fixed_sizer(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.kelly_sizer import FixedSizer
        sizer = resolve("sizer", "fixed")
        assert isinstance(sizer, FixedSizer)

    def test_resolve_quality_gate(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.quality_gate import QualityGatePlugin
        gate = resolve("gate", "quality")
        assert isinstance(gate, QualityGatePlugin)

    def test_resolve_null_gate(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.quality_gate import NullQualityGate
        gate = resolve("gate", "none")
        assert isinstance(gate, NullQualityGate)

    def test_resolve_sentiment_adjuster(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.context_adjuster import SentimentAdjuster
        adj = resolve("adjuster", "sentiment")
        assert isinstance(adj, SentimentAdjuster)

    def test_resolve_null_adjuster(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        from Strategy_Auto_Trader.plugins.context_adjuster import NullAdjuster
        adj = resolve("adjuster", "none")
        assert isinstance(adj, NullAdjuster)

    def test_resolve_unknown_slot_raises(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        with pytest.raises(KeyError, match="Unknown plugin slot"):
            resolve("banana", "kelly")

    def test_resolve_unknown_name_raises(self):
        from Strategy_Auto_Trader.plugins.registry import resolve
        with pytest.raises(KeyError, match="Unknown plugin"):
            resolve("sizer", "neural_net")

    # -- FixedSizer behaviour ---------------------------------------------------

    def test_fixed_sizer_returns_constant(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import FixedSizer
        sizer = FixedSizer(fraction=0.15)
        assert sizer.size([0.05, 0.03, -0.02]) == pytest.approx(0.15)
        assert sizer.position == pytest.approx(0.15)
        assert sizer.current_kelly == pytest.approx(0.15)

    def test_fixed_sizer_default_fraction(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import FixedSizer
        sizer = FixedSizer()
        assert sizer.position == pytest.approx(0.10)

    def test_fixed_sizer_records_trades(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import FixedSizer
        sizer = FixedSizer()
        sizer.record(0.05)
        sizer.record(-0.02)
        assert len(sizer.trade_results) == 2
        assert sizer.trade_results[0] == pytest.approx(0.05)

    def test_fixed_sizer_position_unaffected_by_record(self):
        from Strategy_Auto_Trader.plugins.kelly_sizer import FixedSizer
        sizer = FixedSizer(fraction=0.12)
        for pl in [0.1, 0.2, 0.3, -0.05, 0.15, 0.1, 0.2, 0.1, 0.05, 0.1,
                   0.1, 0.2, 0.3, -0.05, 0.15, 0.1, 0.2, 0.1, 0.05, 0.1]:
            sizer.record(pl)
        assert sizer.position == pytest.approx(0.12)

    # -- batch._build_argv plugin dispatch --------------------------------------

    def test_build_argv_plugin_sizer_fixed(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "plugins": {"sizer": "fixed"}}
        argv = _build_argv(cfg, {})
        assert "--plugin-sizer" in argv
        assert argv[argv.index("--plugin-sizer") + 1] == "fixed"

    def test_build_argv_plugin_gate_none(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "plugins": {"gate": "none"}}
        argv = _build_argv(cfg, {})
        assert "--plugin-gate" in argv
        assert argv[argv.index("--plugin-gate") + 1] == "none"

    def test_build_argv_plugin_adjuster_none(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL", "plugins": {"adjuster": "none"}}
        argv = _build_argv(cfg, {})
        assert "--plugin-adjuster" in argv
        assert argv[argv.index("--plugin-adjuster") + 1] == "none"

    def test_build_argv_no_plugins_key(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        cfg = {"ticker": "AAPL"}
        argv = _build_argv(cfg, {})
        assert "--plugin-sizer" not in argv
        assert "--plugin-gate" not in argv
        assert "--plugin-adjuster" not in argv

    def test_build_argv_plugins_merged_with_defaults(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        defaults = {"plugins": {"sizer": "kelly"}}
        cfg = {"ticker": "AAPL", "plugins": {"sizer": "fixed"}}
        argv = _build_argv(cfg, defaults)
        assert argv[argv.index("--plugin-sizer") + 1] == "fixed"

    def test_build_argv_plugins_from_defaults_only(self):
        from Strategy_Auto_Trader.markov_cli.batch import _build_argv
        defaults = {"plugins": {"gate": "none"}}
        cfg = {"ticker": "MSFT"}
        argv = _build_argv(cfg, defaults)
        assert "--plugin-gate" in argv
        assert argv[argv.index("--plugin-gate") + 1] == "none"

    # -- run._build_arg_parser accepts --plugin-* args --------------------------

    def test_arg_parser_plugin_sizer_default(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY"])
        assert args.plugin_sizer == "kelly"

    def test_arg_parser_plugin_sizer_fixed(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--plugin-sizer", "fixed"])
        assert args.plugin_sizer == "fixed"

    def test_arg_parser_plugin_gate_none(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--plugin-gate", "none"])
        assert args.plugin_gate == "none"

    def test_arg_parser_plugin_adjuster_none(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        args = _build_arg_parser().parse_args(["--ticker", "SPY", "--plugin-adjuster", "none"])
        assert args.plugin_adjuster == "none"

    def test_arg_parser_invalid_plugin_sizer_rejected(self):
        from Strategy_Auto_Trader.markov_cli.run import _build_arg_parser
        with pytest.raises(SystemExit):
            _build_arg_parser().parse_args(["--ticker", "SPY", "--plugin-sizer", "neural_net"])
