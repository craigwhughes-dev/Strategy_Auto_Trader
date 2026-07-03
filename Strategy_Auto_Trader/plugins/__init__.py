"""Strategy plugin architecture for the consolidated engine.

Protocols (protocols.py) define the structural interfaces for each of the
six plugin slots.  Default implementations in the sibling modules satisfy
these protocols and produce bit-identical results to the original inline
consolidated_engine logic.

Plugin slots:
  RegimeModelProtocol    — HMM regime model (stateful)
  SignalGeneratorProtocol — composite vote signal (stateless)
  QualityGateProtocol    — conservative veto layer (stateless)
  ExitRulesProtocol      — exit condition set (stateless per-bar)
  PositionSizerProtocol  — Kelly / fixed position sizing
  ContextAdjusterProtocol — sentiment / VIX threshold nudges
  PrescreenProtocol      — per-ticker volatility prescreen
"""
