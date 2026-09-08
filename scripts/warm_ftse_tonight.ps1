Set-Location "C:\Users\Craig\.claude\skills\Strategy_Auto_Trader"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$log = "logs\warm_ftse_tonight_$(Get-Date -Format yyyyMMdd_HHmmss).log"
uv run python -m Strategy_Auto_Trader.synthetic_backtest_data.warm_hmm_cache `
  --tickers AAF.L AAL.L ABDN.L ABF.L ADM.L ALW.L ANTO.L AUTO.L AV.L AZN.L BA.L BAB.L BARC.L BATS.L BBOX.L BEZ.L BGEO.L BLND.L BNZL.L BP.L BRBY.L BT-A.L BTRW.L CCC.L CCEP.L CCH.L CNA.L CRDA.L CTEC.L DCC.L DGE.L DPLM.L EDV.L ENT.L EXPN.L FCIT.L FRES.L GAW.L GLEN.L GSK.L HLMA.L HLN.L HSBA.L HSX.L HWDN.L IAG.L ICG.L IGG.L III.L IMB.L IMI.L INF.L INVP.L ITRK.L JD.L KGF.L LAND.L LGEN.L LLOY.L LMP.L LSEG.L MKS.L MNG.L MRO.L NG.L NWG.L NXT.L PCT.L PRU.L PSH.L PSN.L PSON.L REL.L RIO.L RKT.L RR.L RTO.L SBRY.L SDLF.L SDR.L SGE.L SGRO.L SHEL.L SMIN.L SMT.L SN.L SPX.L SSE.L STAN.L STJ.L SVT.L TSCO.L ULVR.L UU.L VOD.L WEIR.L WTB.L `
  --start-date 1999-09-01 `
  --end-date 2026-09-01 `
  --workers 2 *>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
