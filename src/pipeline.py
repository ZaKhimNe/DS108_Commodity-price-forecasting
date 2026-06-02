"""
pipeline.py — DS108Pipeline OOP Wrapper

Class DS108Pipeline bọc toàn bộ 22+ module thành một interface thống nhất.
Mỗi stage có method riêng, hỗ trợ chạy từng phần hoặc toàn bộ.

Usage:
    pipeline = DS108Pipeline(root_dir='.', mode='binary')
    pipeline.run_all(skip_ingestion=True)

CLI:
    python src/pipeline.py --mode binary --stage all --skip-ingestion
    python src/pipeline.py --stage checks
    python src/pipeline.py --help
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime


class DS108Pipeline:
    """
    End-to-end pipeline for DS108 Agricultural Commodity Forecasting.

    Architecture: 4-source ETL -> Feature Engineering -> Tensor Packing
                  -> 4 Base Models (LGBM, RF, LSTM, TCN)
                  -> Stacking Ensemble -> Backtesting -> Report

    Parameters
    ----------
    root_dir : str
        Project root directory (default: current directory).
    mode : str
        Prediction mode: 'binary' | 'multiclass' | 'regression'.
    log_level : str
        Logging level: 'DEBUG' | 'INFO' | 'WARNING'.
    """

    STAGES = [
        "sanity_checks",
        "ingestion",
        "preprocessing",
        "integration",
        "tensor_packing",
        "modeling",
        "evaluation",
    ]

    def __init__(
        self,
        root_dir: str = ".",
        mode: str = "binary",
        log_level: str = "INFO",
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.mode     = mode
        self.src_dir  = self.root_dir / "src"
        self.run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._completed_stages: list[str] = []
        self._setup_logging(log_level)
        self.logger = logging.getLogger("DS108Pipeline")
        self.logger.info(f"Pipeline initialized | mode={mode} | run_id={self.run_id}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _setup_logging(self, level: str) -> None:
        log_dir  = self.root_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"pipeline_{self.run_id}.log"

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )

    # ── Script runner ─────────────────────────────────────────────────────────

    def _run_script(self, script_path: str | Path, description: str) -> bool:
        """
        Run a Python script as subprocess from project root.
        Returns True on success (exit code 0), False otherwise.
        """
        script_path = str(script_path)
        if not os.path.exists(script_path):
            self.logger.warning(f"Script not found: {script_path} -- skipped")
            return True  # non-fatal: skip missing optional scripts

        self.logger.info(f"  >> {description}")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=str(self.root_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error(f"  FAILED: {description}")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-10:]:
                    self.logger.error(f"    {line}")
            return False

        self.logger.info(f"  OK: {description}")
        return True

    # ── Stages ────────────────────────────────────────────────────────────────

    def run_sanity_checks(self) -> bool:
        """Stage 0: Data integrity checks (8 automated assertions)."""
        self.logger.info("=" * 55)
        self.logger.info("STAGE 0: Sanity Checks")
        ok = self._run_script(self.src_dir / "00_sanity_checks.py",
                              "Data integrity (8 checks)")
        if not ok:
            self.logger.error("Sanity checks FAILED -- aborting pipeline")
        return ok

    def run_ingestion(self) -> bool:
        """Stage 1: Data ingestion from 4 sources (market, weather, macro, farming)."""
        self.logger.info("=" * 55)
        self.logger.info("STAGE 1: Data Ingestion")
        scripts = [
            ("ingestion/01_market_ingestion.py",  "Market data (KC=F, ZC=F)"),
            ("ingestion/02_weather_ingestion.py", "Weather (Open-Meteo, 5 regions)"),
            ("ingestion/03_macro_ingestion.py",   "Macro (CPI, VIX, USD/BRL)"),
            ("ingestion/04_farming_ingestion.py", "Farming calendar"),
        ]
        for rel, desc in scripts:
            if not self._run_script(self.src_dir / rel, desc):
                return False
        self._completed_stages.append("ingestion")
        return True

    def run_preprocessing(self) -> bool:
        """Stage 2: Causal preprocessing — MIQR despiking, ACU, feature engineering."""
        self.logger.info("=" * 55)
        self.logger.info("STAGE 2: Preprocessing")
        scripts = [
            ("preprocessing/05_weather_anomaly_removal.py", "Weather MIQR (center=False)"),
            ("preprocessing/06_weather_preprocessing.py",   "Weather ffill + rolling"),
            ("preprocessing/07_weather_weekly_aggregation.py", "Weather weekly (W-MON)"),
            ("preprocessing/08_market_acu_filter.py",        "Market ACU + RSI/BB/MACD"),
            ("preprocessing/09_macro_preprocessing.py",      "Macro CPI lag + VIX"),
            ("preprocessing/10_farming_preprocessing.py",    "Farming sin/cos encoding"),
        ]
        for rel, desc in scripts:
            if not self._run_script(self.src_dir / rel, desc):
                return False
        self._completed_stages.append("preprocessing")
        return True

    def run_integration(self) -> bool:
        """Stage 3–4: 4-source merge and CCF lag analysis."""
        self.logger.info("=" * 55)
        self.logger.info("STAGE 3-4: Integration + Lag Analysis")
        scripts = [
            ("11_data_integration.py", "4-source merge + feature engineering"),
            ("12_lag_analysis.py",     "CCF bootstrap lag validation"),
        ]
        for rel, desc in scripts:
            if not self._run_script(self.src_dir / rel, desc):
                return False
        self._completed_stages.append("integration")
        return True

    def run_tensor_packing(self) -> bool:
        """Stage 5: Pack 3-D tensors for LSTM/TCN (mode-specific)."""
        self.logger.info("=" * 55)
        self.logger.info(f"STAGE 5: Tensor Packing (mode={self.mode})")
        script_map = {
            "binary":     "13_tensor_packing.py",
            "multiclass": "13b_tensor_packing_mc.py",
            "regression": "13c_tensor_packing_reg.py",
        }
        script = script_map.get(self.mode, "13_tensor_packing.py")
        if not self._run_script(self.src_dir / script, f"Tensor packing ({self.mode})"):
            return False
        self._completed_stages.append("tensor_packing")
        return True

    def run_modeling(self) -> bool:
        """Stage 6: Train LGBM -> RF -> LSTM -> TCN -> Stacking."""
        self.logger.info("=" * 55)
        self.logger.info(f"STAGE 6: Modeling (mode={self.mode})")

        sfx = "b" if self.mode == "multiclass" else ("c" if self.mode == "regression" else "")
        model_names = {
            "":  ("lgbm_baseline",  "rf_baseline",  "lstm_hybrid",  "tcn_hybrid",  "stacking_ensemble"),
            "b": ("lgbm_multiclass","rf_multiclass","lstm_multiclass","tcn_multiclass","stacking_multiclass"),
            "c": ("lgbm_regression","rf_regression","lstm_regression","tcn_regression","stacking_regression"),
        }
        nums   = ("14", "15", "16", "17", "18")
        names  = model_names[sfx]
        labels = ("LightGBM", "Random Forest", "LSTM", "TCN", "Stacking Ensemble")

        for num, name, label in zip(nums, names, labels):
            script = f"modeling/{num}{sfx}_{name}.py"
            if not self._run_script(self.src_dir / script, f"{label} ({self.mode})"):
                return False
        self._completed_stages.append("modeling")
        return True

    def run_evaluation(self) -> bool:
        """Stage 7–9: Backtesting, walkforward, hurdle, bootstrap, cost analysis, report."""
        self.logger.info("=" * 55)
        self.logger.info("STAGE 7-9: Evaluation")

        sfx = "b" if self.mode == "multiclass" else ("c" if self.mode == "regression" else "")
        bt_name = {
            "":  "backtesting_engine",
            "b": "backtesting_mc",
            "c": "backtesting_reg",
        }[sfx]

        scripts = [(f"modeling/19{sfx}_{bt_name}.py", "Backtesting (Sharpe/MDD/WinRate)")]

        if self.mode in ("binary", "multiclass"):
            scripts += [
                ("modeling/21_walkforward_eval.py",  "Walkforward evaluation (per-year gate)"),
                ("modeling/22_hurdle_model.py",      "Hurdle model (two-stage)"),
                ("modeling/23_bootstrap_sharpe.py",  "Bootstrap CI for Sharpe"),
                ("modeling/24_cost_sensitivity.py",  "Transaction cost sensitivity"),
            ]

        scripts.append(("20_pipeline_report.py", "Pipeline report"))

        for rel, desc in scripts:
            # Evaluation steps: log warning on failure, do not abort pipeline
            if not self._run_script(self.src_dir / rel, desc):
                self.logger.warning(f"Evaluation step non-fatal failure: {desc}")

        self._completed_stages.append("evaluation")
        return True

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def run_all(
        self,
        skip_ingestion: bool = False,
        run_checks: bool = True,
    ) -> bool:
        """
        Run the full pipeline end-to-end.

        Parameters
        ----------
        skip_ingestion : bool
            Skip Stage 1 (use cached raw data). Useful when re-running preprocessing.
        run_checks : bool
            Run sanity checks before pipeline starts.
        """
        self.logger.info("=" * 55)
        self.logger.info(f"DS108 PIPELINE START | mode={self.mode} | run_id={self.run_id}")
        self.logger.info("=" * 55)

        stages = []
        if run_checks:
            stages.append(self.run_sanity_checks)
        if not skip_ingestion:
            stages.append(self.run_ingestion)
        stages += [
            self.run_preprocessing,
            self.run_integration,
            self.run_tensor_packing,
            self.run_modeling,
            self.run_evaluation,
        ]

        for stage_fn in stages:
            if not stage_fn():
                self.logger.error(f"Pipeline ABORTED at: {stage_fn.__name__}")
                return False

        self.logger.info("=" * 55)
        self.logger.info(f"PIPELINE COMPLETE | stages={self._completed_stages}")
        self.logger.info("=" * 55)
        return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS108 Pipeline Runner — Agricultural Commodity Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --stage checks
  python src/pipeline.py --mode binary --stage all --skip-ingestion
  python src/pipeline.py --mode multiclass --stage modeling
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "multiclass", "regression"],
        default="binary",
        help="Prediction mode (default: binary)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "checks", "ingestion", "preprocessing",
                 "integration", "tensor", "modeling", "evaluation"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion (use cached raw data)",
    )
    parser.add_argument(
        "--no-checks",
        action="store_true",
        help="Skip sanity checks before run_all",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    args = parser.parse_args()

    pipeline = DS108Pipeline(root_dir=args.root, mode=args.mode)

    dispatch = {
        "all":          lambda: pipeline.run_all(
                            skip_ingestion=args.skip_ingestion,
                            run_checks=not args.no_checks),
        "checks":       pipeline.run_sanity_checks,
        "ingestion":    pipeline.run_ingestion,
        "preprocessing":pipeline.run_preprocessing,
        "integration":  pipeline.run_integration,
        "tensor":       pipeline.run_tensor_packing,
        "modeling":     pipeline.run_modeling,
        "evaluation":   pipeline.run_evaluation,
    }

    success = dispatch[args.stage]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
