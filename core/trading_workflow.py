#!/usr/bin/env python3
"""Core MVP trading workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from config.env_loader import load_env_variables
from core.pipeline import run_enhanced_analysis
from core.portfolio_manager import PortfolioManager


class AnalysisRunner(Protocol):
    def __call__(
        self,
        symbol: str,
        api_keys: Dict[str, Optional[str]],
        portfolio_name: str,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


class PaperOrderExecutor(Protocol):
    def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...


@dataclass(frozen=True)
class WorkflowResult:
    symbol: str
    portfolio_name: str
    analysis: Dict[str, Any]
    recorded_trade_id: Optional[int] = None
    alpaca_order: Optional[Dict[str, Any]] = None
    skipped_reason: Optional[str] = None


class TradingWorkflow:
    """Coordinates the core analyze -> optional paper-record workflow."""

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        analysis_runner: AnalysisRunner = run_enhanced_analysis,
        api_key_loader: Callable[[], Dict[str, Optional[str]]] = load_env_variables,
        paper_order_executor_factory: Optional[Callable[[], PaperOrderExecutor]] = None,
    ) -> None:
        self.portfolio_manager = portfolio_manager
        self.analysis_runner = analysis_runner
        self.api_key_loader = api_key_loader
        self.paper_order_executor_factory = paper_order_executor_factory

    def run(
        self,
        symbol: str,
        portfolio_name: str = "default",
        *,
        record_paper_trade: bool = False,
        submit_alpaca_paper_order: bool = False,
        user_id: Optional[int] = None,
    ) -> WorkflowResult:
        symbol = symbol.upper().strip()
        portfolio = self.portfolio_manager.get_portfolio(portfolio_name, user_id=user_id)
        if not portfolio and (record_paper_trade or submit_alpaca_paper_order):
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        if user_id is None:
            analysis = self.analysis_runner(symbol, self.api_key_loader(), portfolio_name)
        else:
            analysis = self.analysis_runner(symbol, self.api_key_loader(), portfolio_name, user_id)
        if analysis.get("error"):
            return WorkflowResult(
                symbol=symbol,
                portfolio_name=portfolio_name,
                analysis=analysis,
                skipped_reason=analysis.get("message", "Analysis failed"),
            )

        if not record_paper_trade and not submit_alpaca_paper_order:
            return WorkflowResult(
                symbol=symbol,
                portfolio_name=portfolio_name,
                analysis=analysis,
                skipped_reason="Paper trading disabled",
            )

        action = self._extract_action(analysis)
        quantity = int(analysis.get("quantity") or 0)
        if action not in {"BUY", "SELL"} or quantity <= 0:
            return WorkflowResult(
                symbol=symbol,
                portfolio_name=portfolio_name,
                analysis=analysis,
                skipped_reason="No actionable trade recommendation",
            )

        price = self._extract_price(analysis, quantity)
        if price <= 0:
            return WorkflowResult(
                symbol=symbol,
                portfolio_name=portfolio_name,
                analysis=analysis,
                skipped_reason="Analysis did not include a usable execution price",
            )

        alpaca_order = None
        if submit_alpaca_paper_order:
            executor = self._get_paper_order_executor()
            alpaca_order = executor.execute_signal(
                {
                    **analysis,
                    "symbol": symbol,
                    "recommendation": action,
                    "quantity": quantity,
                }
            )
            if not alpaca_order:
                return WorkflowResult(
                    symbol=symbol,
                    portfolio_name=portfolio_name,
                    analysis=analysis,
                    skipped_reason="Alpaca paper order was not submitted",
                )

        notes = "Recorded by TradingWorkflow local paper mode"
        if alpaca_order:
            notes = (
                "Submitted to Alpaca paper trading "
                f"order_id={alpaca_order.get('id')} "
                f"status={alpaca_order.get('status', 'unknown')}"
            )

        trade_id = self.portfolio_manager.record_trade(
            name=portfolio_name,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            strategy=str(analysis.get("primary_strategy") or "analysis"),
            confidence=self._extract_confidence(analysis),
            notes=notes,
            user_id=user_id,
        )

        return WorkflowResult(
            symbol=symbol,
            portfolio_name=portfolio_name,
            analysis=analysis,
            recorded_trade_id=trade_id,
            alpaca_order=alpaca_order,
        )

    def _get_paper_order_executor(self) -> PaperOrderExecutor:
        if self.paper_order_executor_factory:
            return self.paper_order_executor_factory()

        from core.alpaca_executor_provider import get_alpaca_executor

        return get_alpaca_executor()

    def _extract_action(self, analysis: Dict[str, Any]) -> str:
        return str(analysis.get("recommendation") or analysis.get("signal") or "HOLD").upper()

    def _extract_price(self, analysis: Dict[str, Any], quantity: int) -> float:
        risk = analysis.get("risk_parameters") or {}
        position_value = float(risk.get("position_value") or 0)
        if position_value > 0 and quantity > 0:
            return round(position_value / quantity, 4)

        all_signals = analysis.get("all_signals") or {}
        technical = all_signals.get("technical") or {}
        current_price = float(technical.get("current_price") or 0)
        if current_price > 0:
            return current_price

        return float(analysis.get("current_price") or 0)

    def _extract_confidence(self, analysis: Dict[str, Any]) -> Optional[float]:
        confidence = analysis.get("confidence")
        if confidence is None:
            return None
        if isinstance(confidence, str):
            stripped = confidence.strip().rstrip("%")
            value = float(stripped)
            return value / 100 if value > 1 else value
        return float(confidence)
