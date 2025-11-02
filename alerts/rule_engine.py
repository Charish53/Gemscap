"""Rule engine for evaluating alert conditions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
import operator

from utils.logger import setup_logger
from utils.exceptions import AlertError


@dataclass
class Alert:
    """Alert record."""
    rule_id: str
    rule_name: str
    timestamp: int
    message: str
    severity: str = "info"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class RuleEngine:
    """Engine for evaluating user-defined alert rules."""
    
    OPERATORS = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    
    def __init__(self):
        """Initialize rule engine."""
        self.logger = setup_logger(f"{self.__class__.__name__}")
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Alert] = []
    
    def register_rule(
        self,
        rule_id: str,
        rule_name: str,
        metric: str,
        operator: str,
        threshold: float,
        enabled: bool = True
    ) -> None:
        """
        Register a new alert rule.
        
        Args:
            rule_id: Unique rule identifier
            rule_name: Human-readable rule name
            metric: Metric name (e.g., 'zscore', 'spread', 'correlation')
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            enabled: Whether rule is enabled
        """
        if operator not in self.OPERATORS:
            raise AlertError(f"Unsupported operator: {operator}")
        
        self.rules[rule_id] = {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "enabled": enabled,
        }
        
        self.logger.info(f"Registered rule: {rule_id} - {rule_name}")
    
    def unregister_rule(self, rule_id: str) -> None:
        """
        Remove a rule.
        
        Args:
            rule_id: Rule identifier
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Unregistered rule: {rule_id}")
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id]["enabled"] = True
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id]["enabled"] = False
    
    def evaluate(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[int] = None
    ) -> List[Alert]:
        """
        Evaluate all enabled rules against current metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            timestamp: Optional timestamp (default: current time)
            
        Returns:
            List of triggered alerts
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp() * 1000)
        
        triggered_alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule["enabled"]:
                continue
            
            metric_name = rule["metric"]
            if metric_name not in metrics:
                continue
            
            metric_value = metrics[metric_name]
            operator_func = self.OPERATORS[rule["operator"]]
            threshold = rule["threshold"]
            
            if operator_func(metric_value, threshold):
                alert = Alert(
                    rule_id=rule_id,
                    rule_name=rule["rule_name"],
                    timestamp=timestamp,
                    message=f"{rule['rule_name']}: {metric_name} = {metric_value} {rule['operator']} {threshold}",
                    severity="warning",
                    metadata={
                        "metric": metric_name,
                        "value": metric_value,
                        "operator": rule["operator"],
                        "threshold": threshold,
                    }
                )
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                
                self.logger.warning(f"Alert triggered: {alert.message}")
        
        return triggered_alerts
    
    def get_alerts(
        self,
        rule_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            rule_id: Optional rule filter
            limit: Optional result limit
            
        Returns:
            List of alerts
        """
        alerts = self.alert_history
        
        if rule_id:
            alerts = [a for a in alerts if a.rule_id == rule_id]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history = []
        self.logger.info("Alert history cleared")
    
    def get_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered rules."""
        return self.rules.copy()

