"""
RTB Optimizer - Prediction Models

Models:
- WinRateModel: Predicts P(win | features)
- CTRModel: Predicts P(click | impression)
"""
from .win_rate_model import WinRateModel
from .ctr_model import CTRModel

__all__ = ['WinRateModel', 'CTRModel']
