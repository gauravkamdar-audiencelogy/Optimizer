"""
RTB Optimizer - Prediction Models

Models:
- WinRateModel: Predicts P(win | features) - diagnostic only
- CTRModel: Predicts P(click | impression)
- EmpiricalWinRateModel: Empirical segment win rates with shrinkage
- BidLandscapeModel: V4 - Predicts P(win | bid_amount, features)
"""
from .win_rate_model import WinRateModel
from .ctr_model import CTRModel
from .empirical_win_rate_model import EmpiricalWinRateModel
from .bid_landscape_model import BidLandscapeModel

__all__ = ['WinRateModel', 'CTRModel', 'EmpiricalWinRateModel', 'BidLandscapeModel']
