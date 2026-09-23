"""Bull/bear/sideways regime labels from monthly benchmark returns (2.2)."""


def classify_regime(monthly_return, bull_thresh=0.02, bear_thresh=-0.02):
    if monthly_return > bull_thresh:
        return 'Bull'
    elif monthly_return < bear_thresh:
        return 'Bear'
    else:
        return 'Sideways'
