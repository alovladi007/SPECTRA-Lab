def winrate(n_wins:int, n_total:int)->float:
    return n_wins / max(n_total,1)

def harmful_rate(n_harm:int, n_total:int)->float:
    return n_harm / max(n_total,1)


# Stub functions for API compatibility  
def calculate_metrics(predictions, targets, **kwargs):
    """Placeholder metrics calculator"""
    return {"status": "not_implemented", "message": "calculate_metrics not yet implemented"}

def winrate(a, b):
    """Placeholder winrate"""
    return 0.5

def harmful_rate(responses):
    """Placeholder harmful rate"""
    return 0.0
