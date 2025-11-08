def winrate(n_wins:int, n_total:int)->float:
    return n_wins / max(n_total,1)

def harmful_rate(n_harm:int, n_total:int)->float:
    return n_harm / max(n_total,1)
