"""Internal details required for exact trial-level reproduction."""


def evaluation_batches(environment: str) -> tuple[tuple[int, int], ...]:
    if environment == "cartpole":
        return ((100, 20260801),)
    return ((30, 20260801), (70, 20260831))
