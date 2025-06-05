import numpy as np

HISTORY_CHOICES = ["all", "last", "first", "alternate", "third"]


def get_history_inds(history_choice: str, n_available_inds: int) -> list[int]:
    assert (
        history_choice in HISTORY_CHOICES
    ), f"Invalid history choice: {history_choice} not in {HISTORY_CHOICES}"

    if history_choice == "all":
        inds = np.arange(n_available_inds)
    elif history_choice == "last":
        inds = np.array([max(0, n_available_inds - 1)])
    elif history_choice == "first":
        inds = np.array([0])
    elif history_choice == "alternate":
        inds = np.arange(0, n_available_inds, 2)
    elif history_choice == "third":
        inds = np.arange(0, n_available_inds, 3)
    elif history_choice.lower() == "none":
        inds = []
    else:
        raise ValueError(f"Invalid history_choice: {history_choice}")

    return list(inds)
