"""Deterministic schedule/stopping, legacy experiment.py:21,127-132 (F16/F17)."""
def temperature(config, epoch_zero_based):
    if epoch_zero_based < 0:
        raise ValueError("Epoch must be nonnegative")
    return max(config.tau_initial * config.beta ** epoch_zero_based, config.tau_minimum)

def changed_fraction(current, previous):
    if previous is None:
        return None
    if len(current) == 0 or len(current) != len(previous):
        raise ValueError("Assignments must be nonempty and identically ordered")
    return sum(int(a != b) for a, b in zip(current, previous)) / len(current)

def should_stop(current, previous, epoch_one_based, config):
    change = changed_fraction(current, previous)
    return (epoch_one_based >= config.stopping_first_epoch and
            change is not None and change < config.stopping_tolerance)
