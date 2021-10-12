from devito.symbolics import MIN, MAX

__all__ = ['evalmax', 'evalmin']


def evalmax(a, b):
    """
    Simplify max(a, b) if possible
    """
    try:
        bool(max(a, b))  # Can it be evaluated?
        return max(a, b)
    except TypeError:
        return MAX(a, b)


def evalmin(a, b):
    """
    Simplify min(a, b) expressions if possible
    """
    try:
        bool(min(a, b))  # Can it be evaluated?
        return min(a, b)
    except TypeError:
        return MIN(a, b)
