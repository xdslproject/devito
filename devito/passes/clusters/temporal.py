from devito.passes.clusters.utils import cluster_pass
from devito.ir.support import IterationSpace, SEQUENTIAL
from devito.ir.support.space import Interval
from devito.symbolics import xreplace_indices
from devito.logger import warning

__all__ = ['skewing']


@cluster_pass
def skewing(cluster, *args):

    """
    Skew the accesses along a SEQUENTIAL Dimension.
    Example:

    Transform

    for i = 2, n-1
        for j = 2, m-1
            a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4
        end for
    end for

    to

    for i = 2, n-1
        for j = 2+i, m-1+i
            a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4
        end for
    end for
    """
    processed = []
    # What dimensions do we target?
    # a) SEQUENTIAL Dimensions
    skew_dims = []
    for i in cluster.ispace:
        if SEQUENTIAL in cluster.properties[i.dim]:
            skew_dims.append(i.dim)

    skewable = {i.dim for i in cluster.ispace.intervals
                if not i.dim.symbolic_incr.is_Symbol}

    # Remove candidates
    mapper, intervals = {}, []

    try:
        if len(skew_dims) > 1:
            raise warning("More than 1 dimensions that can be skewed.\
                          Skewing the first one")
        skew_dim = skew_dims[0]  # Skew first one
        intervals.append(Interval(skew_dim, 0, 0))
        index = intervals.index(Interval(skew_dim, 0, 0))
    except BaseException:
        # No time dimensions -> nothing to do
        return cluster

    # Skew dim will not be none here:
    # Initializing a default skewed dim index position in loop
    index = 0

    for i in cluster.ispace.intervals:
        if i.dim not in skew_dims:
            if index < cluster.ispace.intervals.index(i) and i.dim in skewable:
                mapper[i.dim] = i.dim - skew_dim
                intervals.append(Interval(i.dim, skew_dim, skew_dim))
                skewable.remove(i.dim)
            else:
                intervals.append(Interval(i.dim, 0, 0))

        processed = xreplace_indices(cluster.exprs, mapper)

    ispace = IterationSpace(intervals, cluster.ispace.sub_iterators,
                            cluster.ispace.directions)

    rebuilt_cluster = cluster.rebuild(exprs=processed, ispace=ispace)

    return rebuilt_cluster
