from collections import defaultdict

from sympy import Add, Mul, collect

from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import estimate_cost, retrieve_scalars
from devito.tools import ReducerMap
from devito.ir.support import IterationSpace
from devito.ir.support.space import Interval
from devito.ir.clusters.cluster import Cluster
from devito.symbolics import uxreplace, xreplace_indices
from devito.logger import warning

__all__ = ['skewing']


"""
Minimum operation count of an expression so that aggressive factorization
is applied.
"""


@cluster_pass
def skewing(cluster, *args):

    """
    Skew the accesses along the time Dimension.
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
    
    itintervals = cluster.ispace.itintervals 
    itdimensions = cluster.ispace.itdimensions
    sub_iterators = cluster.ispace.sub_iterators
    directions = cluster.ispace.directions
    

    skew_dims = {i.dim for i in cluster.ispace.intervals if i.dim.is_Time}
    root_names = {i.dim.root.name for i in cluster.ispace.intervals if not i.dim.is_Time}

    # Remove candidates
    
    mapper, intervals = {}, []
    passed = []

    try:
        skew_dim = skew_dims.pop()
        intervals.append(Interval(skew_dim, 0, 0))
        index = intervals.index(Interval(skew_dim, 0, 0))
        passed.append(skew_dim)
    except KeyError:
        # No time dimensions -> nothing to do
        return cluster
    
    if len(skew_dims) > 0:
        raise warning("More than 1 time dimensions. Avoid skewing")
        return cluster

    # Initializing a default time_dim index position in loop
    index = 0
    # Skew dim will not be none here:
    
    for i in cluster.ispace.itintervals:
        if i.dim not in passed:
            if index < cluster.ispace.itintervals.index(i) and i.dim.name in root_names:
                mapper[i.dim] = i.dim - skew_dim
                intervals.append(Interval(i.dim, skew_dim, skew_dim))
                root_names.remove(i.dim.root.name)
            else:
                intervals.append(Interval(i.dim, 0, 0))

        processed = xreplace_indices(cluster.exprs, mapper)

    ispace = IterationSpace(intervals, sub_iterators, directions)
    cluster = Cluster(processed, ispace, cluster.dspace, guards=cluster.guards, properties=cluster.properties)

    print(ispace)    

    print(mapper)    
    return cluster.rebuild(processed)


