from collections import defaultdict

from sympy import Add, Mul, collect

from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import estimate_cost, retrieve_scalars
from devito.tools import ReducerMap
from devito.ir.support import IterationSpace, SEQUENTIAL
from devito.ir.support.space import Interval
from devito.ir.clusters.cluster import Cluster
from devito.symbolics import uxreplace, xreplace_indices
from devito.logger import warning

__all__ = ['skewing']


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
    # What dimensions so we target?
    # 0) all sequential Dimensions
    skew_dims = [] 
    for i in cluster.ispace:
        if SEQUENTIAL in cluster.properties[i.dim]:
            skew_dims.append(i.dim)
            
    
    root_names = {i.dim.root.name for i in cluster.ispace.intervals if not i.dim in skew_dims}

    # Remove candidates
    
    mapper, intervals = {}, []
    passed = []

    try:
        if len(skew_dims) > 1:
            raise warning("More than 1 dimensions that can be skewed. Skewing the first one")
        skew_dim = skew_dims[0] #get first one
        intervals.append(Interval(skew_dim, 0, 0))
        index = intervals.index(Interval(skew_dim, 0, 0))
    except BaseException:
        # No time dimensions -> nothing to do
        return cluster
    
    # Initializing a default time_dim index position in loop
    index = 0
    # Skew dim will not be none here:
    
    for i in cluster.ispace.intervals:
        if i.dim not in skew_dims:
            if index < cluster.ispace.intervals.index(i) and i.dim.name in root_names:
                mapper[i.dim] = i.dim - skew_dim
                intervals.append(Interval(i.dim, skew_dim, skew_dim))
                root_names.remove(i.dim.root.name)
            else:
                intervals.append(Interval(i.dim, 0, 0))

        processed = xreplace_indices(cluster.exprs, mapper)

    ispace = IterationSpace(intervals, cluster.ispace.sub_iterators, cluster.ispace.directions)
    cluster = Cluster(processed, ispace, cluster.dspace, guards=cluster.guards, properties=cluster.properties)

    print(ispace)    

    print(mapper)    
    return cluster # .rebuild(processed)


