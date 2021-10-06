from devito.passes.clusters.utils import cluster_pass
from devito.types import IncrDimension, Symbol
from devito.ir.support import Interval, IntervalGroup, IterationSpace
from devito.tools import split
from devito.symbolics import xreplace_indices, MIN, MAX

__all__ = ['relax']


class Temp(Symbol):
    pass


@cluster_pass
def relax(c, options):
    """
    Relax min/max bounds.

    Parameters
    ----------
    cluster : Cluster
        Input Cluster, subject of the optimization pass.

    options : dict
        The optimization options.

    """
    if not any(i.dim.is_Incr for i in c.ispace):
        return c

    # Create the new Intervals
    # intervals = []
    it_ints = [i for i in c.ispace if i.dim.is_Incr]
    outer, inner = split(it_ints, lambda i: not i.dim.parent.is_Incr)

    roots_max = {i.dim.root: i.dim.root.symbolic_max for i in c.ispace}

    new_dims = {}
    new_intervals = []
    new_relations = set()

    for i in c.ispace:
        if i in inner:
            root_max = roots_max[i.dim.root]

            try:
                iter_max = (min(i.dim.symbolic_max, root_max))
                bool(iter_max)  # Can it be evaluated?
            except TypeError:
                iter_max = MIN(i.dim.symbolic_max, root_max)

            d_new = IncrDimension(i.dim.name, i.dim.parent, i.dim.symbolic_min,
                                  iter_max, step=i.dim.step, size=i.dim.size)
            new_intervals.append(Interval(d_new, i.lower, i.upper))

            new_dims[i.dim] = d_new
            exprs = xreplace_indices(c.exprs, {i.dim: d_new})

            properties = dict(c.properties)
            properties[d_new] = properties[i.dim]
            properties.pop(i.dim)

            directions = dict(c.directions)
            directions[d_new] = directions[i.dim]
            directions.pop(i.dim)

            sub_iterators = dict(c.ispace.sub_iterators)
            sub_iterators.pop(i.dim, None)
            sub_iterators.update({d_new: c.ispace.sub_iterators.get(i.dim, [])})
            assert len(sub_iterators) == len(c.ispace.sub_iterators)
            assert len(sub_iterators[d_new]) == len(c.ispace.sub_iterators[i.dim])

            assert len(properties) == len(c.properties)

            for j in c.ispace.intervals.relations:
                if not j:
                    new_relations.add(j)
                elif i.dim in j:
                    index = j.index(i.dim)
                    nrel = list(j)
                    nrel[index] = d_new
                    new_relations.add(tuple(nrel))
                else:
                    new_relations.add(j)

        else:
            new_intervals.append(i)

    try:
        assert len(c.ispace) == len(new_intervals)
        assert len(new_relations) == len(c.ispace.intervals.relations)
    except AssertionError:
        import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()

    # Required to build intervals
    # relations = c.ispace.relations

    # Required to build IterationSpace
    intervals = IntervalGroup(new_intervals, relations=new_relations)

    # Required to build a cluster
    properties = dict(c.properties)
    ispace = IterationSpace(intervals, sub_iterators, directions)

    return c.rebuild(exprs=exprs, ispace=ispace, properties=properties)
