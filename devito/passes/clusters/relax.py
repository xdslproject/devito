from devito.passes.clusters.utils import cluster_pass
from devito.types import IncrDimension, Symbol
from devito.ir.support import Interval, IntervalGroup, IterationSpace
from devito.ir.clusters import Queue
from devito.tools import split, as_list, as_tuple
from devito.symbolics import xreplace_indices, MIN, MAX, uxreplace

__all__ = ['relaxing']


def relaxing(clusters, options):
    """
    """

    return Relaxing(options).process(clusters)


class Relaxing(Queue):

    def __init__(self, options):
        self.levels = options['blocklevels']

        super(Relaxing, self).__init__()

    def callback(self, clusters, prefix):
        """
        Relax min/max bounds.

        Parameters
        ----------
        cluster : Cluster
            Input Cluster, subject of the optimization pass.

        options : dict
            The optimization options.

        """
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not d.is_Incr:
            return clusters

        root_max = d.root.symbolic_max

        # import pdb;pdb.set_trace()

        try:
            iter_max = (min(d.symbolic_max, root_max))
            bool(iter_max)  # Can it be evaluated?
        except TypeError:
            iter_max = MIN(d.symbolic_max, root_max)

        # d_new = d.func(max=iter_max)

        d_new = IncrDimension(d.name, d.parent, d.symbolic_min, iter_max, step=d.step)
        processed = []

        for c in clusters:

            it_ints = [i for i in c.ispace if i.dim.is_Incr]
            outer, inner = split(it_ints, lambda i: not i.dim.parent.is_Incr)

            # if d exists in inner
            relatives = [d]
            if any(d is i.dim for i in inner):
                for o in outer:
                    if o.dim is d.parent:
                        relatives.append(o.dim)

            # relatives.append(d)

            mapper = {}
            mapper[d] = d_new
            for r in relatives[1:]:
                mapper[r] = IncrDimension(r.name, r.parent, r.symbolic_min,
                                          r.symbolic_max)

            # Create the new Intervals
            intervals = []
            for i in c.ispace:
                if i.dim in relatives:
                    intervals.append(i.switch(mapper[i.dim]))
                else:
                    intervals.append(i)

            relations = set()
            for r in c.ispace.intervals.relations:
                rel = as_list(r)
                new_rel = []
                for ii in rel:
                    new_i = ii.xreplace(mapper)
                    new_rel.append(new_i)

                relations.add(as_tuple(new_rel))
                '''
                elif d in r:
                    index = r.index(d)
                    nrel = list(r)
                    nrel[index] = d_new
                    relations.add(tuple(nrel))
                else:
                    relations.add(r)
                '''

            # import pdb;pdb.set_trace()

            directions = dict(c.ispace.directions)
            for rel in relatives:
                directions.pop(rel, None)
            directions.update({mapper[rel]: c.ispace.directions.get(rel, [])
                               for rel in relatives})

            sub_iterators = dict(c.ispace.sub_iterators)
            for rel in relatives:
                sub_iterators.pop(rel, None)
            sub_iterators.update({mapper[rel]: c.ispace.sub_iterators.get(rel, [])
                                  for rel in relatives})

            # try:
            #    assert len(sub_iterators) == len(c.ispace.sub_iterators)
            #except:
            # import pdb;pdb.set_trace()
            #assert len(sub_iterators[d_new]) == len(c.ispace.sub_iterators[i.dim])

            intervals = IntervalGroup(intervals, relations=relations)
            ispace = IterationSpace(intervals, sub_iterators, directions)
            # Use the innermost IncrDimension in place of `d`
            exprs = [uxreplace(e, mapper) for e in c.exprs]

            properties = dict(c.properties)
            for rel in relatives:
                properties.pop(rel, None)
            properties.update({mapper[rel]: c.properties.get(rel, [])
                               for rel in relatives})

            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       properties=properties))

        # import pdb;pdb.set_trace()
        return processed
'''

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

                d_new = i.dim.func(max=iter_max)
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

        # Use the innermost IncrDimension in place of `d`
        exprs = [uxreplace(e, {i.dim: d_new}) for e in c.exprs]

        return c.rebuild(exprs=exprs, ispace=ispace, properties=properties)

'''
'''

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

                d_new = i.dim.func(max=iter_max)
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

        # Required to build intervals
        # relations = c.ispace.relations

        # Required to build IterationSpace
        intervals = IntervalGroup(new_intervals, relations=new_relations)

        # Required to build a cluster
        properties = dict(c.properties)
        ispace = IterationSpace(intervals, sub_iterators, directions)

        # Use the innermost IncrDimension in place of `d`
        exprs = [uxreplace(e, {i.dim: d_new}) for e in c.exprs]

        return c.rebuild(exprs=exprs, ispace=ispace, properties=properties)
'''