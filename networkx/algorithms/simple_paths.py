# -*- coding: utf-8 -*-
#    Copyright (C) 2012 by
#    Sergio Nery Simoes <sergionery@gmail.com>
#    All rights reserved.
#    BSD license.
import collections
import math
import bisect
import cProfile, pstats, io
import logging
import networkx as nx

from heapq import heappush, heappop
from itertools import count, islice
from networkx.algorithms.shortest_paths.weighted import LY_shortest_path_with_attrs, _LY_dijkstra
from bisect import bisect_left
from networkx.utils import not_implemented_for
from networkx.utils import pairwise
from operator import itemgetter
from collections import deque
from networkx.utils import generate_unique_node


__author__ = """\n""".join(['Sérgio Nery Simões <sergionery@gmail.com>',
                            'Aric Hagberg <aric.hagberg@gmail.com>',
                            'Andrey Paramonov',
                            'Jordi Torrents <jordi.t21@gmail.com>'])

__all__ = [
    'all_simple_paths',
    'is_simple_path',
    'shortest_simple_paths',
]

##############----------- The functions for the full and heuristic Pareto sets are:
##############----------- 1. get_full_pareto_set; 
##############----------- 2. get_ε_pareto_set; 
##############----------- 3. get_bucket_pareto_set;
##############----------- 4. get_ε_ratio_pareto_set;
##############----------- 5. get_ratio_bucket_pareto_set


def _weight_function(G, weight):
  """Returns a function that returns the weight of an edge.
  The returned function is specifically suitable for input to
  functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.
  Parameters
  ----------
  G : NetworkX graph.
  weight : string or function
      If it is callable, `weight` itself is returned. If it is a string,
      it is assumed to be the name of the edge attribute that represents
      the weight of an edge. In that case, a function is returned that
      gets the edge weight according to the specified edge attribute.
  Returns
  -------
  function
      This function returns a callable that accepts exactly three inputs:
      a node, an node adjacent to the first one, and the edge attribute
      dictionary for the eedge joining those nodes. That function returns
      a number representing the weight of an edge.
  If `G` is a multigraph, and `weight` is not callable, the
  minimum edge weight over all parallel edges is returned. If any edge
  does not have an attribute with key `weight`, it is assumed to
  have weight one.
  """
  if callable(weight):
    return weight
  # If the weight keyword argument is not callable, we assume it is a
  # string representing the edge attribute containing the weight of
  # the edge.
  if G.is_multigraph():
    return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
  return lambda u, v, data: data.get(weight, 1)



def is_simple_path(G, nodes):
    """Returns True if and only if the given nodes form a simple path in
    `G`.

    A *simple path* in a graph is a nonempty sequence of nodes in which
    no node appears more than once in the sequence, and each adjacent
    pair of nodes in the sequence is adjacent in the graph.

    Parameters
    ----------
    nodes : list
        A list of one or more nodes in the graph `G`.

    Returns
    -------
    bool
        Whether the given list of nodes represents a simple path in
        `G`.

    Notes
    -----
    A list of zero nodes is not a path and a list of one node is a
    path. Here's an explanation why.

    This function operates on *node paths*. One could also consider
    *edge paths*. There is a bijection between node paths and edge
    paths.

    The *length of a path* is the number of edges in the path, so a list
    of nodes of length *n* corresponds to a path of length *n* - 1.
    Thus the smallest edge path would be a list of zero edges, the empty
    path. This corresponds to a list of one node.

    To convert between a node path and an edge path, you can use code
    like the following::

        >>> from networkx import pairwise
        >>> nodes = [0, 1, 2, 3]
        >>> edges = list(pairwise(nodes))
        >>> edges
        [(0, 1), (1, 2), (2, 3)]
        >>> nodes = [edges[0][0]] + [v for u, v in edges]
        >>> nodes
        [0, 1, 2, 3]

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.is_simple_path(G, [2, 3, 0])
    True
    >>> nx.is_simple_path(G, [0, 2])
    False

    """
    # The empty list is not a valid path. Could also return
    # NetworkXPointlessConcept here.
    if len(nodes) == 0:
        return False
    # If the list is a single node, just check that the node is actually
    # in the graph.
    if len(nodes) == 1:
        return nodes[0] in G
    # Test that no node appears more than once, and that each
    # adjacent pair of nodes is adjacent.
    return (len(set(nodes)) == len(nodes) and
            all(v in G[u] for u, v in pairwise(nodes)))


def all_simple_paths(G, source, target, cutoff=None):
    """Generate all simple paths in the graph G from source to target.

    A simple path is a path with no repeated nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : nodes
       Single node or iterable of nodes at which to end path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.  If there are no paths
       between the source and target within the given cutoff the generator
       produces no output.

    Examples
    --------
    This iterator generates lists of nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=3):
        ...     print(path)
        ...
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]

    You can generate only those paths that are shorter than a certain
    length by using the `cutoff` keyword argument::

        >>> paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)
        >>> print(list(paths))
        [[0, 1, 3], [0, 2, 3], [0, 3]]

    To get each path as the corresponding list of edges, you can use the
    :func:`networkx.utils.pairwise` helper function::

        >>> paths = nx.all_simple_paths(G, source=0, target=3)
        >>> for path in map(nx.utils.pairwise, paths):
        ...     print(list(path))
        [(0, 1), (1, 2), (2, 3)]
        [(0, 1), (1, 3)]
        [(0, 2), (2, 1), (1, 3)]
        [(0, 2), (2, 3)]
        [(0, 3)]

    Pass an iterable of nodes as target to generate all paths ending in any of several nodes::

        >>> G = nx.complete_graph(4)
        >>> for path in nx.all_simple_paths(G, source=0, target=[3, 2]):
        ...     print(path)
        ...
        [0, 1, 2]
        [0, 1, 2, 3]
        [0, 1, 3]
        [0, 1, 3, 2]
        [0, 2]
        [0, 2, 1, 3]
        [0, 2, 3]
        [0, 3]
        [0, 3, 1, 2]
        [0, 3, 2]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph using a functional programming approach::

        >>> from itertools import chain
        >>> from itertools import product
        >>> from itertools import starmap
        >>> from functools import partial
        >>>
        >>> chaini = chain.from_iterable
        >>>
        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = partial(nx.all_simple_paths, G)
        >>> list(chaini(starmap(all_paths, product(roots, leaves))))
        [[0, 1, 2], [0, 3, 2]]

    The same list computed using an iterative approach::

        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = (v for v, d in G.out_degree() if d == 0)
        >>> all_paths = []
        >>> for root in roots:
        ...     for leaf in leaves:
        ...         paths = nx.all_simple_paths(G, root, leaf)
        ...         all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 2], [0, 3, 2]]

    Iterate over each path from the root nodes to the leaf nodes in a
    directed acyclic graph passing all leaves together to avoid unnecessary
    compute::

        >>> G = nx.DiGraph([(0, 1), (2, 1), (1, 3), (1, 4)])
        >>> roots = (v for v, d in G.in_degree() if d == 0)
        >>> leaves = [v for v, d in G.out_degree() if d == 0]
        >>> all_paths = []
        >>> for root in roots:
        ...     paths = nx.all_simple_paths(G, root, leaves)
        ...     all_paths.extend(paths)
        >>> all_paths
        [[0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4]]

    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of simple paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    all_shortest_paths, shortest_path

    """
    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError:
            raise nx.NodeNotFound('target node %s not in graph' % target)
    if source in targets:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    if G.is_multigraph():
        return _all_simple_paths_multigraph(G, source, targets, cutoff)
    else:
        return _all_simple_paths_graph(G, source, targets, cutoff)


def _all_simple_paths_graph(G, source, targets, cutoff):
    visited = collections.OrderedDict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()


def _all_simple_paths_multigraph(G, source, targets, cutoff):
    visited = collections.OrderedDict.fromkeys([source])
    stack = [(v for u, v in G.edges(source))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):
                stack.append((v for u, v in G.edges(child)))
            else:
                visited.popitem()
        else:  # len(visited) == cutoff:
            for target in targets - set(visited.keys()):
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield list(visited) + [target]
            stack.pop()
            visited.popitem()


@not_implemented_for('multigraph')
def shortest_simple_paths(G, source, target, weight=None):
    """Generate all simple paths in the graph G from source to target,
       starting from shortest ones.

    A simple path is a path with no repeated nodes.

    If a weighted shortest path search is to be used, no negative weights
    are allowed.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    weight : string
        Name of the edge attribute to be used as a weight. If None all
        edges are considered to have unit weight. Default value None.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths, in order from
       shortest to longest.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    NetworkXError
       If source or target nodes are not in the input graph.

    NetworkXNotImplemented
       If the input graph is a Multi[Di]Graph.

    Examples
    --------

    >>> G = nx.cycle_graph(7)
    >>> paths = list(nx.shortest_simple_paths(G, 0, 3))
    >>> print(paths)
    [[0, 1, 2, 3], [0, 6, 5, 4, 3]]

    You can use this function to efficiently compute the k shortest/best
    paths between two nodes.

    >>> from itertools import islice
    >>> def k_shortest_paths(G, source, target, k, weight=None):
    ...     return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    >>> for path in k_shortest_paths(G, 0, 3, 2):
    ...     print(path)
    [0, 1, 2, 3]
    [0, 6, 5, 4, 3]

    Notes
    -----
    This procedure is based on algorithm by Jin Y. Yen [1]_.  Finding
    the first $K$ paths requires $O(KN^3)$ operations.

    See Also
    --------
    all_shortest_paths
    shortest_path
    all_simple_paths

    References
    ----------
    .. [1] Jin Y. Yen, "Finding the K Shortest Loopless Paths in a
       Network", Management Science, Vol. 17, No. 11, Theory Series
       (Jul., 1971), pp. 712-716.

    """
    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)

    if target not in G:
        raise nx.NodeNotFound('target node %s not in graph' % target)

    if weight is None:
        length_func = len
        shortest_path_func = _bidirectional_shortest_path
    else:
        def length_func(path):
            return sum(G.adj[u][v][weight] for (u, v) in zip(path, path[1:]))
        shortest_path_func = _bidirectional_dijkstra

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = shortest_path_func(G, root[-1], target,
                                                      ignore_nodes=ignore_nodes,
                                                      ignore_edges=ignore_edges,
                                                      weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break


class PathBuffer(object):

    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path


def _bidirectional_shortest_path(G, source, target,
                                 ignore_nodes=None,
                                 ignore_edges=None,
                                 weight=None):
    """Returns the shortest path between source and target ignoring
       nodes and edges in the containers ignore_nodes and ignore_edges.

    This is a custom modification of the standard bidirectional shortest
    path implementation at networkx.algorithms.unweighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       starting node for path

    target : node
       ending node for path

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    weight : None
       This function accepts a weight argument for convenience of
       shortest_simple_paths function. It will be ignored.

    Returns
    -------
    path: list
       List of nodes in a path from source to target.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    See Also
    --------
    shortest_path

    """
    # call helper to do the real work
    results = _bidirectional_pred_succ(G, source, target, ignore_nodes, ignore_edges)
    pred, succ, w = results

    # build path from pred+w+succ
    path = []
    # from w to target
    while w is not None:
        path.append(w)
        w = succ[w]
    # from source to w
    w = pred[path[0]]
    while w is not None:
        path.insert(0, w)
        w = pred[w]

    return len(path), path


def _bidirectional_pred_succ(G, source, target, ignore_nodes=None, ignore_edges=None):
    """Bidirectional shortest path helper.
       Returns (pred,succ,w) where
       pred is a dictionary of predecessors from w to the source, and
       succ is a dictionary of successors from w to the target.
    """
    # does BFS from both source and target and meets in the middle
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
    if target == source:
        return ({target: None}, {source: None}, source)

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():
            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w
                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w
                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:
            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges \
                                and (w, v) not in ignore_edges:
                            yield w
                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    # predecesssor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc(v):
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        # found path
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred(v):
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        # found path
                        return pred, succ, w

    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))


def _bidirectional_dijkstra(G, source, target, weight='weight',
                            ignore_nodes=None, ignore_edges=None):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    This function returns the shortest path between source and target
    ignoring nodes and edges in the containers ignore_nodes and
    ignore_edges.

    This is a custom modification of the standard Dijkstra bidirectional
    shortest path implementation at networkx.algorithms.weighted

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node.

    target : node
       Ending node.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    Returns a tuple of two dictionaries keyed by node.
    The first dictionary stores distance from the source.
    The second stores the path from the source to that node.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is pi*r*r while the
    others are 2*pi*r/2*r/2, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
    if source == target:
        return (0, [source])

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():
            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w
                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w
                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:
            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges \
                                and (w, v) not in ignore_edges:
                            yield w
                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{source: 0}, {target: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if(dir == 0):  # forward
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[v][w].items()))
                else:
                    minweight = G[v][w].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[v][w].get(weight,1)
            else:  # back, must remember to change v,w->w,v
                if G.is_multigraph():
                    minweight = min((dd.get(weight, 1)
                                     for k, dd in G[w][v].items()))
                else:
                    minweight = G[w][v].get(weight, 1)
                vwLength = dists[dir][v] + minweight  # G[w][v].get(weight,1)

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError(
                        "Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
    

def get_full_pareto_set(G, source, target, req_time, t_0, t_H, dt):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))
  full_pareto_bag = mltcrtr_lbl_set_alg_bwds(G, source, target, discrete_t_0, discrete_t_H, dt)
  
  path_id = count()
  pareto_set = dict()
  missed_paths = 0
  
  for label_id, attrs in full_pareto_bag[source][discrete_request_time].items():
      path = deque([source])
      next_node = attrs['pred_node']
      next_time_intrv = attrs['pred_time_int']
      next_label_id = attrs['pred_label_id']
      update_path_dict = True
      while next_node != None and next_time_intrv != None and next_label_id != None:
          if next_label_id not in full_pareto_bag[next_node][next_time_intrv]:
              missed_paths += 1
              update_path_dict = False
              break
          path.append(next_node)
          new_node = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_node']
          new_time_intrv = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_time_int']
          new_label_id = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_label_id']
          next_node = new_node
          next_time_intrv = new_time_intrv
          next_label_id = new_label_id
      if update_path_dict:
          pareto_set.update({str(next(path_id)): {'path' : path, 'label' : attrs['opt_crt_val']}})
  
  return (pareto_set, missed_paths)


def mltcrtr_lbl_set_alg_bwds(G, source, target, t_0, t_H, dt):
    Gpred = G._pred
    
    c = count()
    labels_bag = dict()
    labels_to_extend = dict()
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: dict()})
        for t in range(t_0, t_H+dt, dt):
            # t = t%86400
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                label_id = str(next(c))
                labels_bag[n].update({t_1: {label_id: {'opt_crt_val': (0,0,0), 'pred_node': None, \
                                                     'pred_time_int': None, 'pred_label_id': None, 'prev_edge_type': None, \
                                                         'prev_dstr_node_graph_type': None , 'prev_mode': None}}})
                labels_to_extend[n].update({t_1: {label_id}})
            else:
                labels_bag[n].update({t_1: dict()})
                labels_to_extend[n].update({t_1: set()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])

    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False
            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']
                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    # e_distance = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    # e_distance = 0
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    # e_distance = e['distance']
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if u_n_type == 'walk_graph_node' and (v_n_type != 'stop_node' and v_n_type != 'station_node'):
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    # e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time
                    
                if rolled_v_arr_time <= t_H:                     
                    for label_id, info in labels_bag[v][v_arr_time].items():
                        if label_id not in labels_to_extend[v][v_arr_time]:
                            continue
                        if u == info['pred_node']:
                            continue
                        prev_mode = info['prev_mode']
                        pr_ed_tp = info['prev_edge_type']
                        pre_dstr_n_gr_tp = info['prev_dstr_node_graph_type']
                        
                        # restraint walking before taxi modes - active
                        if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                        (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                            continue
                        
                        if e_type == 'access_edge':
                            u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                            u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                            v_n_type = e['dstr_node_type']
                            
                            if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            
                            if u_n_gr_type == 'Walk':
                              prev_mode = v_n_gr_type   
#                           when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; 
#                           e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip 
#                           and back to taxi/carsharing trip
                            if pr_ed_tp == 'access_edge': #active
                              if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or (pre_dstr_n_gr_tp == 'Bus' and \
                                u_n_gr_type == 'Bus') or (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                  continue#penalty = 1000000000000000
                             
#                           avoid paths that include two consecutive taxis or carsharign legs in one trip
                            if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                                prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                                (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                                 u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                    continue#penalty = 1000000000000000 active
                            
                            # restraint pick up -active
                            if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                                if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                                  continue#penalty = 1000000000000000
                              
                            # restraint drop off - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                                continue
                            
                            # restraint walking after taxi modes - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and pr_ed_tp == 'walk_edge':
                                continue
                          
                        total_travel_time_till_u = e_tt + e_wait_time + info['opt_crt_val'][0]
                        cost_till_u = e_cost + info['opt_crt_val'][1]
                        # walk_till_u = e_walk_time + info['opt_crt_val'][2]
                        trips_till_u = e_trip_num + info['opt_crt_val'][2]
                        # wait_time_till_u = e_wait_time + info['wt']
                        # distance_till_u = e_distance + info['l']
                        # walk_time_till_u = e_walk_time + info['wkt']
                        
                        new_cost_label = (total_travel_time_till_u, cost_till_u, trips_till_u)
                        criteria_num = len(new_cost_label)
                        labels_to_be_deleted = deque([])
                        if not(labels_bag[u][t_2]):
                            non_dominated_label = 1
                        else:
                            for label, label_info in labels_bag[u][t_2].items():
                                temp_pareto_cost_label = label_info['opt_crt_val']
                                
                                if new_cost_label == temp_pareto_cost_label:
                                    non_dominated_label = 1
                                    break
                                q_1 = 0 
                                q_2 = 0
                                for i, j in zip(new_cost_label, temp_pareto_cost_label):
                                    if i>=j:
                                        q_1 += 1
                                    if i==j:
                                        q_2 += 1
                                if q_1 == criteria_num and q_2 != criteria_num:
                                    non_dominated_label = 0
                                    break
                                q_3 = 0
                                q_4 = 0
                                for i, j in zip(new_cost_label, temp_pareto_cost_label):
                                    if i<=j:
                                        q_3 += 1
                                    if i==j:
                                        q_4 += 1
                                if q_3 == criteria_num and q_4 != criteria_num:
                                    labels_to_be_deleted.append(label)
                                non_dominated_label = 1
                      
                        if non_dominated_label:
                            if labels_to_be_deleted:
                                for labelid in labels_to_be_deleted:                                      
                                    del(labels_bag[u][t_2][labelid])
                                    labels_to_extend[u][t_2].discard(labelid)
                            insert_in_se_list = True     
                            new_label_id = str(next(c))
                            labels_to_extend[u][t_2].add(new_label_id)
                            labels_bag[u][t_2].update({new_label_id: {'opt_crt_val' : new_cost_label, 'pred_node' : v, \
                                                                    'pred_time_int': v_arr_time, 'pred_label_id' : label_id, \
                                                                        'prev_edge_type': e_type, \
                                                                            'prev_dstr_node_graph_type': v_n_gr_type, \
                                                                                'prev_mode': prev_mode}})
                                                                                                      
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        
        for time in labels_to_extend[v]:
            labels_to_extend[v][time] = set()
    
    return labels_bag

def get_ε_pareto_set(G, source, target, req_time, t_0, t_H, dt, ε):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))
  full_pareto_bag = ε_mltcrtr_lbl_set_alg_bwds(G, source, target, discrete_t_0, discrete_t_H, dt, ε)
  
  path_id = count()
  pareto_set = dict()
  missed_paths = 0
  
  for label_id, attrs in full_pareto_bag[source][discrete_request_time].items():
      path = deque([source])
      next_node = attrs['pred_node']
      next_time_intrv = attrs['pred_time_int']
      next_label_id = attrs['pred_label_id']
      update_path_dict = True
      while next_node != None and next_time_intrv != None and next_label_id != None:
          if next_label_id not in full_pareto_bag[next_node][next_time_intrv]:
              missed_paths += 1
              update_path_dict = False
              break
          path.append(next_node)
          new_node = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_node']
          new_time_intrv = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_time_int']
          new_label_id = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_label_id']
          next_node = new_node
          next_time_intrv = new_time_intrv
          next_label_id = new_label_id
      if update_path_dict:
          pareto_set.update({str(next(path_id)): {'path' : path, 'label' : attrs['opt_crt_val']}})
  
  return (pareto_set, missed_paths)


def ε_mltcrtr_lbl_set_alg_bwds(G, source, target, t_0, t_H, dt, ε):
    Gpred = G._pred
    
    c = count()
    labels_bag = dict()
    labels_to_extend = dict()
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: dict()})
        for t in range(t_0, t_H+dt, dt):
            # t = t%86400
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                label_id = str(next(c))
                labels_bag[n].update({t_1: {label_id: {'opt_crt_val': (0,0,0), 'pred_node': None, \
                                                     'pred_time_int': None, 'pred_label_id': None, 'prev_edge_type': None, \
                                                         'prev_dstr_node_graph_type': None , 'prev_mode': None}}})
                labels_to_extend[n].update({t_1: {label_id}})
            else:
                labels_bag[n].update({t_1: dict()})
                labels_to_extend[n].update({t_1: set()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])
    
    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False
            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']
                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    # e_distance = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    # e_distance = 0
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    # e_distance = e['distance']
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if u_n_type == 'walk_graph_node' and (v_n_type != 'stop_node' and v_n_type != 'station_node'):
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    # e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time
                    
                if rolled_v_arr_time <= t_H:                     
                    for label_id, info in labels_bag[v][v_arr_time].items():
                        if label_id not in labels_to_extend[v][v_arr_time]:
                            continue
                        if u == info['pred_node']:
                            continue
                        prev_mode = info['prev_mode']
                        pr_ed_tp = info['prev_edge_type']
                        pre_dstr_n_gr_tp = info['prev_dstr_node_graph_type']
                        
                        # restraint walking before taxi modes - active
                        if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                        (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                            continue
                        
                        if e_type == 'access_edge':
                            u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                            u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                            v_n_type = e['dstr_node_type']
                            
                            if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            
                            if u_n_gr_type == 'Walk':
                              prev_mode = v_n_gr_type   
#                           when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; 
#                           e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip 
#                           and back to taxi/carsharing trip
                            if pr_ed_tp == 'access_edge': #active
                              if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or (pre_dstr_n_gr_tp == 'Bus' and \
                                u_n_gr_type == 'Bus') or (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                  continue#penalty = 1000000000000000
                             
#                           avoid paths that include two consecutive taxis or carsharign legs in one trip
                            if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                                prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                                (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                                 u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                    continue#penalty = 1000000000000000 active
                            
                            # restraint pick up -active
                            if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                                if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                                  continue#penalty = 1000000000000000
                              
                            # restraint drop off - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                                continue
                            
                            # restraint walking after taxi modes - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and pr_ed_tp == 'walk_edge':
                                continue
                          
                        total_travel_time_till_u = e_tt + e_wait_time + info['opt_crt_val'][0]
                        cost_till_u = e_cost + info['opt_crt_val'][1]
                        # walk_till_u = e_walk_time + info['opt_crt_val'][2]
                        trips_till_u = e_trip_num + info['opt_crt_val'][2]
                        # wait_time_till_u = e_wait_time + info['wt']
                        # distance_till_u = e_distance + info['l']
                        # walk_time_till_u = e_walk_time + info['wkt']
                        
                        new_cost_label = (total_travel_time_till_u, cost_till_u, trips_till_u)
                        criteria_num = len(new_cost_label)
                        ε_new_cost_label = [i*ε for i in new_cost_label]
                        
                        labels_to_be_deleted = deque([])
                        if not(labels_bag[u][t_2]):
                            non_dominated_label = 1
                        else:
                            check_next_loop = True
                            for label1, label_info1 in labels_bag[u][t_2].items():
                                temp_pareto_cost_label = label_info1['opt_crt_val']
                                q_1 = 0 
                                q_2 = 0
                                for i, j in zip(temp_pareto_cost_label, ε_new_cost_label):
                                    if i<=j:
                                        q_1 += 1
                                    if i==j:
                                        q_2 += 1
                                if q_1 == criteria_num and q_2 != criteria_num:
                                    non_dominated_label = 0
                                    check_next_loop = False
                                    break
                            if check_next_loop:
                                for label2, label_info2 in labels_bag[u][t_2].items():
                                    temp_pareto_cost_label = label_info2['opt_crt_val']
                                    ε_temp_pareto_cost_label = [i*ε for i in temp_pareto_cost_label]
                                    q_1 = 0
                                    q_2 = 0
                                    for i, j in zip(new_cost_label, ε_temp_pareto_cost_label):
                                        if i<=j:
                                            q_1 += 1
                                        if i==j:
                                            q_2 += 1
                                    if q_1 == criteria_num and q_2 != criteria_num:
                                        labels_to_be_deleted.append(label2)
                                    non_dominated_label = 1
                      
                        if non_dominated_label:
                            if labels_to_be_deleted:
                                for labelid in labels_to_be_deleted:                                      
                                    del(labels_bag[u][t_2][labelid])
                                    labels_to_extend[u][t_2].discard(labelid)
                            insert_in_se_list = True     
                            new_label_id = str(next(c))
                            labels_to_extend[u][t_2].add(new_label_id)
                            labels_bag[u][t_2].update({new_label_id: {'opt_crt_val' : new_cost_label, 'pred_node' : v, \
                                                                    'pred_time_int': v_arr_time, 'pred_label_id' : label_id, \
                                                                        'prev_edge_type': e_type, \
                                                                            'prev_dstr_node_graph_type': v_n_gr_type, \
                                                                                'prev_mode': prev_mode}})
                                                                                                      
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        
        for time in labels_to_extend[v]:
            labels_to_extend[v][time] = set()
    
    return labels_bag

def get_bucket_pareto_set(G, source, target, req_time, t_0, t_H, dt, time_bucket, cost_bucket):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))
  full_pareto_bag = buck_mltcrtr_lbl_set_alg_bwds(G, source, target, discrete_t_0, discrete_t_H, dt, time_bucket, cost_bucket)
  
  path_id = count()
  pareto_set = dict()
  missed_paths = 0
  
  for label_id, attrs in full_pareto_bag[source][discrete_request_time].items():
      path = deque([source])
      next_node = attrs['pred_node']
      next_time_intrv = attrs['pred_time_int']
      next_label_id = attrs['pred_label_id']
      update_path_dict = True
      while next_node != None and next_time_intrv != None and next_label_id != None:
          if next_label_id not in full_pareto_bag[next_node][next_time_intrv]:
              missed_paths += 1
              update_path_dict = False
              break
          path.append(next_node)
          new_node = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_node']
          new_time_intrv = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_time_int']
          new_label_id = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_label_id']
          next_node = new_node
          next_time_intrv = new_time_intrv
          next_label_id = new_label_id
      if update_path_dict:
          pareto_set.update({str(next(path_id)): {'path' : path, 'label' : attrs['opt_crt_val']}})
  
  return (pareto_set, missed_paths)


def buck_mltcrtr_lbl_set_alg_bwds(G, source, target, t_0, t_H, dt, time_bucket, cost_bucket):
    Gpred = G._pred
    
    c = count()
    labels_bag = dict()
    labels_to_extend = dict()
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: dict()})
        for t in range(t_0, t_H+dt, dt):
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                label_id = str(next(c))
                labels_bag[n].update({t_1: {label_id: {'opt_crt_val': (0,0,0), 'pred_node': None, \
                                                     'pred_time_int': None, 'pred_label_id': None, 'prev_edge_type': None, \
                                                         'prev_dstr_node_graph_type': None , 'prev_mode': None}}})
                labels_to_extend[n].update({t_1: {label_id}})
            else:
                labels_bag[n].update({t_1: dict()})
                labels_to_extend[n].update({t_1: set()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])
    
    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False

            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']

                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    # e_distance = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    # e_distance = 0
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    # e_distance = e['distance']
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if u_n_type == 'walk_graph_node' and (v_n_type != 'stop_node' and v_n_type != 'station_node'):
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    # e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time
                    
                if rolled_v_arr_time <= t_H:                     
                    for label_id, info in labels_bag[v][v_arr_time].items():
                        if label_id not in labels_to_extend[v][v_arr_time]:
                            continue
                        if u == info['pred_node']:
                            continue
                        prev_mode = info['prev_mode']
                        pr_ed_tp = info['prev_edge_type']
                        pre_dstr_n_gr_tp = info['prev_dstr_node_graph_type']
                        
                        # restraint walking before taxi modes - active
                        if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                        (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                            continue
                        
                        if e_type == 'access_edge':
                            u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                            u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                            v_n_type = e['dstr_node_type']
                            
                            if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            
                            if u_n_gr_type == 'Walk':
                              prev_mode = v_n_gr_type   
#                           when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; 
#                           e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip 
#                           and back to taxi/carsharing trip
                            if pr_ed_tp == 'access_edge': #active
                              if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or (pre_dstr_n_gr_tp == 'Bus' and \
                                u_n_gr_type == 'Bus') or (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                  continue#penalty = 1000000000000000
                             
#                           avoid paths that include two consecutive taxis or carsharign legs in one trip
                            if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                                prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                                (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                                 u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                    continue#penalty = 1000000000000000 active
                            
                            # restraint pick up -active
                            if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                                if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                                  continue#penalty = 1000000000000000
                              
                            # restraint drop off - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                                continue
                            
                            # restraint walking after taxi modes - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and pr_ed_tp == 'walk_edge':
                                continue

                          
                        total_travel_time_till_u = e_tt + e_wait_time + info['opt_crt_val'][0]
                        cost_till_u = e_cost + info['opt_crt_val'][1]
                        # walk_till_u = e_walk_time + info['opt_crt_val'][2]
                        trips_till_u = e_trip_num + info['opt_crt_val'][2]
                        # wait_time_till_u = e_wait_time + info['wt']
                        # distance_till_u = e_distance + info['l']
                        # walk_time_till_u = e_walk_time + info['wkt']
                        
                        new_cost_label = (total_travel_time_till_u, cost_till_u, trips_till_u)
                        criteria_num = len(new_cost_label)
                        new_buck_cost_label = (total_travel_time_till_u - (total_travel_time_till_u % time_bucket), \
                                               cost_till_u - (cost_till_u % cost_bucket), trips_till_u)
                        
                        labels_to_be_deleted = deque([])
                        if not(labels_bag[u][t_2]):
                            non_dominated_label = 1
                        else:
                            for label, label_info in labels_bag[u][t_2].items():
                                temp_buck_pareto_cost_label = (label_info['opt_crt_val'][0] - (label_info['opt_crt_val'][0] % time_bucket), \
                                                               label_info['opt_crt_val'][1] - (label_info['opt_crt_val'][1] % cost_bucket), \
                                                                   label_info['opt_crt_val'][2])
                                if new_buck_cost_label == temp_buck_pareto_cost_label:
                                    non_dominated_label = 0
                                    break
                                q_1 = 0 
                                q_2 = 0
                                for i, j in zip(temp_buck_pareto_cost_label, new_buck_cost_label):
                                    if i<=j:
                                        q_1 += 1
                                    if i==j:
                                        q_2 += 1
                                if q_1 == criteria_num and q_2 != criteria_num:
                                    non_dominated_label = 0
                                    break
                                q_3 = 0
                                q_4 = 0
                                for i, j in zip(new_buck_cost_label, temp_buck_pareto_cost_label):
                                    if i<=j:
                                        q_3 += 1
                                    if i==j:
                                        q_4 += 1
                                if q_3 == criteria_num and q_4 != criteria_num:
                                    labels_to_be_deleted.append(label)
                                non_dominated_label = 1
                      
                        if non_dominated_label:
                            if labels_to_be_deleted:
                                for labelid in labels_to_be_deleted:                                      
                                    del(labels_bag[u][t_2][labelid])
                                    labels_to_extend[u][t_2].discard(labelid)
                            insert_in_se_list = True     
                            new_label_id = str(next(c))
                            labels_to_extend[u][t_2].add(new_label_id)
                            labels_bag[u][t_2].update({new_label_id: {'opt_crt_val' : new_cost_label, 'pred_node' : v, \
                                                                    'pred_time_int': v_arr_time, 'pred_label_id' : label_id, \
                                                                        'prev_edge_type': e_type, \
                                                                            'prev_dstr_node_graph_type': v_n_gr_type, \
                                                                                'prev_mode': prev_mode}})
                                                                                                      
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        
        for time in labels_to_extend[v]:
            labels_to_extend[v][time] = set()
    
    return labels_bag

def get_ε_ratio_pareto_set(G, source, target, req_time, t_0, t_H, dt, ε):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))
  full_pareto_bag = ε_ratio_mltcrtr_lbl_set_alg_bwds(G, source, target, discrete_t_0, discrete_t_H, dt, ε)
  
  path_id = count()
  pareto_set = dict()
  missed_paths = 0
  
  for label_id, attrs in full_pareto_bag[source][discrete_request_time].items():
      path = deque([source])
      next_node = attrs['pred_node']
      next_time_intrv = attrs['pred_time_int']
      next_label_id = attrs['pred_label_id']
      update_path_dict = True
      while next_node != None and next_time_intrv != None and next_label_id != None:
          if next_label_id not in full_pareto_bag[next_node][next_time_intrv]:
              missed_paths += 1
              update_path_dict = False
              break
          path.append(next_node)
          new_node = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_node']
          new_time_intrv = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_time_int']
          new_label_id = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_label_id']
          next_node = new_node
          next_time_intrv = new_time_intrv
          next_label_id = new_label_id
      if update_path_dict:
          pareto_set.update({str(next(path_id)): {'path' : path, 'label' : attrs['opt_crt_val']}})
  
  return (pareto_set, missed_paths)


def ε_ratio_mltcrtr_lbl_set_alg_bwds(G, source, target, t_0, t_H, dt, ε):
    Gpred = G._pred
    
    c = count()
    labels_bag = dict()
    labels_to_extend = dict()
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: dict()})
        for t in range(t_0, t_H+dt, dt):
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                label_id = str(next(c))
                labels_bag[n].update({t_1: {label_id: {'opt_crt_val': (0,0,0), 'pred_node': None, \
                                                     'pred_time_int': None, 'pred_label_id': None, 'prev_edge_type': None, \
                                                         'prev_dstr_node_graph_type': None , 'prev_mode': None}}})
                labels_to_extend[n].update({t_1: {label_id}})
            else:
                labels_bag[n].update({t_1: dict()})
                labels_to_extend[n].update({t_1: set()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])
    
    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False

            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']

                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    # e_distance = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    # e_distance = 0
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    # e_distance = e['distance']
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if u_n_type == 'walk_graph_node' and (v_n_type != 'stop_node' and v_n_type != 'station_node'):
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    # e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time
                    
                if rolled_v_arr_time <= t_H:                     
                    for label_id, info in labels_bag[v][v_arr_time].items():
                        if label_id not in labels_to_extend[v][v_arr_time]:
                            continue
                        if u == info['pred_node']:
                            continue
                        prev_mode = info['prev_mode']
                        pr_ed_tp = info['prev_edge_type']
                        pre_dstr_n_gr_tp = info['prev_dstr_node_graph_type']
                        
                        # restraint walking before taxi modes - active
                        if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                        (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                            continue
                        
                        if e_type == 'access_edge':
                            u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                            u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                            v_n_type = e['dstr_node_type']
                            
                            if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            
                            if u_n_gr_type == 'Walk':
                              prev_mode = v_n_gr_type   
#                           when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; 
#                           e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip 
#                           and back to taxi/carsharing trip
                            if pr_ed_tp == 'access_edge': #active
                              if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or (pre_dstr_n_gr_tp == 'Bus' and \
                                u_n_gr_type == 'Bus') or (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                  continue#penalty = 1000000000000000
                             
#                           avoid paths that include two consecutive taxis or carsharign legs in one trip
                            if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                                prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                                (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                                 u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                    continue#penalty = 1000000000000000 active
                            
                            # restraint pick up -active
                            if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                                if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                                  continue#penalty = 1000000000000000
                              
                            # restraint drop off - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                                continue
                            
                            # restraint walking after taxi modes - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and pr_ed_tp == 'walk_edge':
                                continue
                          
                        total_travel_time_till_u = e_tt + e_wait_time + info['opt_crt_val'][0]
                        cost_till_u = e_cost + info['opt_crt_val'][1]
                        # walk_till_u = e_walk_time + info['opt_crt_val'][2]
                        trips_till_u = e_trip_num + info['opt_crt_val'][2]
                        # wait_time_till_u = e_wait_time + info['wt']
                        # distance_till_u = e_distance + info['l']
                        # walk_time_till_u = e_walk_time + info['wkt']
                        
                        new_cost_label = (total_travel_time_till_u, cost_till_u, trips_till_u)
                        criteria_num = len(new_cost_label)
                        ε_new_cost_label = [i*ε for i in new_cost_label]
                        
                        labels_to_be_deleted = deque([])
                        if not(labels_bag[u][t_2]):
                            non_dominated_label = 1
                        else:
                            check_next_loop = True
                            for label1, label_info1 in labels_bag[u][t_2].items():
                                temp_pareto_cost_label = label_info1['opt_crt_val']
                                q_1 = 0 
                                q_2 = 0
                                for i, j in zip(temp_pareto_cost_label, ε_new_cost_label):
                                    if i<=j:
                                        q_1 += 1
                                    if i==j:
                                        q_2 += 1
                                if q_1 == criteria_num and q_2 != criteria_num:
                                    non_dominated_label = 0
                                    check_next_loop = False
                                    break
                            if check_next_loop:
                                for label2, label_info2 in labels_bag[u][t_2].items():
                                    temp_pareto_cost_label = label_info2['opt_crt_val']
                                    ε_temp_pareto_cost_label = [i*ε for i in temp_pareto_cost_label]
                                    q_1 = 0
                                    q_2 = 0
                                    for i, j in zip(new_cost_label, ε_temp_pareto_cost_label):
                                        if i<=j:
                                            q_1 += 1
                                        if i==j:
                                            q_2 += 1
                                    if q_1 == criteria_num and q_2 != criteria_num:
                                        labels_to_be_deleted.append(label2)
                                    non_dominated_label = 1
                      
                        if non_dominated_label:
                            if labels_to_be_deleted:
                                for labelid in labels_to_be_deleted:                                      
                                    del(labels_bag[u][t_2][labelid])
                                    labels_to_extend[u][t_2].discard(labelid)
                            insert_in_se_list = True     
                            new_label_id = str(next(c))
                            labels_to_extend[u][t_2].add(new_label_id)
                            labels_bag[u][t_2].update({new_label_id: {'opt_crt_val' : new_cost_label, 'pred_node' : v, \
                                                                    'pred_time_int': v_arr_time, 'pred_label_id' : label_id, \
                                                                        'prev_edge_type': e_type, \
                                                                            'prev_dstr_node_graph_type': v_n_gr_type, \
                                                                                'prev_mode': prev_mode}})
                                                                                                      
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        
        for time in labels_to_extend[v]:
            labels_to_extend[v][time] = set()
    
    return labels_bag

def get_ratio_bucket_pareto_set(G, source, target, req_time, t_0, t_H, dt, time_bucket, cost_bucket):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))
  full_pareto_bag = ratio_buck_mltcrtr_lbl_set_alg_bwds(G, source, target, discrete_t_0, discrete_t_H, dt, time_bucket, cost_bucket)
  
  path_id = count()
  pareto_set = dict()
  missed_paths = 0
  
  for label_id, attrs in full_pareto_bag[source][discrete_request_time].items():
      path = deque([source])
      next_node = attrs['pred_node']
      next_time_intrv = attrs['pred_time_int']
      next_label_id = attrs['pred_label_id']
      update_path_dict = True
      while next_node != None and next_time_intrv != None and next_label_id != None:
          if next_label_id not in full_pareto_bag[next_node][next_time_intrv]:
              missed_paths += 1
              update_path_dict = False
              break
          path.append(next_node)
          new_node = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_node']
          new_time_intrv = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_time_int']
          new_label_id = full_pareto_bag[next_node][next_time_intrv][next_label_id]['pred_label_id']
          next_node = new_node
          next_time_intrv = new_time_intrv
          next_label_id = new_label_id
      if update_path_dict:
          pareto_set.update({str(next(path_id)): {'path' : path, 'label' : attrs['opt_crt_val']}})
  
  return (pareto_set, missed_paths)


def ratio_buck_mltcrtr_lbl_set_alg_bwds(G, source, target, t_0, t_H, dt, time_bucket, cost_bucket):
    Gpred = G._pred
    
    c = count()
    labels_bag = dict()
    labels_to_extend = dict()
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: dict()})
        for t in range(t_0, t_H+dt, dt):
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                label_id = str(next(c))
                labels_bag[n].update({t_1: {label_id: {'opt_crt_val': (0,0,0), 'pred_node': None, \
                                                     'pred_time_int': None, 'pred_label_id': None, 'prev_edge_type': None, \
                                                         'prev_dstr_node_graph_type': None , 'prev_mode': None}}})
                labels_to_extend[n].update({t_1: {label_id}})
            else:
                labels_bag[n].update({t_1: dict()})
                labels_to_extend[n].update({t_1: set()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])
    
    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False

            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']
                
                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    # e_distance = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    # e_distance = 0
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    # e_distance = e['distance']
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    # e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if u_n_type == 'walk_graph_node' and (v_n_type != 'stop_node' and v_n_type != 'station_node'):
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    # e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    # e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    # e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time
                    
                if rolled_v_arr_time <= t_H:                     
                    for label_id, info in labels_bag[v][v_arr_time].items():
                        if label_id not in labels_to_extend[v][v_arr_time]:
                            continue
                        if u == info['pred_node']:
                            continue
                        prev_mode = info['prev_mode']
                        pr_ed_tp = info['prev_edge_type']
                        pre_dstr_n_gr_tp = info['prev_dstr_node_graph_type']
                        
                        # restraint walking before taxi modes - active
                        if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                        (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                            continue
                        
                        if e_type == 'access_edge':
                            u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                            u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                            v_n_type = e['dstr_node_type']
                            
                            if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                                e_tt = 999999999
                                e_wait_time = 999999999
                                e_cost= 999999999
                                # e_distance = 0
                                # e_boarding_num =  999999999
                                e_trip_num = 999999999
                                # e_walk_time = 999999999
                            
                            if u_n_gr_type == 'Walk':
                              prev_mode = v_n_gr_type   
#                           when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; 
#                           e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip 
#                           and back to taxi/carsharing trip
                            if pr_ed_tp == 'access_edge': #active
                              if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or (pre_dstr_n_gr_tp == 'Bus' and \
                                u_n_gr_type == 'Bus') or (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                  continue#penalty = 1000000000000000
                             
#                           avoid paths that include two consecutive taxis or carsharign legs in one trip
                            if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                                prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                                (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                                 u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                    continue#penalty = 1000000000000000 active
                            
                            # restraint pick up -active
                            if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                                if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                                  continue#penalty = 1000000000000000
                              
                            # restraint drop off - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                                continue
                            
                            # restraint walking after taxi modes - active
                            if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                            and pr_ed_tp == 'walk_edge':
                                continue
                          
                        total_travel_time_till_u = e_tt + e_wait_time + info['opt_crt_val'][0]
                        cost_till_u = e_cost + info['opt_crt_val'][1]
                        # walk_till_u = e_walk_time + info['opt_crt_val'][2]
                        trips_till_u = e_trip_num + info['opt_crt_val'][2]
                        # wait_time_till_u = e_wait_time + info['wt']
                        # distance_till_u = e_distance + info['l']
                        # walk_time_till_u = e_walk_time + info['wkt']
                        
                        new_cost_label = (total_travel_time_till_u, cost_till_u, trips_till_u)
                        criteria_num = len(new_cost_label)
                        new_buck_cost_label = (total_travel_time_till_u - (total_travel_time_till_u % time_bucket), \
                                               cost_till_u - (cost_till_u % cost_bucket), trips_till_u)
                        
                        labels_to_be_deleted = deque([])
                        if not(labels_bag[u][t_2]):
                            non_dominated_label = 1
                        else:
                            for label, label_info in labels_bag[u][t_2].items():
                                # temp_pareto_cost_label = label_info['opt_crt_val']
                                temp_buck_pareto_cost_label = (label_info['opt_crt_val'][0] - (label_info['opt_crt_val'][0] % time_bucket), \
                                                               label_info['opt_crt_val'][1] - (label_info['opt_crt_val'][1] % cost_bucket), \
                                                                   label_info['opt_crt_val'][2])
                                if new_buck_cost_label == temp_buck_pareto_cost_label:
                                    non_dominated_label = 0
                                    break
                                q_1 = 0 
                                q_2 = 0
                                for i, j in zip(temp_buck_pareto_cost_label, new_buck_cost_label):
                                    if i<=j:
                                        q_1 += 1
                                    if i==j:
                                        q_2 += 1
                                if q_1 == criteria_num and q_2 != criteria_num:
                                    non_dominated_label = 0
                                    break
                                q_3 = 0
                                q_4 = 0
                                for i, j in zip(new_buck_cost_label, temp_buck_pareto_cost_label):
                                    if i<=j:
                                        q_3 += 1
                                    if i==j:
                                        q_4 += 1
                                if q_3 == criteria_num and q_4 != criteria_num:
                                    labels_to_be_deleted.append(label)
                                non_dominated_label = 1
                      
                        if non_dominated_label:
                            if labels_to_be_deleted:
                                for labelid in labels_to_be_deleted:                                      
                                    del(labels_bag[u][t_2][labelid])
                                    labels_to_extend[u][t_2].discard(labelid)
                            insert_in_se_list = True     
                            new_label_id = str(next(c))
                            labels_to_extend[u][t_2].add(new_label_id)
                            labels_bag[u][t_2].update({new_label_id: {'opt_crt_val' : new_cost_label, 'pred_node' : v, \
                                                                    'pred_time_int': v_arr_time, 'pred_label_id' : label_id, \
                                                                        'prev_edge_type': e_type, \
                                                                            'prev_dstr_node_graph_type': v_n_gr_type, \
                                                                                'prev_mode': prev_mode}})
                                                                                                      
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        
        for time in labels_to_extend[v]:
            labels_to_extend[v][time] = set()
    
    return labels_bag
        
def get_shortest_weighted_path(G, source, target, req_time, t_0, t_H, dt, walk_attrs_weights = [], \
                               bus_attrs_weights = [], train_attrs_weights = [], taxi_attrs_weights = [], \
                                   sms_attrs_weights = [], sms_pool_attrs_weights = [], cs_attrs_weights = [], \
                                       mode_transfer_weight = 0):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]
  
  discrete_request_time = req_time + (dt -(req_time%dt))
  discrete_t_0 = t_0 + (dt -(t_0%dt))
  discrete_t_H = t_H + (dt -(t_H%dt))

  path_data = single_crt_backw_alg(G, source, target, discrete_t_0, discrete_t_H, dt, walk_attrs_weights, \
                                   bus_attrs_weights, train_attrs_weights, taxi_attrs_weights, \
                                       sms_attrs_weights, sms_pool_attrs_weights, cs_attrs_weights, \
                                           mode_transfer_weight)
  
  paths_with_attrs = {}
  path = deque([source])
  if 'pred_node' not in path_data[source][discrete_request_time]:
      print(source, target)
      print(path_data)
  next_node = path_data[source][discrete_request_time]['pred_node']
  next_time_intrv = path_data[source][discrete_request_time]['pred_time_int']
  
  while next_node != None and next_time_intrv != None:
      path.append(next_node)
      new_node = path_data[next_node][next_time_intrv]['pred_node']
      new_time_intrv = path_data[next_node][next_time_intrv]['pred_time_int']
      next_node = new_node
      next_time_intrv = new_time_intrv
  
  paths_with_attrs = {'path': path, 'info': path_data[source][discrete_request_time]}
  
  return paths_with_attrs

def single_crt_backw_alg(G, source, target, t_0, t_H, dt, walk_attrs_weights, \
                                   bus_attrs_weights, train_attrs_weights, taxi_attrs_weights, \
                                       sms_attrs_weights, sms_pool_attrs_weights, cs_attrs_weights, \
                                           mode_transfer_weight):
    Gpred = G._pred
    
    labels_bag = dict()
    labels_to_extend = dict()
    
    for n in G:
        labels_bag.update({n: dict()})
        labels_to_extend.update({n: set()})
        for t in range(t_0, t_H+dt, dt):
            if t>= 86400:
                t_1 = t-86400 - ((t-86400)%dt)
            else:
                t_1 = t
            if n == target:
                labels_bag[n].update({t_1: {'weight': 0, 'pred_node': None, 'pred_time_int': None, \
                                            'prev_edge_type': None, 'prev_dstr_node_graph_type': None, \
                                                'prev_mode': None}})
                labels_to_extend[n].add(t_1)
            else:
                labels_bag[n].update({t_1: dict()})

    de_queue = dict()
    for n in G:
        if n == target:
            de_queue.update({n: 999999999})
        else:
            de_queue.update({n: 0})
    
    se_list = deque([target])
    
    while se_list: 
        v = se_list.popleft()
        de_queue[v] = -1        
        v_n_gr_type = G.nodes[v]['node_graph_type']
        
        for u, e in Gpred[v].items():
            insert_in_se_list = False

            for t in range(t_0, t_H+dt, dt):
                if t >= 86400:
                    t_2 = t-86400 - ((t-86400)%dt)
                else:
                    t_2 = t
                e_type = e['edge_type']

                if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time']
                    e_cost=0
                    e_distance = 0
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_orig_dummy_edge':
                    e_tt = 0
                    e_wait_time = 0
                    e_distance = 0
                    e_cost = 0
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = 0
                    e_distance = e['distance']
                    e_cost = e['car_sharing_fares'][t_2]
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                    
                if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
                    e_wait_time = e['taxi_wait_time'][t_2]
                    vehicle_boarding_time = t_2 + e_wait_time
                    if vehicle_boarding_time >= 86400:
                        vehicle_boarding_time = vehicle_boarding_time-86400 - ((vehicle_boarding_time-86400)%dt)
                    e_tt = e['travel_time'][(vehicle_boarding_time)-(vehicle_boarding_time%dt)]
                    e_cost = e['taxi_fares'][t_2]
                    e_distance = e['distance']
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                
                if e_type == 'walk_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    e_distance = e['distance']
                    e_cost = 0
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'access_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    e_distance = e['distance']
                    e_cost = 0
                    e_boarding_num = 0
                    u_n_type = e['up_node_type']
                    if u_n_type == 'walk_graph_node':
                        e_trip_num = 1
                    else:
                        e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_transfer_edge':
                    e_tt = e['travel_time']
                    e_wait_time = 0
                    e_distance = e['distance']
                    e_cost = 0
                    u_n_type = e['up_node_type']
                    v_n_type = e['dstr_node_type']
                    if (u_n_type == 'stop_node' or u_n_type=='station_node') and v_n_type == 'route_node':
                        e_boarding_num = 1
                    else:
                        e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = e['travel_time']
                    
                if e_type == 'pt_route_edge':
                    e_tt = e['travel_time'][t_2]
                    e_wait_time = e['wait_time'][t_2]['discr_value']
                    e_distance = e['distance']
                    e_cost = e['pt_cost'][t_2]
                    e_boarding_num = 0
                    e_trip_num = 0
                    # e_walk_time = 0
                         
                v_arr_time = t_2 + e_tt + e_wait_time
                if v_arr_time >= 86400:
                    v_arr_time = v_arr_time-86400 - ((v_arr_time-86400)%dt)
                rolled_v_arr_time = t + e_tt + e_wait_time

                if rolled_v_arr_time <= t_H:
                    if v_arr_time not in labels_to_extend[v]:
                        continue
                    if u == labels_bag[v][v_arr_time]['pred_node']:
                        continue
                    prev_mode = labels_bag[v][v_arr_time]['prev_mode']
                    pr_ed_tp = labels_bag[v][v_arr_time]['prev_edge_type']
                    pre_dstr_n_gr_tp = labels_bag[v][v_arr_time]['prev_dstr_node_graph_type']
                    
                    # restraint walking before taxi modes - active
                    if e_type == 'walk_edge' and pr_ed_tp == 'access_edge' and \
                    (pre_dstr_n_gr_tp == 'taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_single_taxi_graph' or pre_dstr_n_gr_tp == 'on_demand_shared_taxi_graph'):
                        continue
                    
                    if e_type == 'access_edge':
                        u_n_type = e['up_node_type']#G.nodes[v]['node_type']
                        u_n_gr_type = e['up_node_graph_type']#G.nodes[u]['node_graph_type']#node_graph_type_data(u, G.nodes[u])
                        v_n_type = e['dstr_node_type']#G.nodes[u]['node_type']#node_type_data(u, G.nodes[u])
                        
                        if u_n_type == 'car_sharing_station_node' and G.nodes[u]['stock_level'][t_2] == G.nodes[u]['capacity']:
                            e_tt = 999999999
                            e_wait_time = 999999999
                            e_cost= 999999999
                            e_distance = 999999999
                            e_boarding_num =  999999999
                            e_trip_num = 999999999
                            # e_walk_time = 0
                        elif v_n_type == 'car_sharing_station_node' and G.nodes[v]['stock_level'][v_arr_time] == 0:
                            e_tt = 999999999
                            e_wait_time = 999999999
                            e_cost= 999999999
                            e_distance = 999999999
                            e_boarding_num =  999999999
                            e_trip_num = 999999999
                            # e_walk_time = 0
                            
                        if u_n_gr_type == 'Walk':
                            prev_mode = v_n_gr_type
#                       when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip and back to taxi/carsharing trip
                        if pr_ed_tp == 'access_edge': #active
                            if (pre_dstr_n_gr_tp == 'Walk' and u_n_gr_type == 'Walk') or \
                                (pre_dstr_n_gr_tp == 'Bus' and u_n_gr_type == 'Bus') or \
                                    (pre_dstr_n_gr_tp == 'Train' and u_n_gr_type == 'Train'):
                                        continue
                         
#                       avoid paths that include two consecutive taxis or carsharign legs in one trip
                        if (prev_mode == 'taxi_graph' or prev_mode == 'on_demand_single_taxi_graph' or \
                            prev_mode == 'on_demand_shared_taxi_graph' or prev_mode == 'car_sharing_graph') and \
                            (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or \
                             u_n_gr_type == 'on_demand_shared_taxi_graph' or u_n_gr_type == 'car_sharing_graph'):
                                continue
                        
                        # restraint pick up -active
                        if v_n_gr_type == 'taxi_graph' or v_n_gr_type == 'on_demand_single_taxi_graph' or v_n_gr_type == 'on_demand_shared_taxi_graph':
                            if e['up_node_zone'] == G.nodes[source]['zone'] and u != source:
                              continue
                          
                        # restraint drop off - active
                        if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                        and e['dstr_node_zone'] == G.nodes[target]['zone'] and v != target:
                            continue
                        
                        # restraint walking after taxi modes - active
                        if (u_n_gr_type == 'taxi_graph' or u_n_gr_type == 'on_demand_single_taxi_graph' or u_n_gr_type == 'on_demand_shared_taxi_graph') \
                        and pr_ed_tp == 'walk_edge':
                            continue
                    
                    v_score = labels_bag[v][v_arr_time]['weight']
                    
                    if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge' or \
                    e_type == 'car_sharing_orig_dummy_edge' or e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
                        score = v_score + cs_attrs_weights[0] * e_tt + cs_attrs_weights[1] * e_wait_time + \
                        cs_attrs_weights[2] * e_distance + cs_attrs_weights[3] * e_cost + cs_attrs_weights[4] * e_boarding_num
                    elif e_type == 'taxi_edge':
                        score = v_score + taxi_attrs_weights[0] * e_tt + taxi_attrs_weights[1] * e_wait_time + \
                            taxi_attrs_weights[2] * e_distance + taxi_attrs_weights[3] * e_cost + taxi_attrs_weights[4] * e_boarding_num
                    elif e_type == 'on_demand_single_taxi_edge':
                        score = v_score + sms_attrs_weights[0] * e_tt + sms_attrs_weights[1] * e_wait_time + \
                        sms_attrs_weights[2] * e_distance + sms_attrs_weights[3] * e_cost + sms_attrs_weights[4] * e_boarding_num
                    elif e_type == 'on_demand_shared_taxi_edge':
                        score = v_score + sms_pool_attrs_weights[0] * e_tt + sms_pool_attrs_weights[1] * e_wait_time + \
                        sms_pool_attrs_weights[2] * e_distance + sms_pool_attrs_weights[3] * e_cost + sms_pool_attrs_weights[4] * e_boarding_num
                    elif e_type == 'walk_edge':
                        score = v_score + walk_attrs_weights[0] * e_tt + walk_attrs_weights[1] * e_wait_time + \
                        walk_attrs_weights[2] * e_distance + walk_attrs_weights[3] * e_cost + walk_attrs_weights[4] * e_boarding_num
                    elif e_type == 'access_edge':
                        score = v_score + walk_attrs_weights[0] * e_tt + walk_attrs_weights[1] * e_wait_time + \
                        walk_attrs_weights[2] * e_distance + walk_attrs_weights[3] * e_cost + walk_attrs_weights[4] * e_boarding_num +\
                            mode_transfer_weight * e_trip_num
                    elif e_type == 'pt_transfer_edge':
                        if v_n_gr_type == 'Bus':
                            score = v_score + walk_attrs_weights[0] * e_tt + walk_attrs_weights[2] * e_distance + bus_attrs_weights[4] * e_boarding_num
                        else:
                            score = v_score + walk_attrs_weights[0] * e_tt + walk_attrs_weights[2] * e_distance + train_attrs_weights[4] * e_boarding_num
                    elif e_type == 'pt_route_edge':
                        if v_n_gr_type == 'Bus':
                            score = v_score + bus_attrs_weights[0] * e_tt + bus_attrs_weights[1] * e_wait_time + \
                            bus_attrs_weights[2] * e_distance + bus_attrs_weights[3] * e_cost + bus_attrs_weights[4] * e_boarding_num
                        else:
                            score = v_score + train_attrs_weights[0] * e_tt + train_attrs_weights[1] * e_wait_time + \
                            train_attrs_weights[2] * e_distance + train_attrs_weights[3] * e_cost + train_attrs_weights[4] * e_boarding_num
        
                    if not(labels_bag[u][t_2]):
                        insert_in_se_list = True
                        labels_bag[u][t_2] = {'weight' : score, 'pred_node' : v, 'pred_time_int': v_arr_time, \
                                              'prev_edge_type': e_type, 'prev_dstr_node_graph_type': v_n_gr_type, \
                                                  'prev_mode': prev_mode}
                        labels_to_extend[u].add(t_2)
                        continue
                    
                    if labels_bag[u][t_2]['weight'] > score:
                        insert_in_se_list = True
                        labels_bag[u][t_2] = {'weight' : score, 'pred_node' : v, 'pred_time_int': v_arr_time, \
                                              'prev_edge_type': e_type, 'prev_dstr_node_graph_type': v_n_gr_type, \
                                                  'prev_mode': prev_mode}
                        labels_to_extend[u].add(t_2)
                else:
                    continue
            
            
            if insert_in_se_list:
                if de_queue[u] == 0:
                    if se_list:
                        de_queue[se_list[-1]] = u
                    de_queue[u] = 999999999
                    se_list.append(u)
                elif de_queue[u] == -1:
                    if se_list:
                        de_queue[u] = se_list[0]
                    else:
                        de_queue[u] = 999999999
                    se_list.appendleft(u)
        labels_to_extend[v] = set()
    try:
        return labels_bag
    except KeyError:
        raise nx.NetworkXNoPath("Node %s not reachable from %s" % (target, source))

