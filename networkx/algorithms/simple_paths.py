# -*- coding: utf-8 -*-
#    Copyright (C) 2012 by
#    Sergio Nery Simoes <sergionery@gmail.com>
#    All rights reserved.
#    BSD license.
import collections
import math
import bisect
from heapq import heappush, heappop
from itertools import count, islice
from networkx.algorithms.shortest_paths.weighted import LY_shortest_path_with_attrs, _LY_dijkstra
from bisect import bisect_left
import networkx as nx
from networkx.utils import not_implemented_for
from networkx.utils import pairwise
from operator import itemgetter

from collections import deque
import math
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


# def lookup_ip(ip, data):
#     sorted_data = sorted(data, key=itemgetter(0))
#     lower_bounds = [lower for lower, _ in data]
#     index = bisect.bisect(lower_bounds, ip) - 1
#     if index < 0:
#         return None
#     # _, upper = sorted_data[index]
#     return(data[index])
#     return index if ip <= upper else None


def find_ge(a, x):  # binary search algorithm #'Find leftmost item greater than or equal to x'
  i = bisect_left(a, x)
  if i != len(a):
    return i, a[i]
  raise ValueError


def calc_time_dep_distance_based_cost(dist_based_cost_data={}, current_time=0):
  current_time = current_time % 86399
  # td_cost_data = list(dist_based_cost_data.keys())
  # # sorted_data = sorted(data, key=itemgetter(0))
  # lower_bounds = [lower for lower, _ in td_cost_data]
  # index = bisect.bisect(lower_bounds, current_time) - 1
  # if index < 0:
  #     return None
  # # _, upper = sorted_data[index]
  # time_interval = td_cost_data[index]
  # return dist_based_cost_data[time_interval]
  #   # return index if ip <= upper else None
  for time_intrvl in dist_based_cost_data:
    if int(current_time) >= int(time_intrvl[0]) and int(current_time) <= int(time_intrvl[1]):
      cost = dist_based_cost_data[time_intrvl]
      break
    if cost is None:
      print('No cost from zone_to_zone data found')
    return cost

def calc_time_dep_zone_to_zone_cost(zn_to_zn_cost_data, current_time, pt_trip_start_zone, u_zone):
  current_time = (current_time-86399)%86399
  if not pt_trip_start_zone:
    print('Zone value for the start of pt trip has not assigned')
    pass
  else:
    for time_intrvl in zn_to_zn_cost_data:
      if int(current_time) >= int(time_intrvl[0]) and int(current_time) <= int(time_intrvl[1]):
        cost = zn_to_zn_cost_data[time_intrvl][(pt_trip_start_zone, u_zone)]
        break
    if cost is None:
      print('No cost from zone_to_zone data found')
    return cost


def calc_plat_wait_time_and_train_id(arr_time=0, edge_departure_time={}):
  # the sorted list generation might need to go outside the algorithm as it increases computational performance
  list_of_ptedge_dep_times = list(edge_departure_time.values()) # removed sorted, assume FIFO
  # for run_id in edge_departure_time:
  #   list_of_ptedge_dep_times.append(edge_departure_time[run_id])
  # list_of_ptedge_dep_times.sort()
  arr_time = arr_time % list_of_ptedge_dep_times[-1]
  index, earlier_dep_time = find_ge(list_of_ptedge_dep_times, arr_time)
  platform_wait_time = earlier_dep_time - arr_time
  vehicle_id = list(edge_departure_time.keys())[index]
  # for run_id in edge_departure_time:
  #   if earlier_dep_time == edge_departure_time[run_id]:
  #     corresponding_runid_to_earl_dep_time = run_id
  #     break
  return platform_wait_time, vehicle_id

def get_time_dep_taxi_cost(current_time=0, taxi_cost_data={}):
  current_time = current_time%86399
  index = int(current_time/300) # hardcoded, will work for 5min interval data
  if index == 1:
    time_intrv = list(taxi_cost_data.keys())[0]
  else:
    time_intrv = list(taxi_cost_data.keys())[index]
  return taxi_cost_data[time_intrv]

  # td_cost_data = list(taxi_cost_data.keys())
  # # sorted_data = sorted(data, key=itemgetter(0))
  # lower_bounds = [lower for lower, _ in td_cost_data]
  # index = bisect.bisect(lower_bounds, current_time) - 1
  # if index < 0:
  #     return None
  # # _, upper = sorted_data[index]
  # time_interval = td_cost_data[index]
  # return taxi_cost_data[time_interval]
  # current_time = (current_time-86399)%86399
  # for time_intrvl, cost in taxi_cost_data.items():
  #   if int(current_time) >= int(time_intrvl[0]) and int(current_time) <= int(time_intrvl[1]):
  #     edge_cost = cost
  #     break
  # return edge_cost

def get_time_dep_taxi_wait_time(current_time=0, taxi_wait_time_data={}):
  current_time = current_time%86399
  index = int(current_time/300) # hardcoded, will work for 5min interval data
  if index == 1:
    time_intrv = list(taxi_wait_time_data.keys())[0]
  else:
    time_intrv = list(taxi_wait_time_data.keys())[index]
  return taxi_wait_time_data[time_intrv]
  # current_time = current_time%86399
  # td_cost_data = list(taxi_wait_time_data.keys())
  # # sorted_data = sorted(data, key=itemgetter(0))
  # lower_bounds = [lower for lower, _ in td_cost_data]
  # index = bisect.bisect(lower_bounds, current_time) - 1
  # if index < 0:
  #     return None
  # # _, upper = sorted_data[index]
  # time_interval = td_cost_data[index]
  # return taxi_wait_time_data[time_interval]
  # current_time = (current_time-86399)%86399
  # for time_intrvl, wait_time in taxi_wait_time_data.items():
  #   if int(current_time) >= int(time_intrvl[0]) and int(current_time) <= int(time_intrvl[1]):
  #     edge_wt_t = wait_time
  #     break
  # return edge_wt_t

def get_time_dep_taxi_travel_time(current_time=0, taxi_travel_time_data={}):
  current_time = current_time%86399
  index = int(current_time/300) # hardcoded, will work for 5min interval data
  if index == 1:
    time_intrv = list(taxi_travel_time_data.keys())[0]
  else:
    time_intrv = list(taxi_travel_time_data.keys())[index]
  return taxi_travel_time_data[time_intrv]
  # current_time = current_time%86399
  # td_cost_data = list(taxi_travel_time_data.keys())
  # # sorted_data = sorted(data, key=itemgetter(0))
  # lower_bounds = [lower for lower, _ in td_cost_data]
  # index = bisect.bisect(lower_bounds, current_time) - 1
  # if index < 0:
  #     return None
  # # _, upper = sorted_data[index]
  # time_interval = td_cost_data[index]
  # return taxi_travel_time_data[time_interval]
  # current_time = (current_time-86399)%86399
  # for time_intrvl, travel_time in taxi_travel_time_data.items():
  #   if int(current_time) >= int(time_intrvl[0]) and int(current_time) <= int(time_intrvl[1]):
  #     edge_tt = travel_time
  #     break
  # return edge_tt



# def calc_pt_route_edge_in_veh_tt_for_run_id(edge_travel_times, run_id_of_first_dep_ptveh):
#   return edge_travel_times[]


def calc_road_link_tt(cur_time, edge_attrs):  # calculate the proper travel time for the respective 5-min interval based on current time in the network
  cur_time = (cur_time-86399)%86399
  tt = None
  for key, value in edge_attrs['weight'].items():
    if cur_time >= int(key[0]) and cur_time <= int(key[1]):  # interval time data needs to be in seconds and hence integers not strings
      tt = math.ceil(value)
  if tt == None:
    print('Current travel time could not be matched with 5min interval')
  else:
    return tt


def _get_timetable(G, departure_time):  # this will work only for directed graphs and the timetable will be a list of departure times
  return lambda u, v, data: data.get(departure_time, None)

def _get_edge_type(G, edge_type):
  return lambda u, v, data: data.get(edge_type, None)

def _get_node_type(G, node_type):
  return lambda u, data: data.get(node_type, None)

def _get_dwnstr_graph_type_data(v, node_graph_type):
    return lambda u, data: data.get(node_graph_type, None)

def _get_travel_time_function(G, travel_time):
  return lambda u, v, data: data.get(travel_time, None)

def _get_distance_function(G, distance):
  return lambda u, v, data: data.get(distance, None)

def _get_pt_additive_cost(G, pt_additive_cost):
  return lambda u, v, data: data.get(pt_additive_cost, None)

def _get_taxi_fares(G, taxi_fares):
  return lambda u, v, data: data.get(taxi_fares, None)


def _get_pt_non_additive_cost(G, pt_non_additive_cost):
  return lambda u, v, data: data.get(pt_non_additive_cost, None)


def _get_taxi_wait_time(G, taxi_wait_time):
  return lambda u, v, data: data.get(taxi_wait_time, None)


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

        >>> from networkx.utils import pairwise
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


def k_shortest_paths_LY(G, source, target, time_of_request, k, travel_time='travel_time', distance='distance', pt_additive_cost='pt_distance_based_cost', pt_non_additive_cost='pt_zone_to_zone_cost', taxi_fares='taxi_fares', taxi_wait_time='taxi_wait_time', timetable='departure_time', edge_type='edge_type', node_type='node_type', node_graph_type='node_graph_type', fare_scheme='distance_based', walk_attrs_w=[], bus_attrs_w=[], train_attrs_w=[], taxi_attrs_w=[], sms_attrs_w=[], sms_pool_attrs_w=[], cs_attrs_w=[], mode_transfer_weight=0):

  return(shortest_simple_paths_LY(G, source, target, time_of_request, k, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, node_graph_type=node_graph_type, fare_scheme=fare_scheme, walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, cs_attrs_w=cs_attrs_w, mode_transfer_weight=mode_transfer_weight))


    # return islice(shortest_simple_paths_LY(G, source, target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, node_graph_type=node_graph_type, fare_scheme=fare_scheme, walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, mode_transfer_weight=mode_transfer_weight), k)
    # list(islice(shortest_simple_paths_LY(G, source, target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, node_graph_type=node_graph_type, fare_scheme=fare_scheme, walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, mode_transfer_weight=mode_transfer_weight), k))


# @not_implemented_for('multigraph')
def shortest_simple_paths_LY(G, source, target, time_of_request, k, travel_time=None, distance=None, pt_additive_cost=None, pt_non_additive_cost=None, taxi_fares=None, taxi_wait_time=None, timetable=None, edge_type=None, node_type=None, node_graph_type=None, fare_scheme=None, walk_attrs_w=[], bus_attrs_w=[], train_attrs_w=[], taxi_attrs_w=[], sms_attrs_w=[], sms_pool_attrs_w=[], cs_attrs_w=[], mode_transfer_weight=0):
  if source not in G:
    raise nx.NodeNotFound('source node %s not in graph' % source)

  if target not in G:
    raise nx.NodeNotFound('target node %s not in graph' % target)

  path_num = 0
  push = heappush
  pop = heappop
  c = count()
  shortest_path_func = LY_shortest_path_with_attrs

  listA = list()
  listB = list()
  kpaths_dict = {}
  prev_path = None

  prev_path_node_travel_time_data = {}
  prev_path_node_wait_time_data = {}
  prev_path_node_dist_data = {}
  prev_path_node_cost_data = {}
  prev_path_node_line_trfs_data = {}
  prev_path_node_mode_trfs_data = {}
  prev_path_node_weight_data = {}
  prev_path_node_prev_edge_type_data = {}
  prev_path_node_prev_graph_type_data = {}
  prev_path_node_last_pt_veh_run_id_data = {}
  prev_path_node_current_time_data = {}
  prev_path_node_last_edge_cost_data = {}
  prev_path_node_pt_trip_start_zone_data = {}
  prev_path_previous_edge_mode_data = {}

  while path_num<=k-1:
#    if path_num == 7:
#        print('oops')
    if not prev_path:
      best_weight, best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels = shortest_path_func(G, source, target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, node_graph_type=node_graph_type, fare_scheme=fare_scheme, ignore_nodes=None, ignore_edges=None, current_time=time_of_request, walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, cs_attrs_w=cs_attrs_w, mode_transfer_weight=mode_transfer_weight, paths=None, orig_source=source)  #shortest_path_nodes_seq_data

      push(listB, (best_weight, next(c), best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels, None))  #shortest_path_nodes_seq_data, None

    else:
      ignore_nodes = set()
      ignore_edges = set()
      kmin1path_node_travel_time_data = {}
      kmin1path_node_travel_time_data.update(prev_path_node_travel_time_data)
      kmin1path_node_wait_time_data = {}
      kmin1path_node_wait_time_data.update(prev_path_node_wait_time_data)
      kmin1path_node_dist_data = {}
      kmin1path_node_dist_data.update(prev_path_node_dist_data)
      kmin1path_node_cost_data = {}
      kmin1path_node_cost_data.update(prev_path_node_cost_data)
      kmin1path_node_line_trfs_data = {}
      kmin1path_node_line_trfs_data.update(prev_path_node_line_trfs_data)
      kmin1path_node_mode_trfs_data = {}
      kmin1path_node_mode_trfs_data.update(prev_path_node_mode_trfs_data)
      kmin1path_node_weight_data = {}
      kmin1path_node_weight_data.update(prev_path_node_weight_data)
      kmin1path_node_prev_edge_type_data = {}
      kmin1path_node_prev_edge_type_data.update(prev_path_node_prev_edge_type_data)
      kmin1path_node_prev_graph_type_data = {}
      kmin1path_node_prev_graph_type_data.update(prev_path_node_prev_graph_type_data)
      kmin1path_node_last_pt_veh_run_id_data = {}
      kmin1path_node_last_pt_veh_run_id_data.update(prev_path_node_last_pt_veh_run_id_data)
      kmin1path_node_current_time_data = {}
      kmin1path_node_current_time_data.update(prev_path_node_current_time_data)
      kmin1path_node_last_edge_cost_data = {}
      kmin1path_node_last_edge_cost_data.update(prev_path_node_last_edge_cost_data)
      kmin1path_node_pt_trip_start_zone_data = {}
      kmin1path_node_pt_trip_start_zone_data.update(prev_path_node_pt_trip_start_zone_data)
      kmin1path_previous_edge_mode_data = {}
      kmin1path_previous_edge_mode_data.update(prev_path_previous_edge_mode_data)
      for i in range(1, len(prev_path)):
        root = prev_path[:i]
        for path in listA:
          if path[:i] == root:
            ignore_edges.add((path[i - 1], path[i]))
        try:
          best_weight, best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels = shortest_path_func(G, root[-1], target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, fare_scheme=fare_scheme, ignore_nodes=ignore_nodes, ignore_edges=ignore_edges, init_weight=kmin1path_node_weight_data[root[-1]], init_travel_time=kmin1path_node_travel_time_data[root[-1]], init_wait_time=kmin1path_node_wait_time_data[root[-1]], init_distance=kmin1path_node_dist_data[root[-1]], init_cost=kmin1path_node_cost_data[root[-1]], init_num_line_trfs=kmin1path_node_line_trfs_data[root[-1]], init_num_mode_trfs=kmin1path_node_mode_trfs_data[root[-1]], last_edge_type=kmin1path_node_prev_edge_type_data[root[-1]], last_upstr_node_graph_type=kmin1path_node_prev_graph_type_data[root[-1]], last_pt_veh_run_id=kmin1path_node_last_pt_veh_run_id_data[root[-1]], current_time=kmin1path_node_current_time_data[root[-1]], last_edge_cost=kmin1path_node_last_edge_cost_data[root[-1]], pt_trip_orig_zone=kmin1path_node_pt_trip_start_zone_data[root[-1]], previous_edge_mode=kmin1path_previous_edge_mode_data[root[-1]], walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, cs_attrs_w=cs_attrs_w, mode_transfer_weight=mode_transfer_weight, pred=None, paths=None, orig_source=source)  #shortest_path_nodes_seq_data # need to change the way this function calculates the shortes path and the weight function it considers. we need elements from thr last node of the route path
          test = 0
          for entry in listB:
            if best_path in entry:
              test += 1
          if test == 0:
            push(listB, (best_weight, next(c), best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels, root, kmin1path_node_travel_time_data, kmin1path_node_wait_time_data, kmin1path_node_dist_data, kmin1path_node_cost_data, kmin1path_node_line_trfs_data, kmin1path_node_mode_trfs_data, kmin1path_node_weight_data, kmin1path_node_prev_edge_type_data, kmin1path_node_prev_graph_type_data, kmin1path_node_last_pt_veh_run_id_data, kmin1path_node_current_time_data, kmin1path_node_last_edge_cost_data, kmin1path_node_pt_trip_start_zone_data, kmin1path_previous_edge_mode_data)) #shortest_path_nodes_seq_data
          # print(listB[0][2])

        except nx.NetworkXNoPath:
          pass
        ignore_nodes.add(root[-1])

    if listB:
      if path_num == 0:
        (prev_best_w, _, prev_best_p, prev_best_tt, prev_best_wait_t, prev_best_dist, prev_best_cost, prev_best_n_line_trfs, prev_best_n_mode_trfs, prev_path_tt_d, prev_path_wtt_d, prev_path_dist_d, prev_path_cost_d, prev_path_n_line_trfs_d, prev_path_n_mode_trfs_d, prev_path_weight_d, prev_previous_e_tp_d, prev_previous_upstr_n_gt_d, prev_l_pt_veh_run_id_d, prev_path_node_curr_t_d, prev_prev_e_cost_d, prev_pt_tr_start_zone_d, prev_edge_md_d, prev_root) = pop(listB)

        prev_path_node_travel_time_data.update(prev_path_tt_d)# = prev_path_in_v_tt_d
        prev_path_node_wait_time_data.update(prev_path_wtt_d)# = prev_path_wait_t_d
        prev_path_node_dist_data.update(prev_path_dist_d)# = prev_path_dist_d
        prev_path_node_cost_data.update(prev_path_cost_d)# = prev_path_cost_d
        prev_path_node_line_trfs_data.update(prev_path_n_line_trfs_d)# = prev_path_n_line_trfs_d
        prev_path_node_mode_trfs_data.update(prev_path_n_mode_trfs_d)# = prev_path_n_mode_trfs_d
        prev_path_node_weight_data.update(prev_path_weight_d)# = prev_path_weight_d
        prev_path_node_prev_edge_type_data.update(prev_previous_e_tp_d)# = prev_previous_e_tp_d
        prev_path_node_prev_graph_type_data.update(prev_previous_upstr_n_gt_d)
        prev_path_node_last_pt_veh_run_id_data.update(prev_l_pt_veh_run_id_d)# = prev_l_pt_veh_run_id_d
        prev_path_node_current_time_data.update(prev_path_node_curr_t_d)# = prev_path_node_curr_t_d
        prev_path_node_last_edge_cost_data.update(prev_prev_e_cost_d)# = prev_prev_e_cost_d
        prev_path_node_pt_trip_start_zone_data.update(prev_pt_tr_start_zone_d)# = prev_pt_tr_start_zone_d
        prev_path_previous_edge_mode_data.update(prev_edge_md_d)
      else:
        (prev_best_w, _, prev_best_p, prev_best_tt, prev_best_wait_t, prev_best_dist, prev_best_cost, prev_best_n_line_trfs, prev_best_n_mode_trfs, prev_path_tt_d, prev_path_wtt_d, prev_path_dist_d, prev_path_cost_d, prev_path_n_line_trfs_d, prev_path_n_mode_trfs_d, prev_path_weight_d, prev_previous_e_tp_d, prev_previous_upstr_n_gt_d, prev_l_pt_veh_run_id_d, prev_path_node_curr_t_d, prev_prev_e_cost_d, prev_pt_tr_start_zone_d, prev_edge_md_d, prev_root, new_kmin1path_node_travel_time_data, new_kmin1path_node_wait_time_data, new_kmin1path_node_dist_data, new_kmin1path_node_cost_data, new_kmin1path_node_line_trfs_data, new_kmin1path_node_mode_trfs_data, new_kmin1path_node_weight_data, new_kmin1path_node_prev_edge_type_data, new_kmin1path_node_prev_graph_type_data, new_kmin1path_node_last_pt_veh_run_id_data, new_kmin1path_node_current_time_data, new_kmin1path_node_last_edge_cost_data, new_kmin1path_node_pt_trip_start_zone_data, new_kmin1path_previous_edge_mode_data) = pop(listB)

        prev_path_node_travel_time_data.update(new_kmin1path_node_travel_time_data)
        prev_path_node_travel_time_data.update(prev_path_tt_d)
        prev_path_node_wait_time_data.update(new_kmin1path_node_wait_time_data)
        prev_path_node_wait_time_data.update(prev_path_wtt_d)# = prev_path_wait_t_d
        prev_path_node_dist_data.update(new_kmin1path_node_dist_data)# = prev_path_dist_d
        prev_path_node_dist_data.update(prev_path_dist_d)
        prev_path_node_cost_data.update(new_kmin1path_node_cost_data)# = prev_path_cost_d
        prev_path_node_cost_data.update(prev_path_cost_d)
        prev_path_node_line_trfs_data.update(new_kmin1path_node_line_trfs_data)# = prev_path_n_line_trfs_d
        prev_path_node_line_trfs_data.update(prev_path_n_line_trfs_d)
        prev_path_node_mode_trfs_data.update(new_kmin1path_node_mode_trfs_data)# = prev_path_n_mode_trfs_d
        prev_path_node_mode_trfs_data.update(prev_path_n_mode_trfs_d)
        prev_path_node_weight_data.update(new_kmin1path_node_weight_data)# = prev_path_weight_d
        prev_path_node_weight_data.update(prev_path_weight_d)
        prev_path_node_prev_edge_type_data.update(new_kmin1path_node_prev_edge_type_data)# = prev_previous_e_tp_d
        prev_path_node_prev_edge_type_data.update(prev_previous_e_tp_d)
        prev_path_node_prev_graph_type_data.update(new_kmin1path_node_prev_graph_type_data)
        prev_path_node_prev_graph_type_data.update(prev_previous_upstr_n_gt_d)
        prev_path_node_last_pt_veh_run_id_data.update(new_kmin1path_node_last_pt_veh_run_id_data)# = prev_l_pt_veh_run_id_d
        prev_path_node_last_pt_veh_run_id_data.update(prev_l_pt_veh_run_id_d)
        prev_path_node_current_time_data.update(new_kmin1path_node_current_time_data)# = prev_path_node_curr_t_d
        prev_path_node_current_time_data.update(prev_path_node_curr_t_d)
        prev_path_node_last_edge_cost_data.update(new_kmin1path_node_last_edge_cost_data)# = prev_prev_e_cost_d
        prev_path_node_last_edge_cost_data.update(prev_prev_e_cost_d)
        prev_path_node_pt_trip_start_zone_data.update(new_kmin1path_node_pt_trip_start_zone_data)# = prev_pt_tr_start_zone_d
        prev_path_node_pt_trip_start_zone_data.update(prev_pt_tr_start_zone_d)
        prev_path_previous_edge_mode_data.update(new_kmin1path_previous_edge_mode_data)
        prev_path_previous_edge_mode_data.update(prev_edge_md_d)

        prev_best_p[:0]=prev_root[:-1]

      listA.append(prev_best_p)
      prev_path = prev_best_p
      kpaths_dict.update({str(prev_path): [prev_best_w, prev_best_tt, prev_best_wait_t, prev_best_dist, prev_best_cost, prev_best_n_line_trfs, prev_best_n_mode_trfs]})

      path_num += 1
    else:
      break
  return(kpaths_dict)


# def shortest_simple_paths_LY(G, source, target, time_of_request, k, travel_time=None, distance=None, pt_additive_cost=None, pt_non_additive_cost=None, taxi_fares=None, taxi_wait_time=None, timetable=None, edge_type=None, node_type=None, node_graph_type=None, fare_scheme=None, walk_attrs_w=[], bus_attrs_w=[], train_attrs_w=[], taxi_attrs_w=[], sms_attrs_w=[], sms_pool_attrs_w=[], cs_attrs_w=[], mode_transfer_weight=0):
#   if source not in G:
#     raise nx.NodeNotFound('source node %s not in graph' % source)

#   if target not in G:
#     raise nx.NodeNotFound('target node %s not in graph' % target)

#   path_num = 0
#   push = heappush
#   pop = heappop
#   c = count()
#   shortest_path_func = LY_shortest_path_with_attrs

#   listA = list()
#   listB = list()
#   kpaths_dict = {}
#   prev_path = None

#   kmin1path_node_travel_time_data = {}
#   kmin1path_node_wait_time_data = {}
#   kmin1path_node_dist_data = {}
#   kmin1path_node_cost_data = {}
#   kmin1path_node_line_trfs_data = {}
#   kmin1path_node_mode_trfs_data = {}
#   kmin1path_node_weight_data = {}
#   kmin1path_node_prev_edge_type_data = {}
#   kmin1path_node_prev_graph_type_data = {}
#   kmin1path_node_last_pt_veh_run_id_data = {}
#   kmin1path_node_current_time_data = {}
#   kmin1path_node_last_edge_cost_data = {}
#   kmin1path_node_pt_trip_start_zone_data = {}
#   kmin1path_previous_edge_mode_data = {}

#   while path_num<=k-1:
#     if not prev_path:
#       best_weight, best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels = shortest_path_func(G, source, target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, node_graph_type=node_graph_type, fare_scheme=fare_scheme, ignore_nodes=None, ignore_edges=None, current_time=time_of_request, walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, cs_attrs_w=cs_attrs_w, mode_transfer_weight=mode_transfer_weight, paths=None, orig_source=source)  #shortest_path_nodes_seq_data

#       push(listB, (best_weight, next(c), best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels, None))  #shortest_path_nodes_seq_data, None

#     else:
#       ignore_nodes = set()
#       ignore_edges = set()
#       for i in range(1, len(prev_path)):
#         root = prev_path[:i]
#         for path in listA:
#           if path[:i] == root:
#             ignore_edges.add((path[i - 1], path[i]))
#         try:
#           best_weight, best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels = shortest_path_func(G, root[-1], target, time_of_request, travel_time=travel_time, distance=distance, pt_additive_cost=pt_additive_cost, pt_non_additive_cost=pt_non_additive_cost, taxi_fares=taxi_fares, taxi_wait_time=taxi_wait_time, timetable=timetable, edge_type=edge_type, node_type=node_type, fare_scheme=fare_scheme, ignore_nodes=ignore_nodes, ignore_edges=ignore_edges, init_weight=kmin1path_node_weight_data[root[-1]], init_travel_time=kmin1path_node_travel_time_data[root[-1]], init_wait_time=kmin1path_node_wait_time_data[root[-1]], init_distance=kmin1path_node_dist_data[root[-1]], init_cost=kmin1path_node_cost_data[root[-1]], init_num_line_trfs=kmin1path_node_line_trfs_data[root[-1]], init_num_mode_trfs=kmin1path_node_mode_trfs_data[root[-1]], last_edge_type=kmin1path_node_prev_edge_type_data[root[-1]], last_upstr_node_graph_type=kmin1path_node_prev_graph_type_data[root[-1]], last_pt_veh_run_id=kmin1path_node_last_pt_veh_run_id_data[root[-1]], current_time=kmin1path_node_current_time_data[root[-1]], last_edge_cost=kmin1path_node_last_edge_cost_data[root[-1]], pt_trip_orig_zone=kmin1path_node_pt_trip_start_zone_data[root[-1]], previous_edge_mode=kmin1path_previous_edge_mode_data[root[-1]], walk_attrs_w=walk_attrs_w, bus_attrs_w=bus_attrs_w, train_attrs_w=train_attrs_w, taxi_attrs_w=taxi_attrs_w, sms_attrs_w=sms_attrs_w, sms_pool_attrs_w=sms_pool_attrs_w, cs_attrs_w=cs_attrs_w, mode_transfer_weight=mode_transfer_weight, pred=None, paths=None, orig_source=source)  #shortest_path_nodes_seq_data # need to change the way this function calculates the shortes path and the weight function it considers. we need elements from thr last node of the route path
#           test = 0
#           for entry in listB:
#             if best_path in entry:
#               test += 1
#           if test == 0:
#             push(listB, (best_weight, next(c), best_path, best_tt, best_wtt, best_dist, best_cost, best_num_line_transfers, best_num_mode_transfers, path_tt_data, path_wtt_data, path_dist_data, path_cost_data, path_line_trf_data, path_mode_trf_data, path_weight_labels, previous_edge_type_labels, previous_upstr_node_graph_type_labels, last_pt_vehicle_run_id_labels, current_time_labels, previous_edge_cost_labels, pt_trip_start_zone_labels, previous_edge_mode_labels, root)) #shortest_path_nodes_seq_data
#           # print(listB[0][2])

#         except nx.NetworkXNoPath:
#           pass
#         ignore_nodes.add(root[-1])

#     if listB:
#       (prev_best_w, _, prev_best_p, prev_best_tt, prev_best_wait_t, prev_best_dist, prev_best_cost, prev_best_n_line_trfs, prev_best_n_mode_trfs, prev_path_tt_d, prev_path_wtt_d, prev_path_dist_d, prev_path_cost_d, prev_path_n_line_trfs_d, prev_path_n_mode_trfs_d, prev_path_weight_d, prev_previous_e_tp_d, prev_previous_upstr_n_gt_d, prev_l_pt_veh_run_id_d, prev_path_node_curr_t_d, prev_prev_e_cost_d, prev_pt_tr_start_zone_d, prev_edge_md_d, root) = pop(listB) #prev_s_path_node_seq_d
#         # path = listB.pop()
#       if path_num != 0:
#         prev_best_p[:0]=root[:-1]
#         # yield prev_best_p
#       listA.append(prev_best_p)
#       prev_path = prev_best_p
#       kpaths_dict.update({str(prev_path): [prev_best_w, prev_best_tt, prev_best_wait_t, prev_best_dist, prev_best_cost, prev_best_n_line_trfs, prev_best_n_mode_trfs]})
#       kmin1path_node_travel_time_data.update(prev_path_tt_d)# = prev_path_in_v_tt_d
#       kmin1path_node_wait_time_data.update(prev_path_wtt_d)# = prev_path_wait_t_d
#       kmin1path_node_dist_data.update(prev_path_dist_d)# = prev_path_dist_d
#       kmin1path_node_cost_data.update(prev_path_cost_d)# = prev_path_cost_d
#       kmin1path_node_line_trfs_data.update(prev_path_n_line_trfs_d)# = prev_path_n_line_trfs_d
#       kmin1path_node_mode_trfs_data.update(prev_path_n_mode_trfs_d)# = prev_path_n_mode_trfs_d
#       kmin1path_node_weight_data.update(prev_path_weight_d)# = prev_path_weight_d
#       kmin1path_node_prev_edge_type_data.update(prev_previous_e_tp_d)# = prev_previous_e_tp_d
#       kmin1path_node_prev_graph_type_data.update(prev_previous_upstr_n_gt_d)
#       kmin1path_node_last_pt_veh_run_id_data.update(prev_l_pt_veh_run_id_d)# = prev_l_pt_veh_run_id_d
#       kmin1path_node_current_time_data.update(prev_path_node_curr_t_d)# = prev_path_node_curr_t_d
#       kmin1path_node_last_edge_cost_data.update(prev_prev_e_cost_d)# = prev_prev_e_cost_d
#       kmin1path_node_pt_trip_start_zone_data.update(prev_pt_tr_start_zone_d)# = prev_pt_tr_start_zone_d
#       kmin1path_previous_edge_mode_data.update(prev_edge_md_d)
#       path_num += 1
#       if path_num == 11:
#         print('stop')
#     else:
#         break
#   return(kpaths_dict)


def LY_shortest_path_with_attrs(G, source, target, time_of_request, travel_time='travel_time', distance='distance', pt_additive_cost='pt_distance_based_cost', pt_non_additive_cost='pt_zone_to_zone_cost', taxi_fares='taxi_fares', taxi_wait_time='taxi_wait_time', timetable='departure_time', edge_type='edge_type', node_type='node_type', node_graph_type='node_graph_type', fare_scheme='distance_based', ignore_nodes=None, ignore_edges=None, init_weight=0, init_travel_time = 0, init_wait_time = 0, init_distance = 0, init_cost = 0, init_num_line_trfs = 0, init_num_mode_trfs = 0, last_edge_type=None, last_upstr_node_graph_type=None, last_pt_veh_run_id=None, current_time=0, last_edge_cost=0, pt_trip_orig_zone=None, previous_edge_mode=None, walk_attrs_w=[], bus_attrs_w=[], train_attrs_w=[], taxi_attrs_w=[], sms_attrs_w=[], sms_pool_attrs_w=[], cs_attrs_w=[], mode_transfer_weight=0, pred=None, paths=None, orig_source=None):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]

  # if paths == None:
  paths = {source: [source]}
  # else:
    # paths = paths
  travel_time = _get_travel_time_function(G, travel_time)
  distance = _get_distance_function(G, distance)
  pt_additive_cost = _get_pt_additive_cost(G, pt_additive_cost)
  pt_non_additive_cost = _get_pt_non_additive_cost(G, pt_non_additive_cost)
  timetable = _get_timetable(G, timetable)
  edge_type = _get_edge_type(G, edge_type)
  node_type = _get_node_type(G, node_type)
  node_graph_type = _get_dwnstr_graph_type_data(G, node_graph_type)
  taxi_fares = _get_taxi_fares(G, taxi_fares)
  taxi_wait_time = _get_taxi_wait_time(G, taxi_wait_time)


  return _LY_dijkstra(G, source, target, time_of_request, travel_time, distance, pt_additive_cost, pt_non_additive_cost, taxi_fares, taxi_wait_time, timetable, edge_type, node_type, node_graph_type, fare_scheme, ignore_nodes, ignore_edges, init_weight, init_travel_time, init_wait_time, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, last_edge_type, last_upstr_node_graph_type, last_pt_veh_run_id, current_time, last_edge_cost, pt_trip_orig_zone, previous_edge_mode, walk_attrs_w, bus_attrs_w, train_attrs_w, taxi_attrs_w, sms_attrs_w, sms_pool_attrs_w, cs_attrs_w, mode_transfer_weight, orig_source, pred=None, paths=paths)



def _LY_dijkstra(G, source, target, time_of_request, travel_time_data, distance_data, pt_additive_cost_data, pt_non_additive_cost_data, taxi_fares, taxi_wait_time, timetable_data, edge_type_data, node_type_data, node_graph_type_data, fare_scheme, ignore_nodes, ignore_edges, init_weight, init_travel_time, init_wait_time, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, last_edge_type, last_upstr_node_graph_type, last_pt_veh_run_id, current_time, last_edge_cost, pt_trip_orig_zone, previous_edge_mode, walk_attrs_w, bus_attrs_w, train_attrs_w, taxi_attrs_w, sms_attrs_w, sms_pool_attrs_w, cs_attrs_w, mode_transfer_weight, orig_source, pred=None, paths=None):

  # handles only directed
    Gsucc = G.successors

    # support optional nodes filter
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate

        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w
            return iterate

        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop


    weight_label = {}  # dictionary of best weight
    travel_time_label = {}
    wait_time_label = {}
    distance_label = {}
    mon_cost_label = {}
    line_trans_num_label = {}
    mode_trans_num_label = {}
    prev_edge_type_label = {}
    prev_upstr_node_graph_type_label = {}
    last_pt_veh_run_id_label = {}
    current_time_label = {}
    prev_edge_cost_label = {}
    pt_trip_start_zone_label = {}
    prev_mode_label = {}
    seen = {}


    c = count()   # use the count c to avoid comparing nodes (may not be able to)
    fringe = []  # fringe is heapq with 3-tuples (distance,c,node) -and I change it to a 4-tuple to store the type of the previous edge
    neighs = Gsucc


    prev_edge_type = last_edge_type
    prev_upstr_node_graph_type = last_upstr_node_graph_type
    run_id_till_node_u = last_pt_veh_run_id
    previous_edge_cost = last_edge_cost
    zone_at_start_of_pt_trip = pt_trip_orig_zone
    prev_mode = previous_edge_mode

    curr_time = current_time
    seen[source] = init_weight

    push(fringe, (init_weight, next(c), source, init_travel_time, init_wait_time, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, prev_edge_type, prev_upstr_node_graph_type, run_id_till_node_u, curr_time, previous_edge_cost, zone_at_start_of_pt_trip, prev_mode))                      # LY: added initial time of request as argument and prev_edge_type

    while fringe:
        (gen_w, _, v, tt, wtt, d, mon_c, n_l_ts, n_m_ts, pr_ed_tp, pre_upstr_n_gr_tp, lt_run_id, curr_time, pr_e_cost, pt_tr_st_z, pr_md) = pop(fringe)

        if v in weight_label:
            continue  # already searched this node.

        weight_label[v] = gen_w
        travel_time_label[v] = tt
        wait_time_label[v] = wtt
        distance_label[v] = d
        mon_cost_label[v] = mon_c
        line_trans_num_label[v] = n_l_ts
        mode_trans_num_label[v] = n_m_ts
        prev_edge_type_label[v] = pr_ed_tp
        prev_upstr_node_graph_type_label[v] = pre_upstr_n_gr_tp
        last_pt_veh_run_id_label[v] = lt_run_id
        current_time_label[v] = curr_time
        prev_edge_cost_label[v] = pr_e_cost
        pt_trip_start_zone_label[v] = pt_tr_st_z
        prev_mode_label[v] = pr_md

        if v == target:
          break
#        if v == '11,5_2,5':
#            print('stop')
#        if v == '40,3_1,6':
#            print('stop')
#        if v == '60,5_2,7':
#            print('stop')
#        if v ==  'w_bus_stop47':
#            print('oops')
#        if v == 'b47':
#            print('oops')
#        if v == 'b38':
#            print('oops')

        for u in neighs(v):
#          if u == '40,3_1,6':
#            print('stop')
          e_type = edge_type_data(v, u, G[v][u])
          # n_type = node_type_data(v, G.nodes[v])
          n_gr_type = node_graph_type_data(v, G.nodes[v])
          prev_mode = pr_md
          zone_at_start_of_pt_trip = pt_tr_st_z
          previous_edge_cost = pr_e_cost


          if e_type == 'car_sharing_station_egress_edge' or e_type == 'car_sharing_station_access_edge':
            penalty = 0
            e_tt = 0
            e_wait_time = G[v][u]['wait_time']
            e_cost=0
            e_distance = 0
            e_num_lin_trf = 0
            e_num_mode_trf = 0

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf
            time_till_u = curr_time + e_tt + e_wait_time

            if e_type == 'car_sharing_station_egress_edge' and pr_ed_tp != 'access_edge':
              penalty = math.inf

            vu_dist = weight_label[v] + cs_attrs_w[0]*e_tt + cs_attrs_w[1]*e_wait_time + cs_attrs_w[2]*e_distance + cs_attrs_w[3]*e_cost + cs_attrs_w[4]*e_num_lin_trf + penalty


          if e_type == 'car_sharing_orig_dummy_edge':
            e_wait_time = 0
            e_tt = 0
            e_distance = 0
            e_cost = 0
            e_num_mode_trf = 0
            e_num_lin_trf = 0

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf
            time_till_u = curr_time + e_tt + e_wait_time

            vu_dist = weight_label[v] + cs_attrs_w[0]*e_tt + cs_attrs_w[1]*e_wait_time + cs_attrs_w[2]*e_distance + cs_attrs_w[3]*e_cost + cs_attrs_w[4]*e_num_lin_trf

          if e_type == 'car_sharing_dest_dummy_edge' or e_type == 'car_sharing_dual_edge':
            e_wait_time = 0
            tt_d = travel_time_data(v, u, G[v][u])
            if tt_d is None:
              print('Missing in_veh_tt value in edge {}'.format((v, u)))
              continue
            e_tt = get_time_dep_taxi_travel_time(curr_time+e_wait_time, tt_d) # used this function cause it works the way it should, need to however have some generic function to extract data from different type of dictionaries without being mode-specific
            e_distance = distance_data(v, u, G[v][u])
            if e_distance is None:
              print('Missing distance value in edge {}'.format((v, u)))
              continue
            e_cost_data = G[v][u]['car_sharing_fares']
            e_cost = get_time_dep_taxi_cost(curr_time, e_cost_data) # # used this function cause it works the way it should, need to however have some generic function to extract data from different type of dictionaries without being mode-specific
            e_num_mode_trf = 0
            e_num_lin_trf = 0

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf
            time_till_u = curr_time + e_tt + e_wait_time

            vu_dist = weight_label[v] + cs_attrs_w[0]*e_tt + cs_attrs_w[1]*e_wait_time + cs_attrs_w[2]*e_distance + cs_attrs_w[3]*e_cost + cs_attrs_w[4]*e_num_lin_trf


          if e_type == 'taxi_edge' or e_type == 'on_demand_single_taxi_edge' or e_type == 'on_demand_shared_taxi_edge':
            e_wait_time_data = taxi_wait_time(v, u, G[v][u])
            e_wait_time = get_time_dep_taxi_wait_time(curr_time, e_wait_time_data)
            tt_d = travel_time_data(v, u, G[v][u])
            if tt_d is None:
              print('Missing in_veh_tt value in edge {}'.format((v, u)))
              continue
            e_tt = get_time_dep_taxi_travel_time(curr_time+e_wait_time, tt_d)
            e_distance = distance_data(v, u, G[v][u])        # funtion that extracts the edge's distance attribute
            if e_distance is None:
              print('Missing distance value in edge {}'.format((v, u)))
              continue
            e_num_mode_trf = 0
            e_num_lin_trf = 0

            # if pr_ed_tp == 'access_edge':
            #   zone_at_start_of_pt_trip = G.nodes[v]['zone']
            #   previous_edge_cost = 0


            e_cost_data = taxi_fares(v, u, G[v][u])
            e_cost = get_time_dep_taxi_cost(curr_time, e_cost_data)


            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf
            time_till_u = curr_time + e_tt + e_wait_time

            if e_type == 'taxi_edge':
              vu_dist = weight_label[v] + taxi_attrs_w[0]*e_tt + taxi_attrs_w[1]*e_wait_time + taxi_attrs_w[2]*e_distance + taxi_attrs_w[3]*e_cost + taxi_attrs_w[4]*e_num_lin_trf
            elif e_type == 'on_demand_single_taxi_edge':
              vu_dist = weight_label[v] + sms_attrs_w[0]*e_tt + sms_attrs_w[1]*e_wait_time + sms_attrs_w[2]*e_distance + sms_attrs_w[3]*e_cost + sms_attrs_w[4]*e_num_lin_trf
            elif e_type == 'on_demand_shared_taxi_edge':
              vu_dist = weight_label[v] + sms_pool_attrs_w[0]*e_tt + sms_pool_attrs_w[1]*e_wait_time + sms_pool_attrs_w[2]*e_distance + sms_pool_attrs_w[3]*e_cost + sms_pool_attrs_w[4]*e_num_lin_trf

          if e_type == 'walk_edge':
            e_tt = travel_time_data(v, u, G[v][u])  # funtion that extracts the edge's in-vehicle travel time attribute
            if e_tt is None:
              print('Missing in_veh_tt value in edge {}'.format((v, u)))
              continue
            e_wait_time = 0
            e_distance = distance_data(v, u, G[v][u])        # funtion that extracts the edge's distance attribute
            if e_distance is None:
              print('Missing distance value in edge {}'.format((v, u)))
              continue
            e_cost = 0
            e_num_mode_trf = 0
            e_num_lin_trf = 0

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf
            

            time_till_u = curr_time + e_tt + e_wait_time

            vu_dist = weight_label[v] + walk_attrs_w[0]*e_tt + walk_attrs_w[1]*e_wait_time + walk_attrs_w[2]*e_distance + walk_attrs_w[3]*e_cost + walk_attrs_w[4]*e_num_lin_trf
          # if e_type == 'orig_dummy_edge':
          #   road_edge_cost = weight(v, u, G[v][u])
          #   if road_edge_cost is None:
          #     continue
          #   vu_dist = dist[v] + road_edge_cost
          # if e_type == 'dest_dummy_edge' or e_type == 'dual_edge':
          #   road_edge_cost = calc_road_link_tt(dist[v], G[v][u])             # the travel time assigned here is the travel time of the corresponding 5min interval based on historic data
          #   if road_edge_cost is None:
          #     continue
          #   vu_dist = dist[v] + road_edge_cost
          if e_type == 'access_edge':
            penalty = 0
            # when we are at an access edge that connects graphs we need to penalize unreasonable connections and path loops; e.g., from walk node to another mode-specific node and back to walk node, from a taxi/carsharing trip to a walk trip and back to taxi/carsharing trip
            if (pr_md == 'taxi_graph' or pr_md == 'on_demand_single_taxi_graph' or pr_md == 'on_demand_shared_taxi_graph' or pr_md == 'car_sharing_graph') and (G.nodes[u]['node_graph_type'] == 'taxi_graph' or G.nodes[u]['node_graph_type'] == 'on_demand_single_taxi_graph' or G.nodes[u]['node_graph_type'] == 'on_demand_shared_taxi_graph' or G.nodes[u]['node_graph_type'] == 'car_sharing_graph'):
              penalty = math.inf
            if pr_ed_tp == 'access_edge':
              if (pre_upstr_n_gr_tp == 'Walk' and G.nodes[u]['node_graph_type'] == 'Walk') or (pre_upstr_n_gr_tp == 'Bus' and G.nodes[u]['node_graph_type'] == 'Bus') or (pre_upstr_n_gr_tp == 'Train' and G.nodes[u]['node_graph_type'] == 'Tain'):
                penalty = math.inf
            if G.nodes[u]['node_type'] == 'car_sharing_station_node':
              if G.nodes[u]['stock_level'] == 0:
                penalty = math.inf
            if G.nodes[v]['node_type'] == 'car_sharing_station_node':
              if G.nodes[v]['stock_level'] == G.nodes[v]['capacity']:
                penalty = math.inf
            # restraint pick up and drop off
            if G.nodes[v]['node_graph_type'] == 'Walk' and (G.nodes[u]['node_graph_type'] == 'taxi_graph' or G.nodes[u]['node_graph_type'] == 'on_demand_single_taxi_graph' or G.nodes[u]['node_graph_type'] == 'on_demand_shared_taxi_graph'):
              if G.nodes[orig_source]['node_graph_type'] == 'Walk' and v != orig_source and pr_md==None:
                  penalty = math.inf
            if G.nodes[u]['node_graph_type'] == 'Walk' and (G.nodes[v]['node_graph_type'] == 'taxi_graph' or G.nodes[v]['node_graph_type'] == 'on_demand_single_taxi_graph' or G.nodes[v]['node_graph_type'] == 'on_demand_shared_taxi_graph'):
              if G.nodes[u]['zone'] == G.nodes[target]['zone'] and u != target:
                penalty = math.inf

            e_tt = 0
            e_wait_time = 0
            e_distance = 0
            e_cost = 0
            e_num_mode_trf = 0
            e_num_lin_trf = 0
            
            # from mode and line transfers we store the previous path mode (not considering walk as a mode) and if a path with a new mode starts then we have a mode transfer, if with the same mode then a line transfer
            if G.nodes[v]['node_graph_type'] == 'Walk':
              if pr_md != None and G.nodes[u]['node_graph_type'] != pr_md:
                e_num_mode_trf = 1
                e_num_lin_trf = 0
              elif (pr_md =='Train' or pr_md =='Bus') and G.nodes[u]['node_graph_type'] == pr_md:
                e_num_mode_trf = 0
                e_num_lin_trf = 1
            if G.nodes[u]['node_graph_type'] == 'Walk':
              prev_mode = G.nodes[v]['node_graph_type']

            # # following two conditions are just because we lack proper walk graph representation in VC
            # if G.nodes[v]['node_graph_type'] == 'Bus' and G.nodes[u]['node_graph_type'] == 'Train':
            #   e_num_mode_trf = 1
            #   e_num_lin_trf = 0
            #   prev_mode = G.nodes[v]['node_graph_type']
            # if G.nodes[u]['node_graph_type'] == 'Bus' and G.nodes[v]['node_graph_type'] == 'Train':
            #   e_num_mode_trf = 1
            #   e_num_lin_trf = 0
            #   prev_mode = G.nodes[v]['node_graph_type']

            time_till_u = curr_time + e_tt + e_wait_time

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

            vu_dist = weight_label[v] + bus_attrs_w[4]*e_num_lin_trf + mode_transfer_weight*e_num_mode_trf + penalty
          # cost calculation process for a transfer edge in bus or train stops/stations

          if e_type == 'pt_transfer_edge':
            # for zone_to_zone pt fare scheme we store the zone of the stop/station in which a pt trip started (origin); this zone will be used for the calculcation of the edge cost based on which pt stop the algorithm checks and hence the final stop of the pt trip
            if fare_scheme == 'zone_to_zone':
              if pr_ed_tp == 'access_edge':
                if pr_md != 'Bus' and pr_md != 'Train':
                  zone_at_start_of_pt_trip = G.nodes[v]['zone']
                  previous_edge_cost = 0
            e_tt = travel_time_data(v, u, G[v][u])  # funtion that extracts the edge's in-vehicle travel time attribute
            if e_tt is None:
              print('Missing in_veh_tt value in edge {}'.format((v, u)))
              continue
            e_wait_time = 0
            e_distance = distance_data(v, u, G[v][u])        # funtion that extracts the edge's distance attribute
            if e_distance is None:
              print('Missing distance value in edge {}'.format((v, u)))
              continue
            e_num_mode_trf = 0
            # to cmpute line transfers in pt we check the previous edge type; if the previous edge type is also a tranfer edge then we have a line transfer; this constraint allows us to avoid adding a line transfer when the algorithm traverses a transfer edge at the start of a pt trip
            if pr_ed_tp == 'pt_transfer_edge':
              e_num_lin_trf = 1
            else:
              e_num_lin_trf = 0

            e_cost = 0

            time_till_u = curr_time + e_tt + e_wait_time

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

            if G.nodes[u]['node_graph_type'] == 'Bus' and G.nodes[v]['node_graph_type'] == 'Bus':
              vu_dist = weight_label[v] + bus_attrs_w[0]*e_tt + bus_attrs_w[1]*e_wait_time + bus_attrs_w[2]*e_distance + bus_attrs_w[3]*e_cost + bus_attrs_w[4]*e_num_lin_trf
            else:
              vu_dist = weight_label[v] + train_attrs_w[0]*e_tt + train_attrs_w[1]*e_wait_time + train_attrs_w[2]*e_distance + train_attrs_w[3]*e_cost + train_attrs_w[4]*e_num_lin_trf

          # cost calculation process for a pt route edge in bus or train stops/stations
          if e_type == 'pt_route_edge':
            # for pt route edges the waiting time and travel time is calculated differently; based on the time-dependent model and the FIFO assumption, if the type of previous edge is transfer edge, we assume that the fastest trip will be the one with the first departing bus/train after the current time (less waiting time) and the travel time will be the one of the corresponding pt vehicle run_id; but if the type of the previous edge is a route edge, then for this line/route a pt_vehcile has already been chosen and the edge travel time will be the one for this specific vehicle of the train/bus line (in this case the wait time is 0)

            if pr_ed_tp == 'pt_transfer_edge':
              dep_timetable = timetable_data(v, u, G[v][u])  # fuction that extracts the stop's/station's timetable dict
              if dep_timetable is None:
                print('Missing timetable value in edge'.format((v, u)))
                continue
              e_wait_time, pt_vehicle_run_id = calc_plat_wait_time_and_train_id(curr_time, dep_timetable)  # function that extracts waiting time for next pt vehicle and the vehicle_id; the next departing vehicle is being found using a binary search algorithm that operates on a sorted list of the deparure times for this edge (departure times of the downstream stop/station)
              if e_wait_time is None:
                print('Missing wait_time value in edge'.format((v, u)))
                continue
              tt_d = travel_time_data(v, u, G[v][u])  # fuction that extracts the travel time dict
              if tt_d is None:
                print('Missing in_veh_tt value in edge'.format((v, u)))
                continue
              e_tt = tt_d[pt_vehicle_run_id] #calc_pt_route_edge_in_veh_tt_for_run_id(tt_d, pt_vehicle_run_id)  # fuction that travel time for corresponding pt vehicle run_id
            elif pr_ed_tp == 'pt_route_edge':
              e_wait_time = 0
              pt_vehicle_run_id = lt_run_id
              e_tt = G[v][u]['departure_time'][pt_vehicle_run_id] - curr_time + G[v][u]['travel_time'][pt_vehicle_run_id]  # the subtraction fo the first two terms is the dwell time in the downstream station and the 3rd term is the travel time of the pt vehicle run_id that has been selected for the previous route edge
              if e_tt is None:
                print('Missing in_veh_tt value in edge'.format((v, u)))
                continue
            e_distance = distance_data(v, u, G[v][u])  # fuction that extracts the travel time dict
            if e_distance is None:
              print('Missing distance value in edge'.format((v, u)))
              continue
            # edge costs for pt depend on the pt fare scheme; if it is additive (distance_based) or zone_to_zone !! consider adding a price cap !!
            if fare_scheme == 'distance_based':
              dist_bas_cost = pt_additive_cost_data(v, u, G[v][u])  # fuction that extracts the time-dependent distance-based cost dict
              if dist_bas_cost is None:
                print('Missing dist_bas_cost value in edge'.format((v, u)))
                continue
              e_cost = calc_time_dep_distance_based_cost(dist_bas_cost, curr_time)  # fuction that extracts the cost based on time-dependent distance-based cost dict and the current time (finds in which time-zone we currently are)
              cost_till_u = mon_cost_label[v] + e_cost #- pr_e_cost
            elif fare_scheme == 'zone_to_zone':
              zn_to_zn_cost = pt_non_additive_cost_data(v, u, G[v][u])  # fuction that extracts the time-dependent zone_to_zone cost dict
              if zn_to_zn_cost is None:
                print('Missing zn_to_zn_cost value in edge'.format((v, u)))
                continue
              pt_cur_cost = calc_time_dep_zone_to_zone_cost(zn_to_zn_cost, curr_time, pt_tr_st_z, G.nodes[u]['zone'])  # function that extracts the cost of the edge based on the zone at the start of the pt trip, the zone of current stop/station and the current time we are in
#              if pt_cur_cost == None or pr_e_cost == None:
#                  print('stop')
              if pt_cur_cost < pr_e_cost:
                e_cost = 0
                previous_edge_cost = pr_e_cost
              else:
                e_cost = pt_cur_cost - pr_e_cost
                previous_edge_cost = pt_cur_cost  # here only for the case of zone_to_zone pt fare schemes we update the previous edge cost only after the label (edge weight) calculation
              # if pt_cur_cost<pr_e_cost:
              #   print('Previous cost is higher than new cost in {}'.format(paths[v]))
            e_num_lin_trf = 0
            e_num_mode_trf = 0

            time_till_u = curr_time + e_tt + e_wait_time

            travel_time_till_u = travel_time_label[v] + e_tt
            wait_time_till_u = wait_time_label[v] + e_wait_time
            distance_till_u = distance_label[v] + e_distance
            cost_till_u = mon_cost_label[v] + e_cost
            line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
            mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

            if G.nodes[u]['node_graph_type'] == 'Bus' and G.nodes[v]['node_graph_type'] == 'Bus':
              vu_dist = weight_label[v] + bus_attrs_w[0]*e_tt + bus_attrs_w[1]*e_wait_time + bus_attrs_w[2]*e_distance + bus_attrs_w[3]*e_cost + bus_attrs_w[4]*e_num_lin_trf
            else:
              vu_dist = weight_label[v] + train_attrs_w[0]*e_tt + train_attrs_w[1]*e_wait_time + train_attrs_w[2]*e_distance + train_attrs_w[3]*e_cost + train_attrs_w[4]*e_num_lin_trf


          if u in weight_label:
            if vu_dist < weight_label[u]:
              # print(weight_label, weight_label[u], paths[v])
              print('Negative weight in node {}, in edge {}, {}?'.format(u, v, u))
              raise ValueError('Contradictory paths found:',
                               'negative weights?')
          elif u not in seen or vu_dist < seen[u]:
            seen[u] = vu_dist
            if e_type == 'pt_route_edge' and pr_ed_tp != 'pt_route_edge':
              push(fringe, (vu_dist, next(c), u, travel_time_till_u, wait_time_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, n_gr_type, pt_vehicle_run_id, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip, prev_mode))
            elif e_type == 'pt_route_edge' and pr_ed_tp == 'pt_route_edge':
              push(fringe, (vu_dist, next(c), u, travel_time_till_u, wait_time_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, n_gr_type, pt_vehicle_run_id, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip, prev_mode))
            elif e_type == 'pt_transfer_edge':
              push(fringe, (vu_dist, next(c), u, travel_time_till_u, wait_time_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, n_gr_type, None, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip, prev_mode))
            elif e_type != 'pt_route_edge' and e_type != 'pt_transfer_edge':
              push(fringe, (vu_dist, next(c), u, travel_time_till_u, wait_time_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, n_gr_type, None, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip, prev_mode))
            if paths is not None:
              paths[u] = paths[v] + [u]
            if pred is not None:
              pred[u] = [v]
          elif vu_dist == seen[u]:
            if pred is not None:
              pred[u].append(v)

        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.

    try:
        return (weight_label[target], paths[target], travel_time_label[target], wait_time_label[target], distance_label[target], mon_cost_label[target], line_trans_num_label[target], mode_trans_num_label[target], travel_time_label, wait_time_label, distance_label, mon_cost_label, line_trans_num_label, mode_trans_num_label, weight_label, prev_edge_type_label, prev_upstr_node_graph_type_label, last_pt_veh_run_id_label, current_time_label, prev_edge_cost_label, pt_trip_start_zone_label, prev_mode_label)
    except KeyError:
        raise nx.NetworkXNoPath("Node %s not reachable from %s" % (target, source))
