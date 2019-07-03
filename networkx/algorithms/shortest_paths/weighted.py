# -*- coding: utf-8 -*-
#    Copyright (C) 2004-2019 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Authors:  Aric Hagberg <hagberg@lanl.gov>
#           Loïc Séguin-C. <loicseguin@gmail.com>
#           Dan Schult <dschult@colgate.edu>
#           Niels van Adrichem <n.l.m.vanadrichem@tudelft.nl>
"""
Shortest path algorithms for weighed graphs.
"""

from collections import deque
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import math
from networkx.utils import generate_unique_node
from bisect import bisect_left


__all__ = ['dijkstra_path',
           'dijkstra_path_length',
           'bidirectional_dijkstra',
           'single_source_dijkstra',
           'single_source_dijkstra_path',
           'single_source_dijkstra_path_length',
           'multi_source_dijkstra',
           'multi_source_dijkstra_path',
           'multi_source_dijkstra_path_length',
           'all_pairs_dijkstra',
           'all_pairs_dijkstra_path',
           'all_pairs_dijkstra_path_length',
           'dijkstra_predecessor_and_distance',
           'bellman_ford_path',
           'bellman_ford_path_length',
           'single_source_bellman_ford',
           'single_source_bellman_ford_path',
           'single_source_bellman_ford_path_length',
           'all_pairs_bellman_ford_path',
           'all_pairs_bellman_ford_path_length',
           'bellman_ford_predecessor_and_distance',
           'negative_edge_cycle',
           'goldberg_radzik',
           'johnson']


def find_ge(a, x):                                  # binary search algorithm
  'Find leftmost item greater than or equal to x'
  i = bisect_left(a, x)
  if i != len(a):
    return a[i]
  raise ValueError


def calc_time_dep_distance_based_cost(dist_based_cost_data, current_time):
  for time_zone, cost in dist_based_cost_data.items():
    if int(current_time) >= int(time_zone[0]) and int(current_time) <= int(time_zone[1]):
      edge_cost = cost
      break
  return edge_cost


def calc_time_dep_zone_to_zone_cost(zn_to_zn_cost_data, current_time, pt_trip_start_zone, u_zone):
  if not pt_trip_start_zone:
    print('Zone value for the start of pt trip has not assigned')
    pass
  else:
    for time_zone in zn_to_zn_cost_data:
      if int(current_time) >= int(time_zone[0]) and int(current_time) <= int(time_zone[1]):
        cost = zn_to_zn_cost_data[time_zone][(pt_trip_start_zone, u_zone)]
        break
    if cost is None:
      print('No cost from zone_to_zone data found')
    return cost


def calc_plat_wait_time_and_train_id(arr_time=0, edge_departure_time={}):
  # the sorted list generation might need to go outside the algorithm as it increases computational performance
  list_of_ptedge_dep_times = []
  for run_id in edge_departure_time:
    list_of_ptedge_dep_times.append(edge_departure_time[run_id])
  list_of_ptedge_dep_times.sort()
  earlier_dep_time = find_ge(list_of_ptedge_dep_times, arr_time)
  platform_wait_time = earlier_dep_time - arr_time
  for run_id in edge_departure_time:
    if earlier_dep_time == edge_departure_time[run_id]:
      corresponding_runid_to_earl_dep_time = run_id
      break
  return platform_wait_time, corresponding_runid_to_earl_dep_time


def calc_pt_route_edge_in_veh_tt_for_run_id(edge_travel_times, run_id_of_first_dep_ptveh):
  tt = None
  for run_id in edge_travel_times:
    if run_id == run_id_of_first_dep_ptveh:
      tt = edge_travel_times[run_id]
      break
  if tt == None:
    print('Run_id of first departing train could not be found in the travel time dict')
  else:
    return tt


def calc_road_link_tt(cur_time, edge_attrs):  # calculate the proper travel time for the respective 5-min interval based on current time in the network
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
  return lambda u, v, data: data.get(node_type, None)


def _get_in_vehicle_tt_function(G, in_vehicle_tt):
  return lambda u, v, data: data.get(in_vehicle_tt, None)


def _get_distance_function(G, distance):
  return lambda u, v, data: data.get(distance, None)


def _get_distance_based_cost(G, distance_based_cost):
  return lambda u, v, data: data.get(distance_based_cost, None)


def _get_zone_to_zone_cost(G, zone_to_zone_cost):
  return lambda u, v, data: data.get(zone_to_zone_cost, None)


def _get_walk_tt(G, walk_tt):
  return lambda u, v, data: data.get(walk_tt, None)


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


def dijkstra_path(G, source, target, time_of_request, in_vehicle_tt='in_vehicle_tt', walk_tt='walk_tt', distance='distance', distance_based_cost='distance_based_cost', zone_to_zone_cost='zone_to_zone_cost', timetable='departure_time', edge_type='edge_type', fare_scheme='distance_based'):  # LY: added initial time of request as argument
  """Returns the shortest weighted path from source to target in G.

  Uses Dijkstra's Method to compute the shortest weighted path
  between two nodes in a graph.

  Parameters
  ----------
  G : NetworkX graph

  source : node
     Starting node

  target : node
     Ending node

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  path : list
     List of nodes in a shortest path.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  NetworkXNoPath
     If no path exists between source and target.

  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> print(nx.dijkstra_path(G,0,4))
  [0, 1, 2, 3, 4]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  The weight function can be used to include node weights.

  >>> def func(u, v, d):
  ...     node_u_wt = G.nodes[u].get('node_weight', 1)
  ...     node_v_wt = G.nodes[v].get('node_weight', 1)
  ...     edge_wt = d.get('weight', 1)
  ...     return node_u_wt/2 + node_v_wt/2 + edge_wt

  In this example we take the average of start and end node
  weights of an edge and add it to the weight of the edge.

  See Also
  --------
  bidirectional_dijkstra(), bellman_ford_path()
  """
  (weight, path, weight_dict, paths_dict) = single_source_dijkstra(G, source, time_of_request, target=target, in_vehicle_tt=in_vehicle_tt, walk_tt=walk_tt, distance=distance, distance_based_cost=distance_based_cost, zone_to_zone_cost=zone_to_zone_cost, timetable=timetable, edge_type=edge_type, fare_scheme=fare_scheme)
  return path, weight, weight_dict, paths_dict


def dijkstra_path_length(G, source, target, time_of_request, weight='weight', timetable='departure_time', edge_type='edge_type'):
  """Returns the shortest weighted path length in G from source to target.

  Uses Dijkstra's Method to compute the shortest weighted path length
  between two nodes in a graph.

  Parameters
  ----------
  G : NetworkX graph

  source : node label
     starting node for path

  target : node label
     ending node for path

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  length : number
      Shortest path length.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  NetworkXNoPath
      If no path exists between source and target.

  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> print(nx.dijkstra_path_length(G,0,4))
  4

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  See Also
  --------
  bidirectional_dijkstra(), bellman_ford_path_length()

  """
  if source == target:
    return 0
  weight = _weight_function(G, weight)
  timetable = _get_timetable(G, timetable)
  edge_type = _get_edge_type(G, edge_type)
  length = _dijkstra(G, source, time_of_request, weight, timetable, edge_type, target=target)
  try:
    return length[target]
  except KeyError:
    raise nx.NetworkXNoPath(
        "Node %s not reachable from %s" % (target, source))


def single_source_dijkstra_path(G, source, cutoff=None, weight='weight'):
  """Find shortest weighted paths in G from a source node.

  Compute shortest path between source and all other reachable
  nodes for a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  source : node
     Starting node for path.

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  paths : dictionary
     Dictionary of shortest path lengths keyed by target.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> path=nx.single_source_dijkstra_path(G,0)
  >>> path[4]
  [0, 1, 2, 3, 4]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  See Also
  --------
  single_source_dijkstra(), single_source_bellman_ford()

  """
  return multi_source_dijkstra_path(G, {source}, cutoff=cutoff,
                                    weight=weight)


def single_source_dijkstra_path_length(G, source, cutoff=None,
                                       weight='weight'):
  """Find shortest weighted path lengths in G from a source node.

  Compute the shortest path length between source and all other
  reachable nodes for a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  source : node label
     Starting node for path

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  length : dict
      Dict keyed by node to shortest path length from source.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length = nx.single_source_dijkstra_path_length(G, 0)
  >>> length[4]
  4
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  See Also
  --------
  single_source_dijkstra(), single_source_bellman_ford_path_length()

  """
  return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff,
                                           weight=weight)


def single_source_dijkstra(G, source, time_of_request, target=None, cutoff=None, in_vehicle_tt='in_vehicle_tt', walk_tt='walk_tt', distance='distance', distance_based_cost='distance_based_cost', zone_to_zone_cost='zone_to_zone_cost', timetable='departure_time', edge_type='edge_type', fare_scheme='distance_based'):        # LY: added initial time of request as argument
  """Find shortest weighted paths and lengths from a source node.

  Compute the shortest path length between source and all other
  reachable nodes for a weighted graph.

  Uses Dijkstra's algorithm to compute shortest paths and lengths
  between a source and all other reachable nodes in a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  source : node label
     Starting node for path

  target : node label, optional
     Ending node for path

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  distance, path : pair of dictionaries, or numeric and list.
     If target is None, paths and lengths to all nodes are computed.
     The return value is a tuple of two dictionaries keyed by target nodes.
     The first dictionary stores distance to each target node.
     The second stores the path to each target node.
     If target is not None, returns a tuple (distance, path), where
     distance is the distance from source to target and path is a list
     representing the path from source to target.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length, path = nx.single_source_dijkstra(G, 0)
  >>> print(length[4])
  4
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  >>> path[4]
  [0, 1, 2, 3, 4]
  >>> length, path = nx.single_source_dijkstra(G, 0, 1)
  >>> length
  1
  >>> path
  [0, 1]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  Based on the Python cookbook recipe (119466) at
  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

  This algorithm is not guaranteed to work if edge weights
  are negative or are floating point numbers
  (overflows and roundoff errors can cause problems).

  See Also
  --------
  single_source_dijkstra_path()
  single_source_dijkstra_path_length()
  single_source_bellman_ford()
  """
  return multi_source_dijkstra(G, {source}, time_of_request, cutoff=cutoff, target=target, in_vehicle_tt=in_vehicle_tt, walk_tt=walk_tt, distance=distance, distance_based_cost=distance_based_cost, zone_to_zone_cost=zone_to_zone_cost, timetable=timetable, edge_type=edge_type, fare_scheme=fare_scheme)     # LY: added initial time of request as argument


def multi_source_dijkstra_path(G, sources, cutoff=None, weight='weight'):
  """Find shortest weighted paths in G from a given set of source
  nodes.

  Compute shortest path between any of the source nodes and all other
  reachable nodes for a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  sources : non-empty set of nodes
      Starting nodes for paths. If this is just a set containing a
      single node, then all paths computed by this function will start
      from that node. If there are two or more nodes in the set, the
      computed paths may begin from any one of the start nodes.

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  paths : dictionary
     Dictionary of shortest paths keyed by target.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> path = nx.multi_source_dijkstra_path(G, {0, 4})
  >>> path[1]
  [0, 1]
  >>> path[3]
  [4, 3]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  Raises
  ------
  ValueError
      If `sources` is empty.
  NodeNotFound
      If any of `sources` is not in `G`.

  See Also
  --------
  multi_source_dijkstra(), multi_source_bellman_ford()

  """
  length, path = multi_source_dijkstra(G, sources, cutoff=cutoff,
                                       weight=weight)
  return path


def multi_source_dijkstra_path_length(G, sources, cutoff=None,
                                      weight='weight'):
  """Find shortest weighted path lengths in G from a given set of
  source nodes.

  Compute the shortest path length between any of the source nodes and
  all other reachable nodes for a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  sources : non-empty set of nodes
      Starting nodes for paths. If this is just a set containing a
      single node, then all paths computed by this function will start
      from that node. If there are two or more nodes in the set, the
      computed paths may begin from any one of the start nodes.

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  length : dict
      Dict keyed by node to shortest path length to nearest source.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length = nx.multi_source_dijkstra_path_length(G, {0, 4})
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 1
  4: 0

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  Raises
  ------
  ValueError
      If `sources` is empty.
  NodeNotFound
      If any of `sources` is not in `G`.

  See Also
  --------
  multi_source_dijkstra()

  """
  if not sources:
    raise ValueError('sources must not be empty')
  weight = _weight_function(G, weight)
  return _dijkstra_multisource(G, sources, weight, cutoff=cutoff)


def multi_source_dijkstra(G, sources, time_of_request, target=None, cutoff=None, in_vehicle_tt='in_vehicle_tt', walk_tt='walk_tt', distance='distance', distance_based_cost='distance_based_cost', zone_to_zone_cost='zone_to_zone_cost', timetable='departure_time', edge_type='edge_type', fare_scheme='distance_based'):        # LY: added initial time of request as argument
  """Find shortest weighted paths and lengths from a given set of
  source nodes.

  Uses Dijkstra's algorithm to compute the shortest paths and lengths
  between one of the source nodes and the given `target`, or all other
  reachable nodes if not specified, for a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  sources : non-empty set of nodes
      Starting nodes for paths. If this is just a set containing a
      single node, then all paths computed by this function will start
      from that node. If there are two or more nodes in the set, the
      computed paths may begin from any one of the start nodes.

  target : node label, optional
     Ending node for path

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  distance, path : pair of dictionaries, or numeric and list
     If target is None, returns a tuple of two dictionaries keyed by node.
     The first dictionary stores distance from one of the source nodes.
     The second stores the path from one of the sources to that node.
     If target is not None, returns a tuple of (distance, path) where
     distance is the distance from source to target and path is a list
     representing the path from source to target.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length, path = nx.multi_source_dijkstra(G, {0, 4})
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 1
  4: 0
  >>> path[1]
  [0, 1]
  >>> path[3]
  [4, 3]

  >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)
  >>> length
  1
  >>> path
  [0, 1]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The weight function can be used to hide edges by returning None.
  So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
  will find the shortest red path.

  Based on the Python cookbook recipe (119466) at
  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

  This algorithm is not guaranteed to work if edge weights
  are negative or are floating point numbers
  (overflows and roundoff errors can cause problems).

  Raises
  ------
  ValueError
      If `sources` is empty.
  NodeNotFound
      If any of `sources` is not in `G`.

  See Also
  --------
  multi_source_dijkstra_path()
  multi_source_dijkstra_path_length()

  """
  if not sources:
    raise ValueError('sources must not be empty')
  if target in sources:
    return (0, [target])
  in_vehicle_tt = _get_in_vehicle_tt_function(G, in_vehicle_tt)
  walk_tt = _get_walk_tt(G, walk_tt)
  distance = _get_distance_function(G, distance)
  distance_based_cost = _get_distance_based_cost(G, distance_based_cost)
  zone_to_zone_cost = _get_zone_to_zone_cost(G, zone_to_zone_cost)
  timetable = _get_timetable(G, timetable)
  edge_type = _get_edge_type(G, edge_type)
  paths = {source: [source] for source in sources}  # dictionary of paths
  weight = _dijkstra_multisource(G, sources, time_of_request, in_vehicle_tt, walk_tt, distance, distance_based_cost, zone_to_zone_cost, timetable, edge_type, fare_scheme, paths=paths, cutoff=cutoff, target=target)      # LY: added initial time of request as argument
  if target is None:
    return (weight, paths)
  try:
    return (weight[target], paths[target], weight, paths)
  except KeyError:
    raise nx.NetworkXNoPath("No path to {}.".format(target))


def _dijkstra(G, source, time_of_request, weight, timetable, edge_type, pred=None, paths=None, cutoff=None,
              target=None):
  """Uses Dijkstra's algorithm to find shortest weighted paths from a
  single source.

  This is a convenience function for :func:`_dijkstra_multisource`
  with all the arguments the same, except the keyword argument
  `sources` set to ``[source]``.

  """
  return _dijkstra_multisource(G, [source], time_of_request, weight, timetable, edge_type, pred=pred, paths=paths,
                               cutoff=cutoff, target=target)


def _dijkstra_multisource(G, sources, time_of_request, in_vehicle_tt, walk_tt, distance, distance_based_cost, zone_to_zone_cost, timetable, edge_type, fare_scheme, pred=None, paths=None, cutoff=None, target=None):        # LY: added initial time of request as argument
  """Uses Dijkstra's algorithm to find shortest weighted paths

  Parameters
  ----------
  G : NetworkX graph

  sources : non-empty iterable of nodes
      Starting nodes for paths. If this is just an iterable containing
      a single node, then all paths computed by this function will
      start from that node. If there are two or more nodes in this
      iterable, the computed paths may begin from any one of the start
      nodes.

  weight: function
      Function with (u, v, data) input that returns that edges weight

  pred: dict of lists, optional(default=None)
      dict to store a list of predecessors keyed by that node
      If None, predecessors are not stored.

  paths: dict, optional (default=None)
      dict to store the path list from source to each node, keyed by node.
      If None, paths are not stored.

  target : node label, optional
      Ending node for path. Search is halted when target is found.

  cutoff : integer or float, optional
      Depth to stop the search. Only return paths with length <= cutoff.

  Returns
  -------
  distance : dictionary
      A mapping from node to shortest distance to that node from one
      of the source nodes.

  Raises
  ------
  NodeNotFound
      If any of `sources` is not in `G`.

  Notes
  -----
  The optional predecessor and path dictionaries can be accessed by
  the caller through the original pred and paths objects passed
  as arguments. No need to explicitly return pred or paths.

  """
  G_succ = G._succ if G.is_directed() else G._adj

  push = heappush
  pop = heappop
  node_labels = {}  # dictionary of best weight
  # nodes_labels_dict = {}
  seen = {}
  # fringe is heapq with 3-tuples (distance,c,node) -and I change it to a 4-tuple to store the type of the previous edge
  # use the count c to avoid comparing nodes (may not be able to)
  c = count()
  fringe = []
  prev_edge_type = None
  run_id_till_node_u = None
  prev_edge_cost = 0
  edges_cost_till_u = 0
  zone_at_start_of_pt_trip = None

  for source in sources:
    if source not in G:
      raise nx.NodeNotFound("Source {} not in G".format(source))
    seen[source] = 0
    current_time = time_of_request
    push(fringe, (current_time, next(c), source, prev_edge_type, run_id_till_node_u, current_time, prev_edge_cost, edges_cost_till_u, zone_at_start_of_pt_trip))                      # LY: added initial time of request as argument and prev_edge_type
  while fringe:
    (d, _, v, in_vehicle_tt, wait_time, walking_tt, distance, cost, num_line_trfs, num_mode_trsfs, pr_ed_tp, last_run_id, curr_time, pr_e_cost, cost_t_u, pt_trip_start_zone) = pop(fringe)
    if v in node_labels:
      continue  # already searched this node.
    node_labels[v]['weight'] = d
    node_labels[v]['in_vehicle_tt'] = in_vehicle_tt
    node_labels[v]['wait_time'] = wait_time
    node_labels[v]['walk_tt'] = walking_tt
    node_labels[v]['distance'] = distance
    node_labels[v]['cost'] = cost
    node_labels[v]['line_trasnf_num'] = num_line_trfs
    node_labels[v]['mode_transf_num'] = num_mode_trsfs
    node_labels[v]['current_time'] = time_till_u
    node_labels[v]['prev_edge_type'] = pr_ed_tp
    node_labels[v]['last_pt_veh_run_id'] = last_run_id
    node_labels[v]['previous_edge_cost'] = previous_edge_cost
    node_labels[v]['pt_trip_start_zone'] = pt_trip_start_zone
    # nodes_labels_dict[v]['weight'] = d
    # nodes_labels_dict[v][]
    if v == target:
      break
    # assign
    for u, e in G_succ[v].items():
      time_till_u = curr_time
      previous_edge_cost = pr_e_cost
      cost_till_u = cost_t_u
      e_type = edge_type(v, u, e)
      if e_type == 'walk_edge':
        e_in_veh_tt = in_vehicle_tt(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance(v, u, e)        # funtion that extracts the edge's distance attribute
        if e_distance is None:
          print('Missing distance value in edge {}'.format((v, u)))
          continue
        e_cost = 0
        e_num_mode_trf = 0
        e_num_lin_trf = 0
        time_till_u = time_till_u + e_in_veh_tt + e_wait_time + e_walking_tt    # we store the time till upstream node and we use it for extracting the correct time-dependent attributes of each edge
        # prev_edge_cost = e_cost
        cost_till_u += e_cost   # we store the cost till updtream node as well
        vu_dist = node_labels[v] + e_in_veh_tt + e_wait_time + e_distance + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt
      if e_type == 'orig_dummy_edge':
        road_edge_cost = weight(v, u, e)
        if road_edge_cost is None:
          continue
        vu_dist = dist[v] + road_edge_cost
      if e_type == 'dest_dummy_edge' or e_type == 'dual_edge':
        road_edge_cost = calc_road_link_tt(dist[v], e)             # the travel time assigned here is the travel time of the corresponding 5min interval based on historic data
        if road_edge_cost is None:
          continue
        vu_dist = dist[v] + road_edge_cost
      if e_type == 'access_edge':
        e_in_veh_tt = in_vehicle_tt(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance(v, u, e)        # funtion that extracts the edge's distance attribute
        if e_distance is None:
          print('Missing distance value in edge {}'.format((v, u)))
          continue
        e_cost = 0
        if pr_ed_tp == 'walk_edge':
          e_num_mode_trf = 1
        else:
          e_num_mode_trf = 0
        e_num_lin_trf = 0
        time_till_u = time_till_u + e_in_veh_tt + e_wait_time + e_walking_tt    # we store the time till upstream node and we use it for extracting the correct time-dependent attributes of each edge
        # prev_edge_cost = e_cost
        cost_till_u += e_cost   # we store the cost till updtream node as well
        vu_dist = node_labels[v] + e_in_veh_tt + e_wait_time + e_distance + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt
      # cost calculation process for a transfer edge in bus or train stops/stations
      if e_type == 'pt_transfer_edge':
        # for zone_to_zone pt fare scheme we store the zone of the stop/station in which a pt trip started (origin); this zone will be used for the calculcation of the edge cost based on which pt stop the algorithm checks and hence the final stop of the pt trip
        if fare_scheme == 'zone_to_zone' and pr_ed_tp != 'pt_transfer_edge' and pr_ed_tp != 'pt_route_edge':
          zone_at_start_of_pt_trip = G.nodes[v]['zone']
          previous_edge_cost = 0
        e_in_veh_tt = in_vehicle_tt(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance(v, u, e)        # funtion that extracts the edge's distance attribute
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
        time_till_u = time_till_u + e_in_veh_tt + e_wait_time + e_walking_tt    # we store the time till upstream node and we use it for extracting the correct time-dependent attributes of each edge
        # prev_edge_cost = e_cost
        cost_till_u += e_cost   # we store the cost till updtream node as well
        vu_dist = node_labels[v] + e_in_veh_tt + e_wait_time + e_cost + 10000 * e_num_lin_trf + e_num_mode_trf + e_walking_tt  # + e_distance
      # cost calculation process for a pt route edge in bus or train stops/stations
      if e_type == 'pt_route_edge':
        # for pt route edges the waiting time and travel time is calculated differently; based on the time-dependent model and the FIFO assumption, if the type of previous edge is transfer edge, we assume that the fastest trip will be the one with the first departing bus/train after the current time (less waiting time) and the travel time will be the one of the corresponding pt vehicle run_id; but if the type of the previous edge is a route edge, then for this line/route a pt_vehcile has already been chosen and the edge travel time will be the one for this specific vehicle of the train/bus line (in this case the wait time is 0)
        if pr_ed_tp == 'pt_transfer_edge':
          dep_timetable = timetable(v, u, e)  # fuction that extracts the stop's/station's timetable dict
          if dep_timetable is None:
            print('Missing timetable value in edge'.format((v, u)))
            continue
          e_wait_time, pt_vehicle_run_id = calc_plat_wait_time_and_train_id(time_till_u, dep_timetable)  # function that extracts waiting time for next pt vehicle and the vehicle_id; the next departing vehicle is being found using a binary search algorithm that operates on a sorted list of the deparure times for this edge (departure times of the downstream stop/station)
          if e_wait_time is None:
            print('Missing wait_time value in edge'.format((v, u)))
            continue
          in_vehicle_tt_data = in_vehicle_tt(v, u, e)  # fuction that extracts the travel time dict
          if in_vehicle_tt_data is None:
            print('Missing in_veh_tt value in edge'.format((v, u)))
            continue
          e_in_veh_tt = calc_pt_route_edge_in_veh_tt_for_run_id(in_vehicle_tt_data, pt_vehicle_run_id)  # fuction that travel time for corresponding pt vehicle run_id
        elif pr_ed_tp == 'pt_route_edge':
          e_in_veh_tt = e['departure_time'][last_run_id] - time_till_u + e['in_vehicle_tt'][last_run_id]  # the subtraction fo the first two terms is the dwell time in the downstream station and the 3rd term is the travel time of the pt vehicle run_id that has been selected for the previous route edge
          if e_in_veh_tt is None:
            print('Missing in_veh_tt value in edge'.format((v, u)))
            continue
          e_wait_time = 0
        e_distance = distance(v, u, e)  # fuction that extracts the travel time dict
        if e_distance is None:
          print('Missing distance value in edge'.format((v, u)))
          continue
        # edge costs for pt depend on the pt fare scheme; if it is additive (distance_based) or zone_to_zone !! consider adding a price cap !!
        if fare_scheme == 'distance_based':
          dist_bas_cost = distance_based_cost(v, u, e)  # fuction that extracts the time-dependent distance-based cost dict
          if dist_bas_cost is None:
            print('Missing dist_bas_cost value in edge'.format((v, u)))
            continue
          e_cost = calc_time_dep_distance_based_cost(dist_bas_cost, time_till_u)  # fuction that extracts the cost based on time-dependent distance-based cost dict and the current time (finds in which time-zone we currently are)
        if fare_scheme == 'zone_to_zone':
          zn_to_zn_cost = zone_to_zone_cost(v, u, e)  # fuction that extracts the time-dependent zone_to_zone cost dict
          if zn_to_zn_cost is None:
            print('Missing zn_to_zn_cost value in edge'.format((v, u)))
            continue
          pt_cur_cost = calc_time_dep_zone_to_zone_cost(zn_to_zn_cost, time_till_u, zone_at_start_of_pt_trip, G.nodes[u]['zone'])  # function that extracts the cost of the edge based on the zone at the start of the pt trip, the zone of current stop/station and the current time we are in
          e_cost = pt_cur_cost
        e_num_lin_trf = 0
        e_num_mode_trf = 0
        e_walking_tt = walk_tt(v, u, e)  # fuction that extracts the walking travel time
        if e_walking_tt is None:
          print('Missing walking_tt value in edge'.format((v, u)))
          continue
        time_till_u = time_till_u + e_in_veh_tt + e_wait_time + e_walking_tt  # we store the time till upstream node and we use it for extracting the correct time-dependent attributes of each edge
        cost_till_u += e_cost - previous_edge_cost  # we store the cost till updtream node as well
        vu_dist = node_labels[v] + e_in_veh_tt + e_wait_time + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt - previous_edge_cost  # + e_distance # in the case of pt route edges we always subtract the previous edge cost, this way, in case the algorithms checks a successive pt stop/station that is of different zone we add the new cost and remove the previous one; if we have a distance-based cost then the previous edge cost will always be zero
        if fare_scheme == 'zone_to_zone':
          previous_edge_cost = pt_cur_cost  # here only for the case of zone_to_zone pt fare schemes we update the previous edge cost only after the label (edge weight) calculation
      if cutoff is not None:
        if vu_dist > cutoff:
          continue
      if u in node_labels:
        if vu_dist < node_labels[u]:
          raise ValueError('Contradictory paths found:',
                           'negative weights?')
      elif u not in seen or vu_dist < seen[u]:
        seen[u] = vu_dist
        if e_type == 'pt_route_edge' and pr_ed_tp != 'pt_route_edge':
          push(fringe, (vu_dist, next(c), u, e_in_veh_tt, e_wait_time, e_walking_tt, e_distance, e_cost, e_num_lin_trf, e_num_mode_trf, e_type, pt_vehicle_run_id, time_till_u, previous_edge_cost, cost_till_u, zone_at_start_of_pt_trip))
        elif e_type == 'pt_route_edge' and pr_ed_tp == 'pt_route_edge':
          push(fringe, (vu_dist, next(c), u, e_in_veh_tt, e_wait_time, e_walking_tt, e_distance, e_cost, e_num_lin_trf, e_num_mode_trf, e_type, last_run_id, time_till_u, previous_edge_cost, cost_till_u, zone_at_start_of_pt_trip))
        elif e_type == 'pt_transfer_edge':
          push(fringe, (vu_dist, next(c), u, e_in_veh_tt, e_wait_time, e_walking_tt, e_distance, e_cost, e_num_lin_trf, e_num_mode_trf, e_type, None, time_till_u, previous_edge_cost, cost_till_u, zone_at_start_of_pt_trip))
        elif e_type != 'pt_route_edge' and e_type != 'pt_transfer_edge':
          push(fringe, (vu_dist, next(c), u, e_in_veh_tt, e_wait_time, e_walking_tt, e_distance, e_cost, e_num_lin_trf, e_num_mode_trf, e_type, None, time_till_u, previous_edge_cost, cost_till_u, zone_at_start_of_pt_trip))
        if paths is not None:
          paths[u] = paths[v] + [u]
        if pred is not None:
          pred[u] = [v]
      elif vu_dist == seen[u]:
        if pred is not None:
          pred[u].append(v)

  # The optional predecessor and path dictionaries can be accessed
  # by the caller via the pred and paths objects passed as arguments.
  return node_labels


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight='weight'):
  """Compute weighted shortest path length and predecessors.

  Uses Dijkstra's Method to obtain the shortest weighted paths
  and return dictionaries of predecessors for each node and
  distance for each node from the `source`.

  Parameters
  ----------
  G : NetworkX graph

  source : node label
     Starting node for path

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  pred, distance : dictionaries
     Returns two dictionaries representing a list of predecessors
     of a node and the distance to each node.
     Warning: If target is specified, the dicts are incomplete as they
     only contain information for the nodes along a path to target.

  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The list of predecessors contains more than one element only when
  there are more than one shortest paths to the key node.

  Examples
  --------
  >>> import networkx as nx
  >>> G = nx.path_graph(5, create_using = nx.DiGraph())
  >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
  >>> sorted(pred.items())
  [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
  >>> sorted(dist.items())
  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

  >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0, 1)
  >>> sorted(pred.items())
  [(0, []), (1, [0])]
  >>> sorted(dist.items())
  [(0, 0), (1, 1)]
  """

  weight = _weight_function(G, weight)
  pred = {source: []}  # dictionary of predecessors
  return (pred, _dijkstra(G, source, weight, pred=pred, cutoff=cutoff))


def all_pairs_dijkstra(G, cutoff=None, weight='weight'):
  """Find shortest weighted paths and lengths between all nodes.

  Parameters
  ----------
  G : NetworkX graph

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Yields
  ------
  (node, (distance, path)) : (node obj, (dict, dict))
      Each source node has two associated dicts. The first holds distance
      keyed by target and the second holds paths keyed by target.
      (See single_source_dijkstra for the source/target node terminology.)
      If desired you can apply `dict()` to this function to create a dict
      keyed by source node to the two dicts.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> len_path = dict(nx.all_pairs_dijkstra(G))
  >>> print(len_path[3][0][1])
  2
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('3 - {}: {}'.format(node, len_path[3][0][node]))
  3 - 0: 3
  3 - 1: 2
  3 - 2: 1
  3 - 3: 0
  3 - 4: 1
  >>> len_path[3][1][1]
  [3, 2, 1]
  >>> for n, (dist, path) in nx.all_pairs_dijkstra(G):
  ...     print(path[1])
  [0, 1]
  [1]
  [2, 1]
  [3, 2, 1]
  [4, 3, 2, 1]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The yielded dicts only have keys for reachable nodes.
  """
  for n in G:
    dist, path = single_source_dijkstra(G, n, cutoff=cutoff, weight=weight)
    yield (n, (dist, path))


def all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight'):
  """Compute shortest path lengths between all nodes in a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  distance : iterator
      (source, dictionary) iterator with dictionary keyed by target and
      shortest path length as the key value.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length = dict(nx.all_pairs_dijkstra_path_length(G))
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('1 - {}: {}'.format(node, length[1][node]))
  1 - 0: 1
  1 - 1: 0
  1 - 2: 1
  1 - 3: 2
  1 - 4: 3
  >>> length[3][2]
  1
  >>> length[2][2]
  0

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  The dictionary returned only has keys for reachable node pairs.
  """
  length = single_source_dijkstra_path_length
  for n in G:
    yield (n, length(G, n, cutoff=cutoff, weight=weight))


def all_pairs_dijkstra_path(G, cutoff=None, weight='weight'):
  """Compute shortest paths between all nodes in a weighted graph.

  Parameters
  ----------
  G : NetworkX graph

  cutoff : integer or float, optional
     Depth to stop the search. Only return paths with length <= cutoff.

  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.

     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.

  Returns
  -------
  distance : dictionary
     Dictionary, keyed by source and target, of shortest paths.

  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> path = dict(nx.all_pairs_dijkstra_path(G))
  >>> print(path[0][4])
  [0, 1, 2, 3, 4]

  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.

  See Also
  --------
  floyd_warshall(), all_pairs_bellman_ford_path()

  """
  path = single_source_dijkstra_path
  # TODO This can be trivially parallelized.
  for n in G:
    yield (n, path(G, n, cutoff=cutoff, weight=weight))


def bellman_ford_predecessor_and_distance(G, source, target=None,
                                          weight='weight'):
  """Compute shortest path lengths and predecessors on shortest paths
  in weighted graphs.
  The algorithm has a running time of $O(mn)$ where $n$ is the number of
  nodes and $m$ is the number of edges.  It is slower than Dijkstra but
  can handle negative edge weights.
  Parameters
  ----------
  G : NetworkX graph
     The algorithm works for all types of graphs, including directed
     graphs and multigraphs.
  source: node label
     Starting node for path
  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.
     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.
  Returns
  -------
  pred, dist : dictionaries
     Returns two dictionaries keyed by node to predecessor in the
     path and to the distance from the source respectively.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  NetworkXUnbounded
     If the (di)graph contains a negative cost (di)cycle, the
     algorithm raises an exception to indicate the presence of the
     negative cost (di)cycle.  Note: any negative weight edge in an
     undirected graph is a negative cost cycle.
  Examples
  --------
  >>> import networkx as nx
  >>> G = nx.path_graph(5, create_using = nx.DiGraph())
  >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)
  >>> sorted(pred.items())
  [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
  >>> sorted(dist.items())
  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
  >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0, 1)
  >>> sorted(pred.items())
  [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]
  >>> sorted(dist.items())
  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
  >>> from nose.tools import assert_raises
  >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
  >>> G[1][2]['weight'] = -7
  >>> assert_raises(nx.NetworkXUnbounded, \
                    nx.bellman_ford_predecessor_and_distance, G, 0)
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  The dictionaries returned only have keys for nodes reachable from
  the source.
  In the case where the (di)graph is not connected, if a component
  not containing the source contains a negative cost (di)cycle, it
  will not be detected.
  In NetworkX v2.1 and prior, the source node had predecessor `[None]`.
  In NetworkX v2.2 this changed to the source node having predecessor `[]`
  """
  if source not in G:
    raise nx.NodeNotFound("Node %s is not found in the graph" % source)
  weight = _weight_function(G, weight)
  if any(weight(u, v, d) < 0 for u, v, d in nx.selfloop_edges(G, data=True)):
    raise nx.NetworkXUnbounded("Negative cost cycle detected.")

  dist = {source: 0}
  pred = {source: []}

  if len(G) == 1:
    return pred, dist

  weight = _weight_function(G, weight)

  dist = _bellman_ford(G, [source], weight, pred=pred, dist=dist,
                       target=target)
  return (pred, dist)


def _bellman_ford(G, source, weight, pred=None, paths=None, dist=None,
                  target=None):
  """Relaxation loop for Bellman–Ford algorithm.
  This is an implementation of the SPFA variant.
  See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm
  Parameters
  ----------
  G : NetworkX graph
  source: list
      List of source nodes. The shortest path from any of the source
      nodes will be found if multiple sources are provided.
  weight : function
      The weight of an edge is the value returned by the function. The
      function must accept exactly three positional arguments: the two
      endpoints of an edge and the dictionary of edge attributes for
      that edge. The function must return a number.
  pred: dict of lists, optional (default=None)
      dict to store a list of predecessors keyed by that node
      If None, predecessors are not stored
  paths: dict, optional (default=None)
      dict to store the path list from source to each node, keyed by node
      If None, paths are not stored
  dist: dict, optional (default=None)
      dict to store distance from source to the keyed node
      If None, returned dist dict contents default to 0 for every node in the
      source list
  target: node label, optional
      Ending node for path. Path lengths to other destinations may (and
      probably will) be incorrect.
  Returns
  -------
  Returns a dict keyed by node to the distance from the source.
  Dicts for paths and pred are in the mutated input dicts by those names.
  Raises
  ------
  NodeNotFound
      If any of `source` is not in `G`.
  NetworkXUnbounded
     If the (di)graph contains a negative cost (di)cycle, the
     algorithm raises an exception to indicate the presence of the
     negative cost (di)cycle.  Note: any negative weight edge in an
     undirected graph is a negative cost cycle
  """
  for s in source:
    if s not in G:
      raise nx.NodeNotFound("Source {} not in G".format(s))

  if pred is None:
    pred = {v: [] for v in source}

  if dist is None:
    dist = {v: 0 for v in source}

  G_succ = G.succ if G.is_directed() else G.adj
  inf = float('inf')
  n = len(G)

  count = {}
  q = deque(source)
  in_q = set(source)
  while q:
    u = q.popleft()
    in_q.remove(u)

    # Skip relaxations if any of the predecessors of u is in the queue.
    if all(pred_u not in in_q for pred_u in pred[u]):
      dist_u = dist[u]
      for v, e in G_succ[u].items():
        dist_v = dist_u + weight(v, u, e)

        if dist_v < dist.get(v, inf):
          if v not in in_q:
            q.append(v)
            in_q.add(v)
            count_v = count.get(v, 0) + 1
            if count_v == n:
              raise nx.NetworkXUnbounded(
                  "Negative cost cycle detected.")
            count[v] = count_v
          dist[v] = dist_v
          pred[v] = [u]

        elif dist.get(v) is not None and dist_v == dist.get(v):
          pred[v].append(u)

  if paths is not None:
    dsts = [target] if target is not None else pred
    for dst in dsts:

      path = [dst]
      cur = dst

      while pred[cur]:
        cur = pred[cur][0]
        path.append(cur)

      path.reverse()
      paths[dst] = path

  return dist


def bellman_ford_path(G, source, target, weight='weight'):
  """Returns the shortest path from source to target in a weighted graph G.
  Parameters
  ----------
  G : NetworkX graph
  source : node
     Starting node
  target : node
     Ending node
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight
  Returns
  -------
  path : list
     List of nodes in a shortest path.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  NetworkXNoPath
     If no path exists between source and target.
  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> print(nx.bellman_ford_path(G, 0, 4))
  [0, 1, 2, 3, 4]
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  dijkstra_path(), bellman_ford_path_length()
  """
  length, path = single_source_bellman_ford(G, source,
                                            target=target, weight=weight)
  return path


def bellman_ford_path_length(G, source, target, weight='weight'):
  """Returns the shortest path length from source to target
  in a weighted graph.
  Parameters
  ----------
  G : NetworkX graph
  source : node label
     starting node for path
  target : node label
     ending node for path
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight
  Returns
  -------
  length : number
      Shortest path length.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  NetworkXNoPath
      If no path exists between source and target.
  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> print(nx.bellman_ford_path_length(G,0,4))
  4
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  dijkstra_path_length(), bellman_ford_path()
  """
  if source == target:
    return 0

  weight = _weight_function(G, weight)

  length = _bellman_ford(G, [source], weight, target=target)

  try:
    return length[target]
  except KeyError:
    raise nx.NetworkXNoPath(
        "node %s not reachable from %s" % (source, target))


def single_source_bellman_ford_path(G, source, weight='weight'):
  """Compute shortest path between source and all other reachable
  nodes for a weighted graph.
  Parameters
  ----------
  G : NetworkX graph
  source : node
     Starting node for path.
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight
  Returns
  -------
  paths : dictionary
     Dictionary of shortest path lengths keyed by target.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  Examples
  --------
  >>> G=nx.path_graph(5)
  >>> path=nx.single_source_bellman_ford_path(G,0)
  >>> path[4]
  [0, 1, 2, 3, 4]
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  single_source_dijkstra(), single_source_bellman_ford()
  """
  (length, path) = single_source_bellman_ford(
      G, source, weight=weight)
  return path


def single_source_bellman_ford_path_length(G, source, weight='weight'):
  """Compute the shortest path length between source and all other
  reachable nodes for a weighted graph.
  Parameters
  ----------
  G : NetworkX graph
  source : node label
     Starting node for path
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight.
  Returns
  -------
  length : iterator
      (target, shortest path length) iterator
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length = dict(nx.single_source_bellman_ford_path_length(G, 0))
  >>> length[4]
  4
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  single_source_dijkstra(), single_source_bellman_ford()
  """
  weight = _weight_function(G, weight)
  return _bellman_ford(G, [source], weight)


def single_source_bellman_ford(G, source, target=None, weight='weight'):
  """Compute shortest paths and lengths in a weighted graph G.
  Uses Bellman-Ford algorithm for shortest paths.
  Parameters
  ----------
  G : NetworkX graph
  source : node label
     Starting node for path
  target : node label, optional
     Ending node for path
  Returns
  -------
  distance, path : pair of dictionaries, or numeric and list
     If target is None, returns a tuple of two dictionaries keyed by node.
     The first dictionary stores distance from one of the source nodes.
     The second stores the path from one of the sources to that node.
     If target is not None, returns a tuple of (distance, path) where
     distance is the distance from source to target and path is a list
     representing the path from source to target.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length, path = nx.single_source_bellman_ford(G, 0)
  >>> print(length[4])
  4
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('{}: {}'.format(node, length[node]))
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  >>> path[4]
  [0, 1, 2, 3, 4]
  >>> length, path = nx.single_source_bellman_ford(G, 0, 1)
  >>> length
  1
  >>> path
  [0, 1]
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  single_source_dijkstra()
  single_source_bellman_ford_path()
  single_source_bellman_ford_path_length()
  """
  if source == target:
    return (0, [source])

  weight = _weight_function(G, weight)

  paths = {source: [source]}  # dictionary of paths
  dist = _bellman_ford(G, [source], weight, paths=paths, target=target)
  if target is None:
    return (dist, paths)
  try:
    return (dist[target], paths[target])
  except KeyError:
    msg = "Node %s not reachable from %s" % (source, target)
    raise nx.NetworkXNoPath(msg)


def all_pairs_bellman_ford_path_length(G, weight='weight'):
  """ Compute shortest path lengths between all nodes in a weighted graph.
  Parameters
  ----------
  G : NetworkX graph
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight
  Returns
  -------
  distance : iterator
      (source, dictionary) iterator with dictionary keyed by target and
      shortest path length as the key value.
  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length = dict(nx.all_pairs_bellman_ford_path_length(G))
  >>> for node in [0, 1, 2, 3, 4]:
  ...     print('1 - {}: {}'.format(node, length[1][node]))
  1 - 0: 1
  1 - 1: 0
  1 - 2: 1
  1 - 3: 2
  1 - 4: 3
  >>> length[3][2]
  1
  >>> length[2][2]
  0
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  The dictionary returned only has keys for reachable node pairs.
  """
  length = single_source_bellman_ford_path_length
  for n in G:
    yield (n, dict(length(G, n, weight=weight)))


def all_pairs_bellman_ford_path(G, weight='weight'):
  """ Compute shortest paths between all nodes in a weighted graph.
  Parameters
  ----------
  G : NetworkX graph
  weight: string, optional (default='weight')
     Edge data key corresponding to the edge weight
  Returns
  -------
  distance : dictionary
     Dictionary, keyed by source and target, of shortest paths.
  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> path = dict(nx.all_pairs_bellman_ford_path(G))
  >>> print(path[0][4])
  [0, 1, 2, 3, 4]
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  See Also
  --------
  floyd_warshall(), all_pairs_dijkstra_path()
  """
  path = single_source_bellman_ford_path
  # TODO This can be trivially parallelized.
  for n in G:
    yield (n, path(G, n, weight=weight))


def goldberg_radzik(G, source, weight='weight'):
  """Compute shortest path lengths and predecessors on shortest paths
  in weighted graphs.
  The algorithm has a running time of $O(mn)$ where $n$ is the number of
  nodes and $m$ is the number of edges.  It is slower than Dijkstra but
  can handle negative edge weights.
  Parameters
  ----------
  G : NetworkX graph
     The algorithm works for all types of graphs, including directed
     graphs and multigraphs.
  source: node label
     Starting node for path
  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.
     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.
  Returns
  -------
  pred, dist : dictionaries
     Returns two dictionaries keyed by node to predecessor in the
     path and to the distance from the source respectively.
  Raises
  ------
  NodeNotFound
      If `source` is not in `G`.
  NetworkXUnbounded
     If the (di)graph contains a negative cost (di)cycle, the
     algorithm raises an exception to indicate the presence of the
     negative cost (di)cycle.  Note: any negative weight edge in an
     undirected graph is a negative cost cycle.
  Examples
  --------
  >>> import networkx as nx
  >>> G = nx.path_graph(5, create_using = nx.DiGraph())
  >>> pred, dist = nx.goldberg_radzik(G, 0)
  >>> sorted(pred.items())
  [(0, None), (1, 0), (2, 1), (3, 2), (4, 3)]
  >>> sorted(dist.items())
  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
  >>> from nose.tools import assert_raises
  >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
  >>> G[1][2]['weight'] = -7
  >>> assert_raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, 0)
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  The dictionaries returned only have keys for nodes reachable from
  the source.
  In the case where the (di)graph is not connected, if a component
  not containing the source contains a negative cost (di)cycle, it
  will not be detected.
  """
  if source not in G:
    raise nx.NodeNotFound("Node %s is not found in the graph" % source)
  weight = _weight_function(G, weight)
  if any(weight(u, v, d) < 0 for u, v, d in nx.selfloop_edges(G, data=True)):
    raise nx.NetworkXUnbounded("Negative cost cycle detected.")

  if len(G) == 1:
    return {source: None}, {source: 0}

  if G.is_directed():
    G_succ = G.succ
  else:
    G_succ = G.adj

  inf = float('inf')
  d = {u: inf for u in G}
  d[source] = 0
  pred = {source: None}

  def topo_sort(relabeled):
    """Topologically sort nodes relabeled in the previous round and detect
    negative cycles.
    """
    # List of nodes to scan in this round. Denoted by A in Goldberg and
    # Radzik's paper.
    to_scan = []
    # In the DFS in the loop below, neg_count records for each node the
    # number of edges of negative reduced costs on the path from a DFS root
    # to the node in the DFS forest. The reduced cost of an edge (u, v) is
    # defined as d[u] + weight[u][v] - d[v].
    #
    # neg_count also doubles as the DFS visit marker array.
    neg_count = {}
    for u in relabeled:
      # Skip visited nodes.
      if u in neg_count:
        continue
      d_u = d[u]
      # Skip nodes without out-edges of negative reduced costs.
      if all(d_u + weight(u, v, e) >= d[v]
             for v, e in G_succ[u].items()):
        continue
      # Nonrecursive DFS that inserts nodes reachable from u via edges of
      # nonpositive reduced costs into to_scan in (reverse) topological
      # order.
      stack = [(u, iter(G_succ[u].items()))]
      in_stack = set([u])
      neg_count[u] = 0
      while stack:
        u, it = stack[-1]
        try:
          v, e = next(it)
        except StopIteration:
          to_scan.append(u)
          stack.pop()
          in_stack.remove(u)
          continue
        t = d[u] + weight(u, v, e)
        d_v = d[v]
        if t <= d_v:
          is_neg = t < d_v
          d[v] = t
          pred[v] = u
          if v not in neg_count:
            neg_count[v] = neg_count[u] + int(is_neg)
            stack.append((v, iter(G_succ[v].items())))
            in_stack.add(v)
          elif (v in in_stack and
                neg_count[u] + int(is_neg) > neg_count[v]):
            # (u, v) is a back edge, and the cycle formed by the
            # path v to u and (u, v) contains at least one edge of
            # negative reduced cost. The cycle must be of negative
            # cost.
            raise nx.NetworkXUnbounded(
                'Negative cost cycle detected.')
    to_scan.reverse()
    return to_scan

  def relax(to_scan):
    """Relax out-edges of relabeled nodes.
    """
    relabeled = set()
    # Scan nodes in to_scan in topological order and relax incident
    # out-edges. Add the relabled nodes to labeled.
    for u in to_scan:
      d_u = d[u]
      for v, e in G_succ[u].items():
        w_e = weight(u, v, e)
        if d_u + w_e < d[v]:
          d[v] = d_u + w_e
          pred[v] = u
          relabeled.add(v)
    return relabeled

  # Set of nodes relabled in the last round of scan operations. Denoted by B
  # in Goldberg and Radzik's paper.
  relabeled = set([source])

  while relabeled:
    to_scan = topo_sort(relabeled)
    relabeled = relax(to_scan)

  d = {u: d[u] for u in pred}
  return pred, d


def negative_edge_cycle(G, weight='weight'):
  """Returns True if there exists a negative edge cycle anywhere in G.
  Parameters
  ----------
  G : NetworkX graph
  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.
     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.
  Returns
  -------
  negative_cycle : bool
      True if a negative edge cycle exists, otherwise False.
  Examples
  --------
  >>> import networkx as nx
  >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
  >>> print(nx.negative_edge_cycle(G))
  False
  >>> G[1][2]['weight'] = -7
  >>> print(nx.negative_edge_cycle(G))
  True
  Notes
  -----
  Edge weight attributes must be numerical.
  Distances are calculated as sums of weighted edges traversed.
  This algorithm uses bellman_ford_predecessor_and_distance() but finds
  negative cycles on any component by first adding a new node connected to
  every node, and starting bellman_ford_predecessor_and_distance on that
  node.  It then removes that extra node.
  """
  newnode = generate_unique_node()
  G.add_edges_from([(newnode, n) for n in G])

  try:
    bellman_ford_predecessor_and_distance(G, newnode, weight)
  except nx.NetworkXUnbounded:
    return True
  finally:
    G.remove_node(newnode)
  return False


def bidirectional_dijkstra(G, source, target, weight='weight'):
  r"""Dijkstra's algorithm for shortest paths using bidirectional search.
  Parameters
  ----------
  G : NetworkX graph
  source : node
     Starting node.
  target : node
     Ending node.
  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.
     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.
  Returns
  -------
  length, path : number and list
     length is the distance from source to target.
     path is a list of nodes on a path from source to target.
  Raises
  ------
  NodeNotFound
      If either `source` or `target` is not in `G`.
  NetworkXNoPath
      If no path exists between source and target.
  Examples
  --------
  >>> G = nx.path_graph(5)
  >>> length, path = nx.bidirectional_dijkstra(G, 0, 4)
  >>> print(length)
  4
  >>> print(path)
  [0, 1, 2, 3, 4]
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
  this radius. Volume of the first sphere is `\pi*r*r` while the
  others are `2*\pi*r/2*r/2`, making up half the volume.
  This algorithm is not guaranteed to work if edge weights
  are negative or are floating point numbers
  (overflows and roundoff errors can cause problems).
  See Also
  --------
  shortest_path
  shortest_path_length
  """
  if source not in G or target not in G:
    msg = 'Either source {} or target {} is not in G'
    raise nx.NodeNotFound(msg.format(source, target))

  if source == target:
    return (0, [source])
  push = heappush
  pop = heappop
  # Init:  [Forward, Backward]
  dists = [{}, {}]   # dictionary of final distances
  paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
  fringe = [[], []]  # heap of (distance, node) for choosing node to expand
  seen = [{source: 0}, {target: 0}]  # dict of distances to seen nodes
  c = count()
  # initialize fringe heap
  push(fringe[0], (0, next(c), source))
  push(fringe[1], (0, next(c), target))
  # neighs for extracting correct neighbor information
  if G.is_directed():
    neighs = [G.successors, G.predecessors]
  else:
    neighs = [G.neighbors, G.neighbors]
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


def johnson(G, weight='weight'):
  r"""Uses Johnson's Algorithm to compute shortest paths.
  Johnson's Algorithm finds a shortest path between each pair of
  nodes in a weighted graph even if negative weights are present.
  Parameters
  ----------
  G : NetworkX graph
  weight : string or function
     If this is a string, then edge weights will be accessed via the
     edge attribute with this key (that is, the weight of the edge
     joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
     such edge attribute exists, the weight of the edge is assumed to
     be one.
     If this is a function, the weight of an edge is the value
     returned by the function. The function must accept exactly three
     positional arguments: the two endpoints of an edge and the
     dictionary of edge attributes for that edge. The function must
     return a number.
  Returns
  -------
  distance : dictionary
     Dictionary, keyed by source and target, of shortest paths.
  Raises
  ------
  NetworkXError
     If given graph is not weighted.
  Examples
  --------
  >>> import networkx as nx
  >>> graph = nx.DiGraph()
  >>> graph.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5),
  ... ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
  >>> paths = nx.johnson(graph, weight='weight')
  >>> paths['0']['2']
  ['0', '1', '2']
  Notes
  -----
  Johnson's algorithm is suitable even for graphs with negative weights. It
  works by using the Bellman–Ford algorithm to compute a transformation of
  the input graph that removes all negative weights, allowing Dijkstra's
  algorithm to be used on the transformed graph.
  The time complexity of this algorithm is $O(n^2 \log n + n m)$,
  where $n$ is the number of nodes and $m$ the number of edges in the
  graph. For dense graphs, this may be faster than the Floyd–Warshall
  algorithm.
  See Also
  --------
  floyd_warshall_predecessor_and_distance
  floyd_warshall_numpy
  all_pairs_shortest_path
  all_pairs_shortest_path_length
  all_pairs_dijkstra_path
  bellman_ford_predecessor_and_distance
  all_pairs_bellman_ford_path
  all_pairs_bellman_ford_path_length
  """
  if not nx.is_weighted(G, weight=weight):
    raise nx.NetworkXError('Graph is not weighted.')

  dist = {v: 0 for v in G}
  pred = {v: [] for v in G}
  weight = _weight_function(G, weight)

  # Calculate distance of shortest paths
  dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

  # Update the weight function to take into account the Bellman--Ford
  # relaxation distances.
  def new_weight(u, v, d):
    return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

  def dist_path(v):
    paths = {v: [v]}
    _dijkstra(G, v, new_weight, paths=paths)
    return paths

  return {v: dist_path(v) for v in G}


def LY_shortest_path_with_attrs(G, source, target, time_of_request, in_vehicle_tt='in_vehicle_tt', walk_tt='walk_tt', distance='distance', distance_based_cost='distance_based_cost', zone_to_zone_cost='zone_to_zone_cost', timetable='departure_time', edge_type='edge_type', node_type='node_type', fare_scheme='distance_based', init_in_vehicle_tt = 0, init_wait_time = 0, init_walking_tt = 0, init_distance = 0, init_cost = 0, init_num_line_trfs = 0, init_num_mode_trfs = 0, last_edge_type=None, last_pt_veh_run_id=None, current_time=0, last_edge_cost=0, pt_trip_orig_zone=None, pred=None, paths=None, cutoff=None):

  if source not in G:
    raise nx.NodeNotFound("Source {} not in G".format(source))
  if target not in G:
    raise nx.NodeNotFound("Target {} not in G".format(target))
  if source == target:
    return 0, [target]

  paths = {source: [source]}
  in_vehicle_tt = _get_in_vehicle_tt_function(G, in_vehicle_tt)
  walk_tt = _get_walk_tt(G, walk_tt)
  distance = _get_distance_function(G, distance)
  distance_based_cost = _get_distance_based_cost(G, distance_based_cost)
  zone_to_zone_cost = _get_zone_to_zone_cost(G, zone_to_zone_cost)
  timetable = _get_timetable(G, timetable)
  edge_type = _get_edge_type(G, edge_type)
  node_type = _get_node_type(G, node_type)

  return _LY_dijkstra(G, source, target, time_of_request, in_vehicle_tt, walk_tt, distance, distance_based_cost, zone_to_zone_cost, timetable, edge_type, node_type, fare_scheme, init_in_vehicle_tt, init_wait_time, init_walking_tt, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, last_edge_type, last_pt_veh_run_id, current_time, last_edge_cost, pt_trip_orig_zone, pred=None, paths=paths, cutoff=None)



def _LY_dijkstra(G, source, target, time_of_request, in_vehicle_tt_data, walk_tt_data, distance_data, distance_based_cost_data, zone_to_zone_cost_data, timetable_data, edge_type_data, node_type_data, fare_scheme, init_in_vehicle_tt, init_wait_time, init_walking_tt, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, last_edge_type, last_pt_veh_run_id, current_time, last_edge_cost, pt_trip_orig_zone, pred=None, paths=None, cutoff=None):

  G_succ = G._succ if G.is_directed() else G._adj

  push = heappush
  pop = heappop


  weight_label = {}  # dictionary of best weight
  in_veh_tt_label = {}
  wait_time_label = {}
  walk_tt_label = {}
  distance_label = {}
  mon_cost_label = {}
  line_trans_num_label = {}
  mode_trans_num_label = {}
  prev_edge_type_label = {}
  last_pt_veh_run_id_label = {}
  current_time_label = {}
  prev_edge_cost_label = {}
  pt_trip_start_zone_label = {}
  seen = {}


  c = count()   # use the count c to avoid comparing nodes (may not be able to)
  fringe = []  # fringe is heapq with 3-tuples (distance,c,node) -and I change it to a 4-tuple to store the type of the previous edge


  prev_edge_type = last_edge_type
  run_id_till_node_u = last_pt_veh_run_id
  previous_edge_cost = last_edge_cost
  zone_at_start_of_pt_trip = pt_trip_orig_zone

  curr_time = time_of_request
  seen[source] = 0
  weight_init = 0

  push(fringe, (weight_init, next(c), source, init_in_vehicle_tt, init_wait_time, init_walking_tt, init_distance, init_cost, init_num_line_trfs, init_num_mode_trfs, prev_edge_type, run_id_till_node_u, curr_time, previous_edge_cost, zone_at_start_of_pt_trip))                      # LY: added initial time of request as argument and prev_edge_type

  while fringe:
    (gen_w, _, v, in_v_tt, w_t, wk_t, d, mon_c, n_l_ts, n_m_ts, pr_ed_tp, lt_run_id, curr_time, pr_e_cost, pt_tr_st_z) = pop(fringe)

    if v in weight_label:
      continue  # already searched this node.

    weight_label[v] = gen_w
    in_veh_tt_label[v] = in_v_tt
    wait_time_label[v] = w_t
    walk_tt_label[v] = wk_t
    distance_label[v] = d
    mon_cost_label[v] = mon_c
    line_trans_num_label[v] = n_l_ts
    mode_trans_num_label[v] = n_m_ts
    prev_edge_type_label[v] = pr_ed_tp
    last_pt_veh_run_id_label[v] = lt_run_id
    current_time_label[v] = curr_time
    prev_edge_cost_label[v] = pr_e_cost
    pt_trip_start_zone_label[v] = pt_tr_st_z

    if v == target:
      break

    for u, e in G_succ[v].items():
      e_type = edge_type_data(v, u, e)
      n_type = node_type_data(v, u, e)
      if e_type == 'walk_edge':
        e_in_veh_tt = in_vehicle_tt_data(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt_data(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance_data(v, u, e)        # funtion that extracts the edge's distance attribute
        if e_distance is None:
          print('Missing distance value in edge {}'.format((v, u)))
          continue
        e_cost = 0
        e_num_mode_trf = 0
        e_num_lin_trf = 0

        in_veh_tt_till_u = in_veh_tt_label[v] + e_in_veh_tt
        wait_time_till_u = wait_time_label[v] + e_wait_time
        walk_tt_till_u = walk_tt_label[v] + e_walking_tt
        distance_till_u = distance_label[v] + e_distance
        cost_till_u = mon_cost_label[v] + e_cost
        line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
        mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

        time_till_u = curr_time + e_in_veh_tt + e_wait_time + e_walking_tt

        vu_dist = weight_label[v] + e_in_veh_tt + e_wait_time + e_distance + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt
      # if e_type == 'orig_dummy_edge':
      #   road_edge_cost = weight(v, u, e)
      #   if road_edge_cost is None:
      #     continue
      #   vu_dist = dist[v] + road_edge_cost
      # if e_type == 'dest_dummy_edge' or e_type == 'dual_edge':
      #   road_edge_cost = calc_road_link_tt(dist[v], e)             # the travel time assigned here is the travel time of the corresponding 5min interval based on historic data
      #   if road_edge_cost is None:
      #     continue
      #   vu_dist = dist[v] + road_edge_cost
      if e_type == 'access_edge':
        e_in_veh_tt = in_vehicle_tt_data(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt_data(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance_data(v, u, e)        # funtion that extracts the edge's distance attribute
        if e_distance is None:
          print('Missing distance value in edge {}'.format((v, u)))
          continue
        e_cost = 0
        if n_type == 'walk_graph_node':
          e_num_mode_trf = 1
        else:
          e_num_mode_trf = 0
        e_num_lin_trf = 0

        time_till_u = curr_time + e_in_veh_tt + e_wait_time + e_walking_tt

        in_veh_tt_till_u = in_veh_tt_label[v] + e_in_veh_tt
        wait_time_till_u = wait_time_label[v] + e_wait_time
        walk_tt_till_u = walk_tt_label[v] + e_walking_tt
        distance_till_u = distance_label[v] + e_distance
        cost_till_u = mon_cost_label[v] + e_cost
        line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
        mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

        vu_dist = weight_label[v] + e_in_veh_tt + e_wait_time + e_distance + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt
      # cost calculation process for a transfer edge in bus or train stops/stations


      if e_type == 'pt_transfer_edge':
        # for zone_to_zone pt fare scheme we store the zone of the stop/station in which a pt trip started (origin); this zone will be used for the calculcation of the edge cost based on which pt stop the algorithm checks and hence the final stop of the pt trip
        if fare_scheme == 'zone_to_zone' and pr_ed_tp != 'pt_transfer_edge' and pr_ed_tp != 'pt_route_edge':
          zone_at_start_of_pt_trip = G.nodes[v]['zone']
          previous_edge_cost = 0

        e_in_veh_tt = in_vehicle_tt_data(v, u, e)  # funtion that extracts the edge's in-vehicle travel time attribute
        if e_in_veh_tt is None:
          print('Missing in_veh_tt value in edge {}'.format((v, u)))
          continue
        e_wait_time = 0
        e_walking_tt = walk_tt_data(v, u, e)       # funtion that extracts the edge's walking travel time attribute
        if e_walking_tt is None:
          print('Missing walking_tt value in edge {}'.format((v, u)))
          continue
        e_distance = distance_data(v, u, e)        # funtion that extracts the edge's distance attribute
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

        time_till_u = curr_time + e_in_veh_tt + e_wait_time + e_walking_tt

        in_veh_tt_till_u = in_veh_tt_label[v] + e_in_veh_tt
        wait_time_till_u = wait_time_label[v] + e_wait_time
        walk_tt_till_u = walk_tt_label[v] + e_walking_tt
        distance_till_u = distance_label[v] + e_distance
        cost_till_u = mon_cost_label[v] + e_cost
        line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
        mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

        vu_dist = weight_label[v] + e_in_veh_tt + e_wait_time + e_distance + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt + e_distance
      # cost calculation process for a pt route edge in bus or train stops/stations
      if e_type == 'pt_route_edge':
        # for pt route edges the waiting time and travel time is calculated differently; based on the time-dependent model and the FIFO assumption, if the type of previous edge is transfer edge, we assume that the fastest trip will be the one with the first departing bus/train after the current time (less waiting time) and the travel time will be the one of the corresponding pt vehicle run_id; but if the type of the previous edge is a route edge, then for this line/route a pt_vehcile has already been chosen and the edge travel time will be the one for this specific vehicle of the train/bus line (in this case the wait time is 0)

        if pr_ed_tp == 'pt_transfer_edge':
          dep_timetable = timetable_data(v, u, e)  # fuction that extracts the stop's/station's timetable dict
          if dep_timetable is None:
            print('Missing timetable value in edge'.format((v, u)))
            continue
          e_wait_time, pt_vehicle_run_id = calc_plat_wait_time_and_train_id(curr_time, dep_timetable)  # function that extracts waiting time for next pt vehicle and the vehicle_id; the next departing vehicle is being found using a binary search algorithm that operates on a sorted list of the deparure times for this edge (departure times of the downstream stop/station)
          if e_wait_time is None:
            print('Missing wait_time value in edge'.format((v, u)))
            continue
          in_vehicle_tt_d = in_vehicle_tt_data(v, u, e)  # fuction that extracts the travel time dict
          if in_vehicle_tt_d is None:
            print('Missing in_veh_tt value in edge'.format((v, u)))
            continue
          e_in_veh_tt = calc_pt_route_edge_in_veh_tt_for_run_id(in_vehicle_tt_d, pt_vehicle_run_id)  # fuction that travel time for corresponding pt vehicle run_id
        elif pr_ed_tp == 'pt_route_edge':
          e_in_veh_tt = e['departure_time'][lt_run_id] - curr_time + e['in_vehicle_tt'][lt_run_id]  # the subtraction fo the first two terms is the dwell time in the downstream station and the 3rd term is the travel time of the pt vehicle run_id that has been selected for the previous route edge
          if e_in_veh_tt is None:
            print('Missing in_veh_tt value in edge'.format((v, u)))
            continue
          e_wait_time = 0
        e_distance = distance_data(v, u, e)  # fuction that extracts the travel time dict
        if e_distance is None:
          print('Missing distance value in edge'.format((v, u)))
          continue
        # edge costs for pt depend on the pt fare scheme; if it is additive (distance_based) or zone_to_zone !! consider adding a price cap !!
        if fare_scheme == 'distance_based':
          dist_bas_cost = distance_based_cost_data(v, u, e)  # fuction that extracts the time-dependent distance-based cost dict
          if dist_bas_cost is None:
            print('Missing dist_bas_cost value in edge'.format((v, u)))
            continue
          e_cost = calc_time_dep_distance_based_cost(dist_bas_cost, curr_time)  # fuction that extracts the cost based on time-dependent distance-based cost dict and the current time (finds in which time-zone we currently are)
        if fare_scheme == 'zone_to_zone':
          zn_to_zn_cost = zone_to_zone_cost_data(v, u, e)  # fuction that extracts the time-dependent zone_to_zone cost dict
          if zn_to_zn_cost is None:
            print('Missing zn_to_zn_cost value in edge'.format((v, u)))
            continue
          pt_cur_cost = calc_time_dep_zone_to_zone_cost(zn_to_zn_cost, curr_time, pt_tr_st_z, G.nodes[u]['zone'])  # function that extracts the cost of the edge based on the zone at the start of the pt trip, the zone of current stop/station and the current time we are in
          e_cost = pt_cur_cost
        e_num_lin_trf = 0
        e_num_mode_trf = 0
        e_walking_tt = walk_tt_data(v, u, e)  # fuction that extracts the walking travel time
        if e_walking_tt is None:
          print('Missing walking_tt value in edge'.format((v, u)))
          continue

        time_till_u = curr_time + e_in_veh_tt + e_wait_time + e_walking_tt

        in_veh_tt_till_u = in_veh_tt_label[v] + e_in_veh_tt
        wait_time_till_u = wait_time_label[v] + e_wait_time
        walk_tt_till_u = walk_tt_label[v] + e_walking_tt
        distance_till_u = distance_label[v] + e_distance
        cost_till_u = mon_cost_label[v] + e_cost - pr_e_cost
        line_trasnf_num_till_u = line_trans_num_label[v] + e_num_lin_trf
        mode_transf_num_till_u = mode_trans_num_label[v] + e_num_mode_trf

        vu_dist = weight_label[v] + e_in_veh_tt + e_wait_time + e_cost + e_num_lin_trf + e_num_mode_trf + e_walking_tt - pr_e_cost  # + e_distance # in the case of pt route edges we always subtract the previous edge cost, this way, in case the algorithms checks a successive pt stop/station that is of different zone we add the new cost and remove the previous one; if we have a distance-based cost then the previous edge cost will always be zero
        if fare_scheme == 'zone_to_zone':
          previous_edge_cost = pt_cur_cost  # here only for the case of zone_to_zone pt fare schemes we update the previous edge cost only after the label (edge weight) calculation
      if cutoff is not None:
        if vu_dist > cutoff:
          continue
      if u in weight_label:
        if vu_dist < weight_label[u]:
          # print(weight_label, weight_label[u], paths[v])
          print('Negative weight in node {}, in edge {}, {}?'.format(u, v, u))
          raise ValueError('Contradictory paths found:',
                           'negative weights?')
      elif u not in seen or vu_dist < seen[u]:
        seen[u] = vu_dist
        if e_type == 'pt_route_edge' and pr_ed_tp != 'pt_route_edge':
          push(fringe, (vu_dist, next(c), u, in_veh_tt_till_u, wait_time_till_u, walk_tt_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, pt_vehicle_run_id, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip))
        elif e_type == 'pt_route_edge' and pr_ed_tp == 'pt_route_edge':
          push(fringe, (vu_dist, next(c), u, in_veh_tt_till_u, wait_time_till_u, walk_tt_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, lt_run_id, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip))
        elif e_type == 'pt_transfer_edge':
          push(fringe, (vu_dist, next(c), u, in_veh_tt_till_u, wait_time_till_u, walk_tt_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, None, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip))
        elif e_type != 'pt_route_edge' and e_type != 'pt_transfer_edge':
          push(fringe, (vu_dist, next(c), u, in_veh_tt_till_u, wait_time_till_u, walk_tt_till_u, distance_till_u, cost_till_u, line_trasnf_num_till_u, mode_transf_num_till_u, e_type, None, time_till_u, previous_edge_cost, zone_at_start_of_pt_trip))
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
    return (weight_label[target], paths[target], in_veh_tt_label[target], wait_time_label[target], walk_tt_label[target], distance_label[target], mon_cost_label[target], line_trans_num_label[target], mode_trans_num_label[target], in_veh_tt_label, wait_time_label, walk_tt_label, distance_label, mon_cost_label, line_trans_num_label, mode_trans_num_label, weight_label, prev_edge_type_label, last_pt_veh_run_id_label, current_time_label, prev_edge_cost_label, pt_trip_start_zone_label)
  except KeyError:
    raise nx.NetworkXNoPath("Node %s not reachable from %s" % (target, source))
