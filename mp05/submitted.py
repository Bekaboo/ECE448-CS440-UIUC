# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue
import collections
import itertools

Vertex = collections.namedtuple('Vertex', ['estimation', 'distance', 'coord', 'parent_coord'])

def build_path(expanded, start_coord, destination):
    """
    Build the path from the start to the destination.

    @param expanded: A dictionary of all the expanded vertices.
    @param start_coord: The coord of the start vertex.
    @param destination: The destination vertex.

    @return path: a list of tuples containing the coords of each state in the computed path
    """
    path = []
    current = destination
    while current.coord != start_coord:
        path.insert(0, current.coord)
        current = expanded[current.parent_coord]
    # Insert the start coord
    path.insert(0, current.coord)
    return path

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coords of each state in the computed path
    """
    expanded = {}
    frontiers_rec = []
    frontiers = queue.Queue()
    frontiers.put(Vertex(-1, -1, maze.start, maze.start))
    while not frontiers.empty():
        pivot = frontiers.get()
        expanded[pivot.coord] = pivot
        # If the pivot is the waypoint, we have found the path
        if maze[pivot.coord] == maze.legend.waypoint:
            return build_path(expanded, maze.start, pivot)
        # else we expand the pivot
        for coord in maze.neighbors(pivot.coord[0], pivot.coord[1]):
            if coord not in expanded.keys() and \
                    coord not in frontiers_rec:
                frontiers_rec.append(coord)
                frontiers.put(Vertex(-1, -1, coord, pivot.coord))
    return []   # No path found

def manhattan(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    expanded = {}
    frontiers_rec = []
    frontiers = queue.PriorityQueue()
    start = maze.start
    waypoint = maze.waypoints[0]
    frontiers.put(Vertex(manhattan(start, waypoint), 0, start, start))
    while not frontiers.empty():
        pivot = frontiers.get()
        expanded[pivot.coord] = pivot
        # If the pivot is the waypoint, we have found the path
        if maze[pivot.coord] == maze.legend.waypoint:
            return build_path(expanded, maze.start, pivot)
        # else we expand the pivot
        for coord in maze.neighbors(pivot.coord[0], pivot.coord[1]):
            if coord not in expanded.keys() and \
                    coord not in frontiers_rec:
                frontiers_rec.append(coord)
                frontiers.put(Vertex(pivot.distance + 1 + manhattan(coord, waypoint),
                                     pivot.distance + 1,
                                     coord, pivot.coord))
    return []   # No path found

Edge = collections.namedtuple('Edge', ['weight', 'vertex1', 'vertex2'])

class MST:
    def __init__(self, root: tuple[int, int]):
        self.root = root
        self.edges = set()
        self.vertices = {root}
        self.total_weight = 0

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        self.vertices.add(edge.vertex1)
        self.vertices.add(edge.vertex2)
        self.total_weight += edge.weight

    def edge_acceptable(self, edge):
        return edge.vertex1 in self.vertices and \
            edge.vertex2 not in self.vertices or \
            edge.vertex2 in self.vertices and \
            edge.vertex1 not in self.vertices

def mst_prim(coords):
    """
    Generates a minimum spanning tree for vertices in coords using Prim's
    algorithm.

    @param coords: list of vertex coordinates

    @return mst: a list of tuples containing the coordinates of each edge in
    the mst
    """
    candicate_edges = []
    mst = MST(coords[0])
    for vertex1, vertex2 in itertools.combinations(coords, 2):
        candicate_edges.append(Edge(manhattan(vertex1, vertex2), vertex1, vertex2))
    candicate_edges.sort(key=lambda x: x.weight)

    while len(mst.vertices) < len(coords):
        for shortest_edge in candicate_edges:
            if mst.edge_acceptable(shortest_edge):
                mst.add_edge(shortest_edge)
                break

    return mst


# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    def heuristic(coord, waypoints):
        """
        Heuristic function for A* multiple search, returns the Manhattan
        distance from the current coordinate the nearest waypoint plus
        the waypoint MST length for unexplored waypoints.

        @param coord: The current coordinate
        @param waypoint: The current waypoint

        @return h(coord): The heuristic value for the current coordinate
        """
        return min([manhattan(coord, waypoint) for waypoint in waypoints]) + \
            + mst_prim(waypoints).total_weight

    expanded = {}
    frontiers = queue.PriorityQueue()
    start = maze.start
    waypoints_remaining = list(maze.waypoints)
    frontiers.put(Vertex(min([manhattan(start, waypoint)
                              for waypoint in waypoints_remaining]), 0, start, start))
    while not frontiers.empty():
        pivot = frontiers.get()
        expanded[pivot.coord] = pivot
        if pivot.coord in waypoints_remaining:
            waypoints_remaining.remove(pivot.coord)
        # Return if we have found all the waypoints
        if not waypoints_remaining:
            return build_path(expanded, maze.start, pivot)
        # else we expand the pivot
        for neighbor in maze.neighbors(pivot.coord[0], pivot.coord[1]):
            frontiers.put(Vertex(pivot.distance + 1 + heuristic(neighbor, waypoints_remaining),
                                 pivot.distance + 1,
                                 neighbor, pivot.coord))
    return []   # No path found
