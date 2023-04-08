import submitted
import importlib
import maze
maze1 = maze.Maze('data/part-3/open')

importlib.reload(submitted)
path = submitted.astar_multiple(maze1)
print(path)
