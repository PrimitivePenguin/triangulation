# Triangulation
visualizing_delaunay.py - Visualizing triangulation

Visualization of the process of Delaunay triangulation using the Bowyer-Watson algorithm
1. Create a super triangle that encompasses all points
2. For each point in the list do the following
3. Any triangle whose circumcircle contains the point is added to a "bad triangle" list
4. Create a polygon that is contains all the edge of the triangles
5. Remove the  bad triangles
6. Each point in that polygon edge is now connected to the new point
7. Rinse and repeat

Circumcircle is highlighted by a red dotted circle

Bad triangles are highlighted with a red outline

New triangles being inserted are highlighted in green

The polygon are highlighted in yellow edges and filled in


perlin_terrain_v3.py - Applying  Delaunay triangulation
Created a probability map that is based on the difference in gradient, where higher the gradient the more likely it is to get sampled.
Points were sampled using this map, and Delaunay triangulation was applied to the points. There are losses in quality, however it produces a mesh that is relatively accurate to the original perlin noise map.
