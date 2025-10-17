# triangulation
Visualizing triangulation

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
The polygon are highlighted in yello edges and filled in
