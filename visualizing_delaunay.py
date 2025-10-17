import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons, Button, Slider
from matplotlib.animation import FuncAnimation
import random
import time
from collections import defaultdict

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):  
        return f"Point({self.x:.2f}, {self.y:.2f})"
    def __hash__(self):  # needed for sets and dicts
        return hash((round(self.x, 10), round(self.y, 10)))

class Edge:
    def __init__(self, p0, p1):
        if (p0.x, p0.y) < (p1.x, p1.y):
            self.p0, self.p1 = p0, p1
        else:
            self.p0, self.p1 = p1, p0
        # self.p0 = p0
        # self.p1 = p1
    def __eq__(self, other):
        return self.p0 == other.p0 and self.p1 == other.p1
        # return (self.p0 == other.p0 and self.p1 == other.p1) or (self.p0 == other.p1 and self.p1 == other.p0)
    def __repr__(self):
        return f"Edge({self.p0}, {self.p1})"
    def __hash__(self):  # ADDED: needed for set operations
        # Sort points to ensure consistent hash regardless of order
        return hash((self.p0, self.p1))
        # if (self.p0.x, self.p0.y) < (self.p1.x, self.p1.y):
        #     return hash((self.p0, self.p1))
        # else:
        #     return hash((self.p1, self.p0))
        
class Triangle:
    def __init__ (self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.edges = [Edge(p0, p1), Edge(p1, p2), Edge(p2, p0)]  # edges of the triangle    
    
    # circumcenter of p
    def inCircumCircle(self, point):
        ax = self.p0.x - point.x
        ay = self.p0.y - point.y
        bx = self.p1.x - point.x
        by = self.p1.y - point.y
        cx = self.p2.x - point.x
        cy = self.p2.y - point.y
        
        det = (
            (ax * ax + ay * ay) * (bx * cy - cx * by)
            - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay)
        )
        
        ccw = (self.p1.x - self.p0.x) * (self.p2.y - self.p0.y) - (self.p2.x - self.p0.x) * (self.p1.y - self.p0.y)
        
        if ccw > 0:  # Counter-clockwise
            return det > 0
        else:  # Clockwise
            return det < 0
    
    def get_circumcenter_and_radius(self):
        """Calculate circumcenter and radius of the triangle"""
        ax, ay = self.p0.x, self.p0.y
        bx, by = self.p1.x, self.p1.y
        cx, cy = self.p2.x, self.p2.y
        
        # Calculate the determinant
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        
        if abs(d) < 1e-10:  # Points are collinear
            return None, None
        
        # Calculate circumcenter coordinates
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        
        # Calculate radius
        radius = np.sqrt((ux - ax)**2 + (uy - ay)**2)
        
        return Point(ux, uy), radius
    
    def has_vertex(self, p):
        return p == self.p0 or p == self.p1 or p == self.p2
    def __repr__(self): 
        return f"Triangle({self.p0}, {self.p1}, {self.p2})"

class TriangulationStep:
    """Stores the state of triangulation at each step"""
    def __init__(self, point_added, triangulation, bad_triangles, new_triangles, polygon_edges):
        self.point_added = point_added
        self.triangulation = triangulation.copy()
        self.bad_triangles = bad_triangles.copy()
        self.new_triangles = new_triangles.copy()
        self.polygon_edges = polygon_edges.copy()

def bowyer_watson_with_steps(point_list):
    # print(f"Starting triangulation with {len(point_list)} points")
    triangulation = []
    steps = []
    
    # Create super triangle which all points are in
    xmin = min(p.x for p in point_list)
    xmax = max(p.x for p in point_list)
    ymin = min(p.y for p in point_list)
    ymax = max(p.y for p in point_list)
    square_width = max(xmax - xmin, ymax - ymin)
    # midx = (xmax + xmin) / 2
    # midy = (ymax + ymin) / 2
    
    # p1 = Point(xmin- square_width/2, ymin)
    # p2 = Point(xmax + square_width/2, ymin)
    # p3 = Point(midx, ymin + 2 * square_width)
    dx = xmax - xmin
    dy = ymax - ymin
    delta = max(dx, dy)
    midx = (xmax + xmin) / 2
    midy = (ymax + ymin) / 2
    
    p1 = Point(midx - 10 * delta, midy - delta)
    p2 = Point(midx, midy + 10 * delta)
    p3 = Point(midx + 10 * delta, midy - delta)

    super_tri = Triangle(p1, p2, p3)
    triangulation.append(super_tri)
    
    # Initial step with just super triangle
    steps.append(TriangulationStep(None, triangulation, [], [], []))
    
    # Add each point one by one
    for i, point in enumerate(point_list):
        
        # Find bad triangles (those whose circumcircle contains the new point)
        bad_triangles = []
        for tri in triangulation:
            if tri.inCircumCircle(point):
                bad_triangles.append(tri)
        
        print(f"Found {len(bad_triangles)} bad triangles")
        # Hash tables makes things go faster
        edge_count = defaultdict(int)

        for tri in bad_triangles:
            for edge in tri.edges:
                edge_count[edge] += 1

        polygon_edges = [edge for edge, count in edge_count.items() if count == 1]

        # polygon_edges = []
        
        # for tri in bad_triangles:
        #     for edge in tri.edges:
        #         # Count how many bad triangles share this edge
        #         shared_count = sum(1 for other_tri in bad_triangles if edge in other_tri.edges) # this is a ton of complexity
                
        #         # If edge is only in one bad triangle, it's on the boundary
        #         if shared_count == 1:
        #             polygon_edges.append(edge)
        
        print(f"Polygon has {len(polygon_edges)} boundary edges")
        
        # Remove bad triangles
        for tri in bad_triangles:
            triangulation.remove(tri)
        
        # Create new triangles by connecting the point to each boundary edge
        new_triangles = []
        for edge in polygon_edges:
            new_tri = Triangle(edge.p0, edge.p1, point)
            triangulation.append(new_tri)
            new_triangles.append(new_tri)
        
        # Store this step
        steps.append(TriangulationStep(point, triangulation, bad_triangles, new_triangles, polygon_edges))
        
        print(f"Created {len(polygon_edges)} new triangles")
        print(f"Total triangles: {len(triangulation)}")
    
    # Separate super triangle from final triangulation
    super_triangle = super_tri
    final_triangulation = []
    for t in triangulation:
        if not (t.has_vertex(p1) or t.has_vertex(p2) or t.has_vertex(p3)):
            final_triangulation.append(t)
    
    print(f"\nFinal triangulation has {len(final_triangulation)} triangles")
    return steps, super_triangle, final_triangulation

class DelaunayAnimator:
    def __init__(self, steps, super_triangle, final_triangulation, point_list):
        self.steps = steps
        self.super_triangle = super_triangle
        self.final_triangulation = final_triangulation
        self.point_list = point_list
        self.current_step = 0
        self.is_playing = False
        self.show_circumcircles = False
        self.show_super_triangle = False
        self.show_bad_triangles = True
        self.show_new_triangles = True
        self.show_edges = True
        self.show_square = False  
        self.animation_speed = 500  #0.5s per frame normal speed
        
        # Create figure and setup
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        plt.subplots_adjust(left=0.18, bottom=0.2)
        
        # Initialize plot elements
        self.setup_plot()
        self.setup_controls()
        
        # Start animation
        self.animation = None
        self.update_display()
    def setup_plot(self):
        """Initialize the plot area and limits"""
        # Set fixed bounds for x and y axes
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
    # def setup_plot(self):
    #     """Initialize the plot area and limits"""
    #     # Calculate bounds
    #     all_x = [p.x for p in self.point_list]
    #     all_y = [p.y for p in self.point_list]
        
    #     if self.super_triangle:
    #         all_x.extend([self.super_triangle.p0.x, self.super_triangle.p1.x, self.super_triangle.p2.x])
    #         all_y.extend([self.super_triangle.p0.y, self.super_triangle.p1.y, self.super_triangle.p2.y])
        
    #     margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.1
    #     self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    #     self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
    #     self.ax.set_aspect('equal')
    #     self.ax.grid(True, alpha=0.3)
    
    def setup_controls(self):
        """Setup checkboxes, buttons, and speed slider"""
        # Checkboxes for display options
        rax_check = plt.axes([0.02, 0.6, 0.12, 0.3])
        labels = ['Circumcircles', 'Super Triangle', 'Bad Triangles', 'New Triangles', 'Show Edges', 'Show Square']
        visibility = [self.show_circumcircles, self.show_super_triangle, 
                     self.show_bad_triangles, self.show_new_triangles, 
                     self.show_edges, self.show_square]
        self.check = CheckButtons(rax_check, labels, visibility)
        self.check.on_clicked(self.toggle_display)
        
        # Control buttons
        ax_play = plt.axes([0.02, 0.50, 0.05, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_animation)
        
        ax_prev = plt.axes([0.08, 0.50, 0.04, 0.04])
        self.btn_prev = Button(ax_prev, '◄')
        self.btn_prev.on_clicked(self.prev_step)
        
        ax_next = plt.axes([0.02, 0.45, 0.04, 0.04])
        self.btn_next = Button(ax_next, '►')
        self.btn_next.on_clicked(self.next_step)
        
        ax_reset = plt.axes([0.08, 0.45, 0.04, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_animation)
        
        # Speed control slider
        ax_speed = plt.axes([0.02, 0.35, 0.12, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.5, 100.0, valinit=1.0, valfmt='%.1fx')
        self.speed_slider.on_changed(self.update_speed)
        
        # Add speed label
        plt.figtext(0.02, 0.39, 'Animation Speed:', fontsize=10, weight='bold')
    
    def update_speed(self, val):
        """Update animation speed based on slider value"""
        # Higher slider value = faster animation = lower interval
        self.animation_speed = int(3000 / val)
        
        # If animation is playing, restart it with new speed
        if self.is_playing:
            self.stop_animation()
            self.start_animation()
    
    def toggle_display(self, label):
        # Toggle visiblity of elements
        if label == 'Circumcircles':
            self.show_circumcircles = not self.show_circumcircles
        elif label == 'Super Triangle':
            self.show_super_triangle = not self.show_super_triangle
        elif label == 'Bad Triangles':
            self.show_bad_triangles = not self.show_bad_triangles
        elif label == 'New Triangles':
            self.show_new_triangles = not self.show_new_triangles
        elif label == 'Show Edges':
            self.show_edges = not self.show_edges
        elif label == 'Show Square':
            self.show_square = not self.show_square
        self.update_display()
    
    def toggle_animation(self, event):
        # Start and stop animation
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()
    
    def start_animation(self):
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        self.animation = FuncAnimation(self.fig, self.animate, interval=self.animation_speed, repeat=False)
        plt.draw()
    
    def stop_animation(self):
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        if self.animation:
            self.animation.event_source.stop()
        plt.draw()
    
    def animate(self, frame):
        # animate automatically
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_display()
        else:
            self.stop_animation()
    
    def prev_step(self, event):
        if self.is_playing:
            self.stop_animation()
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
    
    def next_step(self, event):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_display()
    
    def reset_animation(self, event):
        # Reset to the first step
        self.stop_animation()
        self.current_step = 0
        self.update_display()
    
    def update_display(self):
        # Update display
        self.ax.clear()
        self.setup_plot()
        
        current_step = self.steps[self.current_step]
        
        # Get super triangle vertices for filtering
        super_vertices = {self.super_triangle.p0, self.super_triangle.p1, self.super_triangle.p2}
        
        # Plot all points up to current point
        points_to_show = self.point_list[:self.current_step] if self.current_step > 0 else []
        
        for i, point in enumerate(points_to_show):
            self.ax.plot(point.x, point.y, 'ro', markersize=8, zorder=3)
            if self.show_new_triangles:
                self.ax.text(point.x + 0.1, point.y + 0.1, f'{i+1}', fontsize=10, fontweight='bold')
        
        # Highlight current point being added
        if current_step.point_added:
            if self.show_new_triangles:
                # Label the new point being added
                self.ax.plot(current_step.point_added.x, current_step.point_added.y, 
                            'go', markersize=12, zorder=4)

                self.ax.text(current_step.point_added.x + 0.2, current_step.point_added.y + 0.2, 
                            f'NEW', fontsize=10, fontweight='bold', color='green')
        
        # Plot current triangulation (excluding bad triangles and super triangle connections)
        good_triangles = [t for t in current_step.triangulation if t not in current_step.bad_triangles]
        
        for tri in good_triangles:
            # Check if triangle involves super triangle vertices
            has_super_vertex = any(vertex in super_vertices for vertex in [tri.p0, tri.p1, tri.p2])
            
            # Skip triangles with super vertices if super triangle display is off
            if has_super_vertex and not self.show_super_triangle:
                continue
                
            xs = [tri.p0.x, tri.p1.x, tri.p2.x, tri.p0.x]
            ys = [tri.p0.y, tri.p1.y, tri.p2.y, tri.p0.y]
            
            # Use different styling for super triangle connections
            if has_super_vertex:
                self.ax.plot(xs, ys, 'purple', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
            else:
                self.ax.plot(xs, ys, 'b-', linewidth=1, alpha=0.7, zorder=1)
                # Fill only regular triangles
                triangle = plt.Polygon([(tri.p0.x, tri.p0.y), (tri.p1.x, tri.p1.y), (tri.p2.x, tri.p2.y)], 
                                      alpha=0.1, facecolor='lightblue', zorder=0)
                self.ax.add_patch(triangle)
        
        # Plot bad triangles (to be deleted) - but only show if they don't involve super vertices OR super triangle is shown
        if self.show_bad_triangles:
            for tri in current_step.bad_triangles:
                has_super_vertex = any(vertex in super_vertices for vertex in [tri.p0, tri.p1, tri.p2])
                
                if has_super_vertex and not self.show_super_triangle:
                    continue
                    
                xs = [tri.p0.x, tri.p1.x, tri.p2.x, tri.p0.x]
                ys = [tri.p0.y, tri.p1.y, tri.p2.y, tri.p0.y]
                self.ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.8, zorder=2)
                
                triangle = plt.Polygon([(tri.p0.x, tri.p0.y), (tri.p1.x, tri.p1.y), (tri.p2.x, tri.p2.y)], 
                                      alpha=0.2, facecolor='red', zorder=0)
                self.ax.add_patch(triangle)
        
        # Plot new triangles (just created) - but only show if they don't involve super vertices OR super triangle is shown
        if self.show_new_triangles:
            for tri in current_step.new_triangles:
                has_super_vertex = any(vertex in super_vertices for vertex in [tri.p0, tri.p1, tri.p2])
                
                if has_super_vertex and not self.show_super_triangle:
                    continue
                    
                xs = [tri.p0.x, tri.p1.x, tri.p2.x, tri.p0.x]
                ys = [tri.p0.y, tri.p1.y, tri.p2.y, tri.p0.y]
                self.ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.9, zorder=2)
                
                # Fill with green
                triangle = plt.Polygon([(tri.p0.x, tri.p0.y), (tri.p1.x, tri.p1.y), (tri.p2.x, tri.p2.y)], 
                                      alpha=0.2, facecolor='lightgreen', zorder=0)
                self.ax.add_patch(triangle)
        
        # Plot polygon boundary edges - filter out edges connected to super vertices
        if self.show_edges:
            for edge in current_step.polygon_edges:
                has_super_vertex = edge.p0 in super_vertices or edge.p1 in super_vertices
                
                if has_super_vertex and not self.show_super_triangle:
                    continue
                    
                self.ax.plot([edge.p0.x, edge.p1.x], [edge.p0.y, edge.p1.y], 
                            'orange', linewidth=3, alpha=0.8, zorder=2)
        
        # Plot super triangle outline
        if self.show_super_triangle and self.super_triangle:
            xs = [self.super_triangle.p0.x, self.super_triangle.p1.x, self.super_triangle.p2.x, self.super_triangle.p0.x]
            ys = [self.super_triangle.p0.y, self.super_triangle.p1.y, self.super_triangle.p2.y, self.super_triangle.p0.y]
            self.ax.plot(xs, ys, 'purple', linestyle='--', linewidth=3, alpha=0.8, zorder=1)
            
            # Plot super triangle vertices
            for vertex in super_vertices:
                self.ax.plot(vertex.x, vertex.y, 'mo', markersize=10, zorder=3)
        
        # Plot circumcircles - but only for triangles we're actually showing
        if self.show_circumcircles:
            for tri in good_triangles:
                has_super_vertex = any(vertex in super_vertices for vertex in [tri.p0, tri.p1, tri.p2])
                
                if has_super_vertex and not self.show_super_triangle:
                    continue
                    
                center, radius = tri.get_circumcenter_and_radius()
                if center and radius:
                    circle = plt.Circle((center.x, center.y), radius, fill=False, color='red', 
                                      linestyle=':', alpha=0.6, zorder=2)
                    self.ax.add_patch(circle)
                    self.ax.plot(center.x, center.y, 'r+', markersize=6, zorder=2)
        
        # Kind of redundant, might remove later
        if self.show_square:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Create a square that fits the current view
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            size = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.8
            
            square = patches.Rectangle((center_x - size/2, center_y - size/2), 
                                     size, size, 
                                     edgecolor='orange', 
                                     facecolor='none', 
                                     linewidth=2, 
                                     linestyle='-',
                                     alpha=0.7,
                                     zorder=1)
            self.ax.add_patch(square)

        # Update title
        step_info = f"Step {self.current_step}/{len(self.steps)-1}"
        if current_step.point_added:
            step_info += f" - Adding point {current_step.point_added}"
        elif self.current_step == 0:
            step_info += " - Super Triangle"
        
        speed_info = f" | Speed: {self.speed_slider.val:.1f}x"
        self.ax.set_title(f"Bowyer-Watson Algorithm Animation\n{step_info}{speed_info}")
        
        plt.draw()

def run_delaunay_animation():
    random.seed(5)
    
    points = []
    used_coords = set()
    
    while len(points) < 50:
        x, y = random.uniform(-10, 10), random.uniform(-10, 10)
        if (x, y) not in used_coords:
            points.append(Point(x, y))
            used_coords.add((x, y))
    
    print("Input points:")
    for i, p in enumerate(points):
        print(f"  {i+1}: {p}")
    
    # Triangulation
    time_start = time.time()
    steps, super_tri, final_triangulation = bowyer_watson_with_steps(points)
    time_end = time.time()
    print(f"\nTriangulation completed in {time_end - time_start:.2f} seconds")
    # Create and show animation
    animator = DelaunayAnimator(steps, super_tri, final_triangulation, points)
    plt.show()

if __name__ == '__main__':
    run_delaunay_animation()