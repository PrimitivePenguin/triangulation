import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import Delaunay
from scipy.ndimage import sobel
import matplotlib.colors as mcolors

def lerp(a, b, x):
    """Linear interpolation between a and b with x."""
    return a + x * (b - a)

def fade(t):
    """Fade function as defined by Ken Perlin. This eases coordinate values
    so that they will ease towards integral values. This ends up smoothing
    the final output."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def generate_perlin_noise_2d(shape, res, seed=0):
    """Generate a 2D numpy array of perlin noise."""
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3
   
    # Create coordinate grid
    x = np.linspace(0, res[0], shape[0], endpoint=False)
    y = np.linspace(0, res[1], shape[1], endpoint=False)
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
   
    # Get fractional parts
    fx = grid_x - np.floor(grid_x)
    fy = grid_y - np.floor(grid_y)
   
    # Get integer grid coordinates
    ix = np.floor(grid_x).astype(int)
    iy = np.floor(grid_y).astype(int)
   
    # Generate gradients
    np.random.seed(seed)
    gradients = np.random.randn(res[0]+1, res[1]+1, 2)
    gradients /= np.linalg.norm(gradients, axis=2, keepdims=True)
   
    # Get gradients at corners
    g00 = gradients[ix, iy]
    g10 = gradients[ix + 1, iy]
    g01 = gradients[ix, iy + 1]
    g11 = gradients[ix + 1, iy + 1]
   
    # Calculate dot products
    n00 = np.sum(np.stack([fx, fy], axis=2) * g00, axis=2)
    n10 = np.sum(np.stack([fx - 1, fy], axis=2) * g10, axis=2)
    n01 = np.sum(np.stack([fx, fy - 1], axis=2) * g01, axis=2)
    n11 = np.sum(np.stack([fx - 1, fy - 1], axis=2) * g11, axis=2)
   
    # Apply fade function and interpolate
    u = f(fx)
    v = f(fy)
   
    n0 = n00 * (1 - u) + n10 * u
    n1 = n01 * (1 - u) + n11 * u
   
    return np.sqrt(2) * (n0 * (1 - v) + n1 * v)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, seed=0):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    max_amplitude = 0
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]), seed=seed)
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2
    return noise / max_amplitude

def create_3d_perlin_surface(heightmap, height_scale=50, subsample=4):
    """Create 3D surface from 2D heightmap with optional subsampling for performance."""
    # Subsample for performance if needed
    if subsample > 1:
        heightmap_sub = heightmap[::subsample, ::subsample]
    else:
        heightmap_sub = heightmap
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(
        np.arange(0, heightmap.shape[0], subsample),
        np.arange(0, heightmap.shape[1], subsample),
        indexing='ij'
    )
    
    # Scale heights
    z_coords = heightmap_sub * height_scale
    
    return x_coords, y_coords, z_coords

def sample_points_by_gradient(heightmap, max_points=1000, seed=None):
    """Sample points from heightmap based on gradient magnitude probability."""
    if seed is not None:
        np.random.seed(seed)
    
    # Compute gradients
    dx = sobel(heightmap, axis=1)
    dy = sobel(heightmap, axis=0)
    grad_mag = np.sqrt(dx**2 + dy**2)
    
    # Normalize to [0, 1]
    if np.max(grad_mag) > 0:
        grad_mag = grad_mag / np.max(grad_mag)
    
    # Use gradient as probability map for sampling
    flat_map = grad_mag.flatten()
    # Add small epsilon to avoid zero probabilities
    flat_map = flat_map + 1e-8
    prob_map = flat_map / flat_map.sum()
    
    # Sample indices weighted by gradient
    n_samples = min(max_points, flat_map.size)
    indices = np.random.choice(np.arange(flat_map.size), size=n_samples, replace=False, p=prob_map)
    ys, xs = np.unravel_index(indices, heightmap.shape)
    zs = heightmap[ys, xs]
    
    return xs, ys, zs, grad_mag

def create_delaunay_triangulation(xs, ys):
    """Create Delaunay triangulation from x, y coordinates."""
    points = np.column_stack((xs, ys))
    tri = Delaunay(points)
    return tri, points

def create_3d_gradient_edges(points, tri, zs, cmap='terrain', color1=None, color2=None, n_segments=10):
    """Create 3D gradient-colored edges for triangulation."""
    # Get all unique edges from triangles
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)
    
    # Convert to list for consistent ordering
    edges = list(edges)
    
    # Create 3D line segments for gradient coloring
    segments_3d = []
    colors = []
    
    # Set up colormap and normalization
    if color1 is not None and color2 is not None:
        # Create custom colormap from two colors
        custom_colors = [color1, color2]
        cm = mcolors.LinearSegmentedColormap.from_list('custom', custom_colors)
    else:
        # Use the specified colormap (default: terrain)
        cm = plt.get_cmap(cmap)
    
    # Always normalize based on the actual z values
    norm = mcolors.Normalize(vmin=np.min(zs), vmax=np.max(zs))
    
    for edge in edges:
        p1_idx, p2_idx = edge
        x1, y1 = points[p1_idx]
        x2, y2 = points[p2_idx]
        z1 = zs[p1_idx]
        z2 = zs[p2_idx]
        
        # Create multiple segments along the edge for smooth gradient
        for i in range(n_segments):
            t1 = i / n_segments
            t2 = (i + 1) / n_segments
            
            # Interpolate 3D positions
            seg_x1 = x1 + t1 * (x2 - x1)
            seg_y1 = y1 + t1 * (y2 - y1)
            seg_z1 = z1 + t1 * (z2 - z1)
            
            seg_x2 = x1 + t2 * (x2 - x1)
            seg_y2 = y1 + t2 * (y2 - y1)
            seg_z2 = z1 + t2 * (z2 - z1)
            
            segments_3d.append([[seg_x1, seg_y1, seg_z1], [seg_x2, seg_y2, seg_z2]])
            
            # Interpolate colors
            z_interp = z1 + (t1 + t2) / 2 * (z2 - z1)
            colors.append(cm(norm(z_interp)))
    
    return segments_3d, colors

def create_3d_triangular_faces(points, tri, zs, cmap='terrain', color1=None, color2=None, alpha=0.7):
    """Create 3D triangular faces with gradient colors."""
    # Set up colormap and normalization
    if color1 is not None and color2 is not None:
        custom_colors = [color1, color2]
        cm = mcolors.LinearSegmentedColormap.from_list('custom', custom_colors)
    else:
        cm = plt.get_cmap(cmap)
    
    norm = mcolors.Normalize(vmin=np.min(zs), vmax=np.max(zs))
    
    # Create 3D vertices for each triangle
    vertices_3d = []
    face_colors = []
    
    for simplex in tri.simplices:
        # Get the three vertices of the triangle
        triangle_3d = []
        triangle_zs = []
        
        for vertex_idx in simplex:
            x, y = points[vertex_idx]
            z = zs[vertex_idx]
            triangle_3d.append([x, y, z])
            triangle_zs.append(z)
        
        vertices_3d.append(triangle_3d)
        
        # Color based on average height of triangle
        avg_z = np.mean(triangle_zs)
        face_color = cm(norm(avg_z))
        face_colors.append(face_color)
    
    return vertices_3d, face_colors

# Generate terrain parameters
shape = (100, 100)  # Reduced for better 3D surface performance
res = (5, 5)
octaves = 6
persistence = 0.5
height_scale = 30

plt.style.use('dark_background')
# Generate fractal noise (heightmap)
heightmap = generate_fractal_noise_2d(shape, res, octaves, persistence, seed=2)

# Create 3D surface from Perlin noise
X_surface, Y_surface, Z_surface = create_3d_perlin_surface(heightmap, height_scale=height_scale, subsample=2)

# Sample points based on gradient
max_points = 500
xs, ys, zs_original, grad_mag = sample_points_by_gradient(heightmap, max_points=max_points, seed=42)
zs = zs_original * height_scale  # Scale the sampled heights

# Create Delaunay triangulation
tri, points = create_delaunay_triangulation(xs, ys)

# Create 3D triangulation elements
segments_3d, edge_colors = create_3d_gradient_edges(points, tri, zs, cmap='terrain')
faces_3d, face_colors = create_3d_triangular_faces(points, tri, zs, cmap='terrain', alpha=0.6)

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Original 3D Perlin noise surface
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(X_surface, Y_surface, Z_surface, cmap='terrain', 
                        alpha=0.8, linewidth=0, antialiased=True, shade=True)
ax1.set_title('3D Perlin Noise Surface (Original)', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height')
ax1.view_init(elev=30, azim=45)

# 2. Triangulated model only
ax2 = fig.add_subplot(222, projection='3d')
poly_collection = Poly3DCollection(faces_3d, facecolors=face_colors, alpha=0.8, edgecolors='black', linewidths=0.5)
ax2.add_collection3d(poly_collection)
scatter2 = ax2.scatter(xs, ys, zs, c=zs, cmap='terrain', s=30, alpha=1.0, edgecolors='white', linewidth=0.5)
ax2.set_title('Delaunay Triangulated Model', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Height')
ax2.view_init(elev=30, azim=45)

# 3. Overlay: Perlin surface + triangulation (wireframe)
ax3 = fig.add_subplot(223, projection='3d')
# Plot Perlin surface with transparency
surf3 = ax3.plot_surface(X_surface, Y_surface, Z_surface, cmap='terrain', 
                        alpha=0.5, linewidth=0, antialiased=True)
# Add triangulation wireframe
line_collection3 = Line3DCollection(segments_3d, colors=edge_colors, linewidths=2, alpha=0.9)
ax3.add_collection3d(line_collection3)
# Add sample points
scatter3 = ax3.scatter(xs, ys, zs, c=zs, cmap='terrain', s=40, alpha=1.0, 
                      edgecolors='black', linewidth=1, zorder=10)
ax3.set_title('Overlay: Perlin Surface + Triangulation (Wireframe)', fontsize=12)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Height')
ax3.view_init(elev=30, azim=45)

# 4. Overlay: Perlin surface + triangulation (solid faces)
ax4 = fig.add_subplot(224, projection='3d')
# Plot Perlin surface with high transparency
surf4 = ax4.plot_surface(X_surface, Y_surface, Z_surface, cmap='terrain', 
                        alpha=0.3, linewidth=0, antialiased=True)
# Add triangulation faces
poly_collection4 = Poly3DCollection(faces_3d, facecolors=face_colors, alpha=0.7, edgecolors='none')
ax4.add_collection3d(poly_collection4)
# Add triangulation wireframe
line_collection4 = Line3DCollection(segments_3d, colors='black', linewidths=1, alpha=0.6)
ax4.add_collection3d(line_collection4)
# Add sample points
scatter4 = ax4.scatter(xs, ys, zs, c=zs, cmap='terrain', s=35, alpha=1.0, 
                      edgecolors='white', linewidth=1, zorder=10)
ax4.set_title('Overlay: Perlin Surface + Triangulation (Solid)', fontsize=12)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Height')
ax4.view_init(elev=30, azim=45)

# Set consistent limits for all subplots
z_min = min(np.min(Z_surface), np.min(zs)) - 2
z_max = max(np.max(Z_surface), np.max(zs)) + 2

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_zlim(z_min, z_max)

plt.tight_layout()
plt.show()

# Create detailed comparison figure
fig = plt.figure(figsize=(18, 12))

# Large detailed overlay view
ax_main = fig.add_subplot(121, projection='3d')

# Plot original Perlin surface with transparency
surf_main = ax_main.plot_surface(X_surface, Y_surface, Z_surface, cmap='terrain', 
                                alpha=0.4, linewidth=0, antialiased=True, shade=True)



ax_main.set_zlim(z_min, z_max)



# Create an interactive rotating view showing the approximation quality
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(projection='3d')
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

# High-resolution Perlin surface (subsampled less for this view)
X_hires, Y_hires, Z_hires = create_3d_perlin_surface(heightmap, height_scale=height_scale, subsample=1)

# Plot high-res Perlin surface
surf_hires = ax.plot_surface(X_hires, Y_hires, Z_hires, cmap='terrain', 
                            alpha=0.3, linewidth=0, antialiased=True, shade=True)

# Add triangulation

line_overlay = Line3DCollection(segments_3d, colors=edge_colors, linewidths=2, alpha=0.9)
ax.add_collection3d(line_overlay)
# poly_overlay = Poly3DCollection(faces_3d, facecolors=face_colors, alpha=0.8, 
#                                edgecolors='black', linewidths=1)
# ax.add_collection3d(poly_overlay)
# Add sample points
scatter_overaly = ax.scatter(xs, ys, zs, c=zs, cmap='terrain', s=40, alpha=1.0, 
                      edgecolors='black', linewidth=1, zorder=10)
# scatter_overlay = ax.scatter(xs, ys, zs, c='red', s=80, alpha=1.0, 
#                             edgecolors='darkred', linewidth=2, zorder=15,
#                             marker='o', label='Sampled Points')

ax.set_title('High-Resolution Comparison:\nContinuous Perlin Noise vs Discrete Triangulation\n' + 
            f'({len(xs)} sample points approximating {shape[0]}×{shape[1]} surface)', fontsize=14)
ax.set_xlabel('X Position', fontsize=12)
ax.set_ylabel('Y Position', fontsize=12)
ax.set_zlabel('Height', fontsize=12)
ax.view_init(elev=20, azim=30)

# Set limits
ax.set_xlim(0, shape[1])
ax.set_ylim(0, shape[0])
ax.set_zlim(np.min(Z_hires) - 2, np.max(Z_hires) + 2)

# Add legend
ax.legend()

# Add colorbar
cbar = plt.colorbar(surf_hires, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Elevation', fontsize=12)

plt.tight_layout()
plt.show()
# Plot 2D Perlin noise heightmap
plt.figure(figsize=(8, 6))
plt.imshow(heightmap, cmap='terrain', origin='lower', aspect='auto')
plt.colorbar(label='Height')
plt.title('2D Perlin Noise Heightmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()



# 2. Triangulated model only
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# # High-resolution Perlin surface (subsampled less for this view)
# X_hires, Y_hires, Z_hires = create_3d_perlin_surface(heightmap, height_scale=height_scale, subsample=1)

# # Plot high-res Perlin surface
# surf_hires = ax.plot_surface(X_hires, Y_hires, Z_hires, cmap='terrain', 
#                             alpha=0.5, linewidth=0, antialiased=True, shade=True)

# Add triangulation
poly_overlay = Poly3DCollection(faces_3d, facecolors=face_colors, alpha=0.8, 
                               edgecolors='black', linewidths=1)
ax.add_collection3d(poly_overlay)

# Highlight the sampled points


ax.view_init(elev=20, azim=30)

# Set limits
ax.set_xlim(0, shape[1])
ax.set_ylim(0, shape[0])
ax.set_zlim(np.min(Z_hires) - 2, np.max(Z_hires) + 2)
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(18, 12))
ax2 = plt.figure(figsize=(20, 20))
poly_collection = Poly3DCollection(faces_3d, facecolors=face_colors, alpha=0.8, edgecolors='black', linewidths=0.5)
ax2.add_collection3d(poly_collection)
scatter2 = ax2.scatter(xs, ys, zs, c=zs, cmap='terrain', s=30, alpha=1.0, edgecolors='white', linewidth=0.5)
ax2 = fig.add_subplot(projection='3d')

# Simplified model
ax2.set_title('Delaunay Triangulated Model', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Height')
ax2.view_init(elev=30, azim=45)
ax2.grid(False)
plt.tight_layout()
plt.show()


# Print comprehensive statistics
print(f"3D Perlin Noise + Triangulation Analysis:")
print(f"=" * 50)
print(f"Original terrain grid size: {shape}")
print(f"Perlin surface points: {X_surface.size:,}")
print(f"Triangulation sample points: {len(xs)}")
print(f"Triangulation faces: {len(faces_3d):,}")
print(f"Triangulation edges: {len(segments_3d):,}")
print(f"Height scale factor: {height_scale}")
print(f"")
print(f"Height Statistics:")
print(f"Perlin surface range: {np.min(Z_surface):.2f} to {np.max(Z_surface):.2f}")
print(f"Triangulation range: {np.min(zs):.2f} to {np.max(zs):.2f}")
print(f"Original noise range: {np.min(heightmap):.3f} to {np.max(heightmap):.3f}")
print(f"")
print(f"Approximation Quality:")
print(f"Surface area reduction: {(1 - len(xs) / (shape[0] * shape[1])) * 100:.1f}%")
print(f"Data compression ratio: {(shape[0] * shape[1]) / len(xs):.1f}:1")