import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_polygon(n_points, z_tolerance=0.05, radius=1.0, center=(0, 0, 0)):
    """
    Generate a closed polygon contour roughly on a plane in 3D space
    
    Parameters:
    - n_points: Number of points in the polygon
    - z_tolerance: Maximum z-deviation as fraction of radius (0 for perfect plane)
    - radius: Approximate size of the polygon
    - center: Center point of the polygon
    """
    # Generate angles for polygon vertices
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Calculate x and y coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Add some noise to make it irregular
    x += np.random.normal(0, radius*0.05, n_points)
    y += np.random.normal(0, radius*0.05, n_points)
    
    # Add z-coordinate with controlled deviation
    z = np.random.uniform(-radius*z_tolerance, radius*z_tolerance, n_points)
    
    # Apply center offset
    x += center[0]
    y += center[1]
    z += center[2]
    
    # Close the polygon by adding first point at the end
    points = np.column_stack((x, y, z))
    points = np.vstack([points, points[0]])
    
    return points

def generate_planar_contours():
    """Generate two closed polygonal contours roughly on the same plane"""
    # First contour: 5-sided polygon with slight z-variation
    set1 = generate_polygon(n_points=5, z_tolerance=0.02, radius=1.5, center=(-1, -1, 0))
    
    # Second contour: 7-sided polygon with more z-variation
    set2 = generate_polygon(n_points=7, z_tolerance=0.1, radius=1.2, center=(1, 1, 0.1))
    
    return set1, set2

def plot_3d_contours(set1, set2):
    """Plot the two sets of points in 3D space"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first set
    ax.plot(set1[:, 0], set1[:, 1], set1[:, 2], 'b-', label='5-sided polygon', linewidth=2)
    ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], c='b', s=50)
    
    # Plot the second set
    ax.plot(set2[:, 0], set2[:, 1], set2[:, 2], 'r-', label='7-sided polygon', linewidth=2)
    ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], c='r', s=50)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.2])  # Flatten the z-axis
    
    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Two Planar Polygonal Contours in 3D Space')
    ax.legend()
    
    # Set view to emphasize planarity
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

# Generate and plot the contours
set1, set2 = generate_planar_contours()
plot_3d_contours(set1, set2)

# Print the first few points of each set
print("Points of 5-sided polygon:")
print(set1)
print("\nPoints of 7-sided polygon:")
print(set2)
