import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circle(center, radius, n_points=100, noise=0.02):
    """Generate points on a circle with slight noise to make it interesting"""
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(theta) + np.random.normal(0, noise, n_points)
    y = center[1] + radius * np.sin(theta) + np.random.normal(0, noise, n_points)
    z = np.full(n_points, center[2]) + np.random.normal(0, noise/2, n_points)
    return np.column_stack((x, y, z))

def generate_rectangle(center, width, height, n_points=100, noise=0.02):
    """Generate points on a rectangle with rounded corners"""
    # Generate points for each side
    side_points = n_points // 4
    x, y, z = [], [], []
    
    # Top side
    x_top = np.linspace(-width/2, width/2, side_points)
    y_top = np.full(side_points, height/2)
    x.extend(x_top)
    y.extend(y_top)
    
    # Right side
    y_right = np.linspace(height/2, -height/2, side_points)
    x_right = np.full(side_points, width/2)
    x.extend(x_right)
    y.extend(y_right)
    
    # Bottom side
    x_bottom = np.linspace(width/2, -width/2, side_points)
    y_bottom = np.full(side_points, -height/2)
    x.extend(x_bottom)
    y.extend(y_bottom)
    
    # Left side
    y_left = np.linspace(-height/2, height/2, side_points)
    x_left = np.full(side_points, -width/2)
    x.extend(x_left)
    y.extend(y_left)
    
    # Convert to numpy arrays and add center offset and noise
    x = np.array(x) + center[0] + np.random.normal(0, noise, len(x))
    y = np.array(y) + center[1] + np.random.normal(0, noise, len(y))
    z = np.full(len(x), center[2]) + np.random.normal(0, noise/2, len(x))
    
    return np.column_stack((x, y, z))

def generate_planar_contours():
    """Generate two closed contours roughly on the same plane"""
    # First contour: Circle
    set1 = generate_circle(center=(0, 0, 0), radius=1.5)
    
    # Second contour: Rectangle
    set2 = generate_rectangle(center=(0, 0, 0.1), width=2.5, height=1.8)
    
    # Close the contours by adding the first point at the end
    set1 = np.vstack([set1, set1[0]])
    set2 = np.vstack([set2, set2[0]])
    
    return set1, set2

def plot_3d_contours(set1, set2):
    """Plot the two sets of points in 3D space"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first set
    ax.plot(set1[:, 0], set1[:, 1], set1[:, 2], 'b-', label='Circle', linewidth=2)
    ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], c='b', s=20)
    
    # Plot the second set
    ax.plot(set2[:, 0], set2[:, 1], set2[:, 2], 'r-', label='Rectangle', linewidth=2)
    ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], c='r', s=20)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.1])  # Flatten the z-axis
    
    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Two Planar Closed Contours in 3D Space')
    ax.legend()
    
    # Set view to emphasize planarity
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

# Generate and plot the contours
set1, set2 = generate_planar_contours()
plot_3d_contours(set1, set2)

# Print the first few points of each set
print("First 5 points of Circle:")
print(set1[:5])
print("\nFirst 5 points of Rectangle:")
print(set2[:5])

