import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import csv

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

def export_to_csv(points, filename):
    """Export point set to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])  # Write header
        writer.writerows(points)
    print(f"Saved {len(points)} points to {filename}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate two planar polygonal contours in 3D space and export to CSV')
    parser.add_argument('n_points1', type=int, help='Number of points in first contour')
    parser.add_argument('n_points2', type=int, help='Number of points in second contour')
    parser.add_argument('z_tolerance', type=float, help='Maximum z-deviation as fraction of radius')
    parser.add_argument('output1', help='Output filename for first contour')
    parser.add_argument('output2', help='Output filename for second contour')
    parser.add_argument('--plot', action='store_true', help='Show 3D plot of contours')
    
    args = parser.parse_args()
    
    # Generate contours
    set1 = generate_polygon(n_points=args.n_points1, z_tolerance=args.z_tolerance, 
                           radius=1.5, center=(-1, -1, 0))
    set2 = generate_polygon(n_points=args.n_points2, z_tolerance=args.z_tolerance, 
                           radius=1.2, center=(1, 1, 0.1))
    
    # Export to CSV
    export_to_csv(set1, args.output1)
    export_to_csv(set2, args.output2)
    
    # Optional plotting
    if args.plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(set1[:, 0], set1[:, 1], set1[:, 2], 'b-', label=f'Contour 1 ({args.n_points1} points)')
        ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], c='b', s=50)
        
        ax.plot(set2[:, 0], set2[:, 1], set2[:, 2], 'r-', label=f'Contour 2 ({args.n_points2} points)')
        ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], c='r', s=50)
        
        ax.set_box_aspect([1, 1, 0.2])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Two Planar Polygonal Contours in 3D Space')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
