import argparse
import numpy as np

def read_ply(file_path):
    vertices = []
    colors = []
    with open(file_path, 'r') as f:
        while True:
            line = f.readline().strip()
            if line.startswith("end_header"):
                break
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:  # Assuming x, y, z, r, g, b
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(vertices), np.array(colors)

def generate_html(vertices, colors, output_file):
    # Convert vertices and colors to JavaScript arrays
    vertices_js = "[" + ", ".join(f"[{v[0]}, {v[1]}, {v[2]}]" for v in vertices) + "]"
    colors_js = "[" + ", ".join(f"0x{r:02x}{g:02x}{b:02x}" for r, g, b in colors) + "]"

    # HTML template with Three.js
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLY Viewer</title>
        <style>
            body {{ margin: 0; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <script src="three.min.js"></script>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            // Add points
            const vertices = {vertices_js};
            const colors = {colors_js};
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors.flat(), 3));

            const material = new THREE.PointsMaterial({{ size: 0.1, vertexColors: true }});
            const points = new THREE.Points(geometry, material);
            scene.add(points);

            // Position camera
            camera.position.z = 5;

            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                points.rotation.x += 0.01;
                points.rotation.y += 0.01;
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point cloud visualization')
    parser.add_argument('-i', required=True, help='Input PLY file')
    parser.add_argument('-o', required=True, help='Output HTML file')
    args = parser.parse_args()

    vertices, colors = read_ply(args.i)
    generate_html(vertices, colors, args.o)
    print(f"Generated {args.o}")
