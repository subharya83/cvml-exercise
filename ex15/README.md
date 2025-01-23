## Road trip planning

https://project-osrm.org/docs/v5.24.0/api/#

Building a dataset that stores geographical distances between zip codes across the continental US is a complex but achievable task. To make it tractable, you can divide the data collection into sections (e.g., by state or region) and use APIs like Google Maps, OpenStreetMap, or OSRM (Open Source Routing Machine) to calculate road distances. Below is a step-by-step guide and relevant Python scripts to help you build this dataset.

---

### **Step 1: Gather Zip Code Data**
You need a list of all zip codes in the continental US. You can use the US Postal Service's zip code database or open datasets like the one from [GeoNames](https://www.geonames.org/).

```python
import pandas as pd

# Load zip code data (example format: zip, lat, lon, state)
zip_data = pd.read_csv("us_zip_codes.csv")
print(zip_data.head())
```

---

### **Step 2: Divide the Data into Sections**
Divide the zip codes into sections (e.g., by state or region) to make the data collection process manageable.

```python
# Group zip codes by state
state_groups = zip_data.groupby("state")

# Save each state's zip codes to separate files
for state, group in state_groups:
    group.to_csv(f"zip_codes_{state}.csv", index=False)
```

---

### **Step 3: Calculate Distances Between Zip Codes**
Use a routing API to calculate road distances between zip codes. Below is an example using the OSRM API.

#### Install Required Libraries
```bash
pip install requests pandas
```

#### Python Script for Distance Calculation
```python
import requests
import pandas as pd
import time

# OSRM API endpoint
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

def get_road_distance(lon1, lat1, lon2, lat2):
    """Get road distance between two points using OSRM."""
    url = f"{OSRM_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("routes"):
            return data["routes"][0]["distance"] / 1000  # Convert meters to kilometers
    return None

# Load zip code data for a specific state
state_zip_data = pd.read_csv("zip_codes_CA.csv")  # Example: California

# Create an empty DataFrame to store distances
distance_matrix = pd.DataFrame(index=state_zip_data["zip"], columns=state_zip_data["zip"])

# Calculate distances between all pairs of zip codes
for i, row1 in state_zip_data.iterrows():
    for j, row2 in state_zip_data.iterrows():
        if i < j:  # Avoid redundant calculations
            distance = get_road_distance(row1["lon"], row1["lat"], row2["lon"], row2["lat"])
            distance_matrix.at[row1["zip"], row2["zip"]] = distance
            distance_matrix.at[row2["zip"], row1["zip"]] = distance
    time.sleep(1)  # Add delay to avoid overloading the API

# Save the distance matrix to a CSV file
distance_matrix.to_csv("distance_matrix_CA.csv")
```

---

### **Step 4: Combine Data from All Sections**
Once you have distance matrices for all states or regions, combine them into a single dataset.

```python
import pandas as pd
import glob

# Load all distance matrix files
distance_files = glob.glob("distance_matrix_*.csv")

# Combine into a single DataFrame
combined_distances = pd.concat([pd.read_csv(f) for f in distance_files], ignore_index=True)

# Save the combined dataset
combined_distances.to_csv("us_zip_code_distances.csv", index=False)
```

---

### **Step 5: Optimize and Scale**
- **Parallel Processing**: Use Python's `multiprocessing` or libraries like `Dask` to speed up distance calculations.
- **API Limits**: If using paid APIs like Google Maps, monitor usage to avoid exceeding limits.
- **Data Storage**: Store the dataset in a database (e.g., SQLite, PostgreSQL) for efficient querying.

---

### **Step 6: Visualize or Analyze the Data**
You can use libraries like `matplotlib` or `folium` to visualize the distances or analyze patterns.

```python
import matplotlib.pyplot as plt

# Example: Plot a histogram of distances
plt.hist(combined_distances.values.flatten(), bins=50)
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.title("Distribution of Road Distances Between Zip Codes")
plt.show()
```

---

### **Notes**
1. **Data Size**: The dataset will be large, as there are ~40,000 zip codes in the US, resulting in ~800 million pairs.
2. **API Costs**: If using paid APIs, estimate costs beforehand.
3. **OpenStreetMap**: For free and open-source routing, consider using OSRM or similar tools.

By following these steps, you can build a comprehensive dataset of road distances between zip codes across the continental US.


## More docs

To build a dataset of geographical distances and average driving times between zip codes across the continental US using **OSRM (Open Source Routing Machine)** and other open-source tools, you can follow these steps:

1. **Set up OSRM**: Use the OSRM C++ backend to preprocess map data and serve routing requests.
2. **Use Python for Data Handling**: Use Python to manage zip code data, interact with the OSRM backend, and store results.
3. **Parallelize the Process**: Use parallel processing to speed up distance and time calculations.

Below is a detailed guide with **C++ and Python code** to achieve this.

---

### **Step 1: Set Up OSRM**

#### 1.1 Download OSM Data
Download OpenStreetMap (OSM) data for the continental US from [Geofabrik](https://download.geofabrik.de/north-america.html).

```bash
wget https://download.geofabrik.de/north-america/us-latest.osm.pbf
```

#### 1.2 Build OSRM Backend
Clone and build the OSRM backend from its GitHub repository.

```bash
# Clone OSRM
git clone https://github.com/Project-OSRM/osrm-backend.git
cd osrm-backend

# Install dependencies
sudo apt-get install build-essential git cmake pkg-config \
libbz2-dev libstxxl-dev libstxxl1v5 libxml2-dev \
libzip-dev libboost-all-dev lua5.2 liblua5.2-dev libtbb-dev

# Build OSRM
mkdir -p build
cd build
cmake ..
make
```

#### 1.3 Preprocess OSM Data
Extract and preprocess the OSM data using OSRM.

```bash
# Extract data
./osrm-extract ../us-latest.osm.pbf -p ../profiles/car.lua

# Partition and customize
./osrm-partition ../us-latest.osrm
./osrm-customize ../us-latest.osrm

# Start OSRM backend
./osrm-routed ../us-latest.osrm
```

Now, the OSRM backend is running and ready to serve routing requests.

---

### **Step 2: Use Python to Interact with OSRM**

#### 2.1 Install Required Python Libraries
Install the required libraries to interact with OSRM and handle data.

```bash
pip install requests pandas numpy
```

#### 2.2 Python Script to Calculate Distances and Driving Times
Use Python to query the OSRM backend for distances and driving times between zip codes.

```python
import requests
import pandas as pd
import numpy as np
from multiprocessing import Pool

# OSRM API endpoint
OSRM_URL = "http://localhost:5000/route/v1/driving/"

def get_route_info(lon1, lat1, lon2, lat2):
    """Get distance (in km) and driving time (in minutes) between two points."""
    url = f"{OSRM_URL}{lon1},{lat1};{lon2},{lat2}?overview=false"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("routes"):
            distance = data["routes"][0]["distance"] / 1000  # Convert meters to kilometers
            duration = data["routes"][0]["duration"] / 60   # Convert seconds to minutes
            return distance, duration
    return None, None

def process_zip_pair(args):
    """Process a pair of zip codes."""
    zip1, zip2, lat1, lon1, lat2, lon2 = args
    distance, duration = get_route_info(lon1, lat1, lon2, lat2)
    return zip1, zip2, distance, duration

# Load zip code data
zip_data = pd.read_csv("us_zip_codes.csv")  # Columns: zip, lat, lon, state

# Generate all unique pairs of zip codes
zip_pairs = []
for i, row1 in zip_data.iterrows():
    for j, row2 in zip_data.iterrows():
        if i < j:  # Avoid redundant calculations
            zip_pairs.append((row1["zip"], row2["zip"], row1["lat"], row1["lon"], row2["lat"], row2["lon"]))

# Use multiprocessing to speed up calculations
with Pool(processes=8) as pool:  # Adjust the number of processes based on your CPU
    results = pool.map(process_zip_pair, zip_pairs)

# Save results to a DataFrame
results_df = pd.DataFrame(results, columns=["zip1", "zip2", "distance_km", "duration_min"])
results_df.to_csv("us_zip_code_distances_times.csv", index=False)
```

---

### **Step 3: Optimize and Scale**

#### 3.1 Parallelize with C++ (Optional)
If you need even faster processing, you can write a C++ program to interact with the OSRM backend directly.

```cpp
#include <iostream>
#include <curl/curl.h>
#include <json/json.h>

std::string fetch_route_info(double lon1, double lat1, double lon2, double lat2) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if(curl) {
        char url[256];
        snprintf(url, sizeof(url), "http://localhost:5000/route/v1/driving/%.6f,%.6f;%.6f,%.6f?overview=false", lon1, lat1, lon2, lat2);
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

int main() {
    double lon1 = -122.42, lat1 = 37.78;  // Example: San Francisco
    double lon2 = -118.24, lat2 = 34.05;  // Example: Los Angeles

    std::string response = fetch_route_info(lon1, lat1, lon2, lat2);
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;

    if (Json::parseFromStream(reader, response, &root, &errs)) {
        double distance = root["routes"][0]["distance"].asDouble() / 1000;  // km
        double duration = root["routes"][0]["duration"].asDouble() / 60;    // minutes
        std::cout << "Distance: " << distance << " km, Duration: " << duration << " min" << std::endl;
    }

    return 0;
}
```

Compile and run the C++ program:
```bash
g++ -o osrm_query osrm_query.cpp -lcurl -ljsoncpp
./osrm_query
```

#### 3.2 Use a Database for Storage
Store the results in a database (e.g., SQLite, PostgreSQL) for efficient querying and analysis.

```python
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("us_zip_distances.db")
cursor = conn.cursor()

# Create a table to store distances and times
cursor.execute("""
CREATE TABLE IF NOT EXISTS zip_distances (
    zip1 TEXT,
    zip2 TEXT,
    distance_km REAL,
    duration_min REAL
)
""")

# Insert results into the database
results_df.to_sql("zip_distances", conn, if_exists="append", index=False)
conn.close()
```

---

### **Step 4: Visualize or Analyze the Data**
Use Python libraries like `matplotlib` or `folium` to visualize the dataset.

```python
import matplotlib.pyplot as plt

# Example: Plot a histogram of driving times
plt.hist(results_df["duration_min"], bins=50)
plt.xlabel("Driving Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Driving Times Between Zip Codes")
plt.show()
```

---

### **Notes**
1. **Data Size**: The dataset will be large, so consider using a distributed database or cloud storage.
2. **OSRM Performance**: Ensure the OSRM backend is running on a powerful machine to handle many requests.
3. **OpenStreetMap Updates**: Regularly update the OSM data to reflect changes in road networks.

By combining OSRM's C++ backend with Python for data handling, you can efficiently build a dataset of geographical distances and driving times between zip codes across the continental US.