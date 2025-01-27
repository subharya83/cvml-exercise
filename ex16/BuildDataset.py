import os
import argparse
import csv
import googleapiclient.discovery
import googleapiclient.errors

# Set up the YouTube Data API client
def get_youtube_client():
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)
    return youtube

# Function to search for videos based on a query and filter by likes
def search_videos(youtube, query, max_results=10, min_likes=100):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        order="viewCount"  # Sort by view count to get popular videos
    )
    response = request.execute()

    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        
        # Get video statistics to filter by likes
        stats_request = youtube.videos().list(
            part="statistics",
            id=video_id
        )
        stats_response = stats_request.execute()
        like_count = int(stats_response['items'][0]['statistics'].get('likeCount', 0))
        
        if like_count >= min_likes:
            videos.append({
                'video_id': video_id,
                'title': video_title,
                'likes': like_count,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            })
    
    return videos

# Main function to find videos for each camera motion class
def find_videos_for_camera_motions(k=5, min_likes=100, output_file="output.csv"):
    youtube = get_youtube_client()
    
    # Define the camera motion classes and their corresponding search queries
    camera_motions = {
        "aerial": "aerial shot",
        "bird-eye": "bird eye view",
        "crane": "crane shot",
        "dolly": "dolly shot",
        "establishing": "establishing shot",
        "pan": "pan shot",
        "tilt": "tilt shot",
        "zoom": "zoom shot"
    }
    
    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['shot_type', 'URL', 'likes', 'YTID_outputfile.mp4']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Iterate over each camera motion class
        for motion, query in camera_motions.items():
            print(f"Searching for {motion} videos...")
            videos = search_videos(youtube, query, max_results=k, min_likes=min_likes)
            print(f"Found {len(videos)} videos for {motion}.")
            
            # Write the results to the CSV file
            for video in videos:
                writer.writerow({
                    'shot_type': motion,
                    'URL': video['url'],
                    'likes': video['likes'],
                    'YTID_outputfile.mp4': f"YTID_{video['video_id']}.mp4"
                })
    
    print(f"Results saved to {output_file}")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Find YouTube videos for different camera motion classes.")
    parser.add_argument("-k", type=int, default=5, help="Number of videos to find for each class.")
    parser.add_argument("-o", type=str, default="output.csv", help="Output CSV file name.")
    parser.add_argument("-l", type=int, default=100, help="Minimum number of likes to filter videos.")
    return parser.parse_args()

# Run the script
if __name__ == "__main__":
    args = parse_arguments()
    find_videos_for_camera_motions(k=args.k, min_likes=args.l, output_file=args.o)