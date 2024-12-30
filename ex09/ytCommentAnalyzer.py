import os
import argparse
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class YouTubeCommentAnalyzer:
    def __init__(self, api_key):
        """
        Initialize the analyzer with your YouTube Data API key
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def extract_video_id(self, url):
        """
        Extract video ID from YouTube URL
        """
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        return video_id

    def get_subtitles(self, video_id):
        """
        Download and process video subtitles
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Combine all subtitle text
            full_text = ' '.join([entry['text'] for entry in transcript])
            return full_text
        except Exception as e:
            print(f"Error getting subtitles: {str(e)}")
            return None

    def get_comments(self, video_id):
        """
        Download all comments for the video
        """
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Adjust as needed
                textFormat="plainText"
            )
            
            while request:
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'comment_id': item['id'],
                        'text': comment['textDisplay'],
                        'date': comment['publishedAt'],
                        'likes': comment['likeCount']
                    })
                
                # Get the next page of comments
                request = self.youtube.commentThreads().list_next(request, response)
                
        except Exception as e:
            print(f"Error getting comments: {str(e)}")
        
        return comments

    def filter_relevant_comments(self, comments, subtitle_text, similarity_threshold=0.1):
        """
        Filter comments based on relevance to video content
        """
        if not subtitle_text or not comments:
            return []
        
        # Prepare texts for comparison
        texts = [subtitle_text] + [comment['text'] for comment in comments]
        
        # Calculate TF-IDF matrices
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate similarity between subtitles and each comment
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Filter comments based on similarity threshold
            relevant_comments = []
            for comment, similarity in zip(comments, similarities):
                if similarity > similarity_threshold:
                    comment['relevance_score'] = similarity
                    relevant_comments.append(comment)
            
            return relevant_comments
            
        except Exception as e:
            print(f"Error in filtering comments: {str(e)}")
            return []

    def save_to_csv(self, comments, output_file='relevant_comments.csv'):
        """
        Save filtered comments to CSV file
        """
        try:
            df = pd.DataFrame(comments)
            df = df.sort_values('likes', ascending=False)
            
            # Format the DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df = df[['comment_id', 'date', 'likes', 'text', 'relevance_score']]
            
            df.to_csv(output_file, index=False)
            print(f"Comments saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")

    def analyze_video(self, video_id, output_file='relevant_comments.csv', similarity_threshold=0.1):
        """
        Main function to analyze a YouTube video
        """
        # Get subtitles
        print("Downloading subtitles...")
        subtitle_text = self.get_subtitles(video_id)
        if not subtitle_text:
            print("No subtitles available")
            return
        
        # Get comments
        print("Downloading comments...")
        comments = self.get_comments(video_id)
        if not comments:
            print("No comments found")
            return
        
        # Filter relevant comments
        print("Filtering relevant comments...")
        relevant_comments = self.filter_relevant_comments(
            comments, 
            subtitle_text, 
            similarity_threshold
        )
        
        # Save to CSV
        print(f"Saving {len(relevant_comments)} relevant comments...")
        self.save_to_csv(relevant_comments, output_file)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze YouTube video comments and subtitles')
    parser.add_argument('-i', help='YouTube video ID')
    parser.add_argument('-a', required=True, help='YouTube Data API key')
    parser.add_argument('-o', default='relevant_comments.csv', help='Output CSV file path')
    parser.add_argument('-t', type=float, default=0.1, help='Similarity threshold (default: 0.1)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize analyzer and process video
    analyzer = YouTubeCommentAnalyzer(args.ap)
    analyzer.analyze_video(args.i, args.o, args.t)

if __name__ == "__main__":
    main()
