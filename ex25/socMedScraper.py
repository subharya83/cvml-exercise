import argparse
import json
import datetime
import os
import time
import random
from datetime import datetime

# Import required libraries
try:
    import praw
    import tweepy
    import linkedin_api
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "praw", "tweepy", "linkedin_api"])
    import praw
    import tweepy
    import linkedin_api

class SocialMediaScraper:
    def __init__(self, config_file, output_file):
        self.config = self.parse_config(config_file)
        self.output_file = output_file
        self.credentials = self.load_credentials()
        self.results = []

    def parse_config(self, config_file):
        """Parse the configuration file."""
        config = {}
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        
        # Convert date strings to datetime objects
        if 'start_date' in config:
            config['start_date'] = datetime.strptime(config['start_date'], '%Y-%m-%d')
        if 'end_date' in config:
            config['end_date'] = datetime.strptime(config['end_date'], '%Y-%m-%d')
        
        return config

    def load_credentials(self):
        """Load credentials from environment variables or a credentials file."""
        credentials = {
            # Reddit credentials
            'reddit_client_id': os.environ.get('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.environ.get('REDDIT_CLIENT_SECRET'),
            'reddit_username': os.environ.get('REDDIT_USERNAME'),
            'reddit_password': os.environ.get('REDDIT_PASSWORD'),
            
            # Twitter credentials
            'twitter_bearer_token': os.environ.get('TWITTER_BEARER_TOKEN'),
            'twitter_consumer_key': os.environ.get('TWITTER_CONSUMER_KEY'),
            'twitter_consumer_secret': os.environ.get('TWITTER_CONSUMER_SECRET'),
            'twitter_access_token': os.environ.get('TWITTER_ACCESS_TOKEN'),
            'twitter_access_token_secret': os.environ.get('TWITTER_ACCESS_TOKEN_SECRET'),
            
            # LinkedIn credentials
            'linkedin_username': os.environ.get('LINKEDIN_USERNAME'),
            'linkedin_password': os.environ.get('LINKEDIN_PASSWORD')
        }
        
        # If credentials file exists, load from there
        if os.path.exists('credentials.json'):
            with open('credentials.json', 'r') as f:
                file_credentials = json.load(f)
                # Update missing credentials
                for k, v in file_credentials.items():
                    if not credentials.get(k):
                        credentials[k] = v
        
        return credentials

    def scrape_reddit(self):
        """Scrape Reddit for the given topic and date range."""
        print("Scraping Reddit...")
        reddit = praw.Reddit(
            client_id=self.credentials.get('reddit_client_id'),
            client_secret=self.credentials.get('reddit_client_secret'),
            username=self.credentials.get('reddit_username'),
            password=self.credentials.get('reddit_password'),
            user_agent='SocialMediaScraper/1.0'
        )
        
        topic = self.config.get('topic')
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        
        # Convert datetime to timestamp for Reddit's API
        start_timestamp = int(start_date.timestamp()) if start_date else None
        end_timestamp = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())
        
        # Search for submissions
        for submission in reddit.subreddit('all').search(topic, limit=100):
            created_time = datetime.fromtimestamp(submission.created_utc)
            
            # Check if the submission is within the date range
            if start_timestamp and submission.created_utc < start_timestamp:
                continue
            if submission.created_utc > end_timestamp:
                continue
            
            submission_data = {
                'platform': 'reddit',
                'id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'url': f"https://www.reddit.com{submission.permalink}",
                'author': str(submission.author),
                'subreddit': submission.subreddit.display_name,
                'score': submission.score,
                'created_utc': submission.created_utc,
                'created_date': created_time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_comments': submission.num_comments
            }
            
            self.results.append(submission_data)
            
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"Found {len(self.results)} Reddit posts about '{topic}'")

    def scrape_twitter(self):
        """Scrape Twitter for the given topic and date range."""
        print("Scraping Twitter...")
        
        # Set up Twitter API v2 client
        client = tweepy.Client(
            bearer_token=self.credentials.get('twitter_bearer_token'),
            consumer_key=self.credentials.get('twitter_consumer_key'),
            consumer_secret=self.credentials.get('twitter_consumer_secret'),
            access_token=self.credentials.get('twitter_access_token'),
            access_token_secret=self.credentials.get('twitter_access_token_secret')
        )
        
        topic = self.config.get('topic')
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        
        # Format dates for Twitter's API
        start_time = start_date.strftime('%Y-%m-%dT%H:%M:%SZ') if start_date else None
        end_time = end_date.strftime('%Y-%m-%dT%H:%M:%SZ') if end_date else None
        
        # Search for tweets
        tweets = client.search_recent_tweets(
            query=topic,
            max_results=100,
            start_time=start_time,
            end_time=end_time,
            tweet_fields=['created_at', 'public_metrics', 'author_id'],
            user_fields=['username', 'name'],
            expansions=['author_id']
        )
        
        if tweets.data:
            # Create a dictionary to map user IDs to user objects
            users = {user.id: user for user in tweets.includes['users']} if 'users' in tweets.includes else {}
            
            for tweet in tweets.data:
                author = users.get(tweet.author_id)
                username = author.username if author else "unknown"
                name = author.name if author else "Unknown User"
                
                tweet_data = {
                    'platform': 'twitter',
                    'id': tweet.id,
                    'text': tweet.text,
                    'author_id': tweet.author_id,
                    'username': username,
                    'name': name,
                    'created_at': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'retweet_count': tweet.public_metrics['retweet_count'] if hasattr(tweet, 'public_metrics') else 0,
                    'like_count': tweet.public_metrics['like_count'] if hasattr(tweet, 'public_metrics') else 0
                }
                
                self.results.append(tweet_data)
        
        print(f"Found {len(self.results)} Twitter posts about '{topic}'")

    def scrape_linkedin(self):
        """Scrape LinkedIn for the given topic and date range."""
        print("Scraping LinkedIn...")
        
        # Login to LinkedIn
        api = linkedin_api.Linkedin(
            self.credentials.get('linkedin_username'),
            self.credentials.get('linkedin_password')
        )
        
        topic = self.config.get('topic')
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        
        # Search for posts
        search_results = api.search_people(topic)
        
        for result in search_results:
            # Get profile information
            profile_id = result.get('public_id')
            if not profile_id:
                continue
                
            profile = api.get_profile(profile_id)
            
            # Get recent posts from this profile
            try:
                posts = api.get_profile_posts(profile_id, count=10)
                
                for post in posts:
                    # Extract the post date
                    post_date_str = post.get('created', {}).get('time')
                    if not post_date_str:
                        continue
                        
                    post_date = datetime.fromtimestamp(int(post_date_str) / 1000)
                    
                    # Check if the post is within the date range
                    if start_date and post_date < start_date:
                        continue
                    if end_date and post_date > end_date:
                        continue
                    
                    # Check if the post content contains the topic
                    post_content = post.get('commentary', {}).get('text', '')
                    if topic.lower() not in post_content.lower():
                        continue
                    
                    post_data = {
                        'platform': 'linkedin',
                        'id': post.get('entityUrn'),
                        'text': post_content,
                        'author': f"{profile.get('firstName')} {profile.get('lastName')}",
                        'author_id': profile_id,
                        'created_date': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'likes': post.get('socialDetail', {}).get('totalSocialActivityCounts', {}).get('numLikes', 0),
                        'comments': post.get('socialDetail', {}).get('totalSocialActivityCounts', {}).get('numComments', 0)
                    }
                    
                    self.results.append(post_data)
                    
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(2, 4))
                    
            except Exception as e:
                print(f"Error getting posts for {profile_id}: {e}")
        
        print(f"Found {len(self.results)} LinkedIn posts about '{topic}'")

    def scrape(self):
        """Scrape all configured platforms."""
        if 'topic' not in self.config:
            print("Error: 'topic' not specified in the configuration file.")
            return
        
        # Scrape platforms
        if self.config.get('scrape_reddit', 'true').lower() == 'true':
            try:
                self.scrape_reddit()
            except Exception as e:
                print(f"Error scraping Reddit: {e}")
        
        if self.config.get('scrape_twitter', 'true').lower() == 'true':
            try:
                self.scrape_twitter()
            except Exception as e:
                print(f"Error scraping Twitter: {e}")
        
        if self.config.get('scrape_linkedin', 'true').lower() == 'true':
            try:
                self.scrape_linkedin()
            except Exception as e:
                print(f"Error scraping LinkedIn: {e}")
        
        # Save results to JSON file
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Scraping complete! Found {len(self.results)} total results.")
        print(f"Results saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description='Scrape social media platforms for a given topic.')
    parser.add_argument('-c', '--config', required=True, help='Path to the configuration file')
    parser.add_argument('-o', '--output', default='output.json', help='Path to the output JSON file')
    args = parser.parse_args()
    
    scraper = SocialMediaScraper(args.config, args.output)
    scraper.scrape()

if __name__ == "__main__":
    main()