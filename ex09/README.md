## Find relevant comments for a given YouTube video

### Problem statement
For a given youtube video link, 
* Download the subtitles if available
* Download all the comments posted on the video, 
* Filter out all the comments that are irrelevant to the contents of the video (based on the subtitles)
* Rank the rest of the comments into a csv file that has the following fields - comment_id, comment_date_time, comment_number_of_likes

### Code organization

1. Install the required dependencies:
```shell
pip install google-api-python-client youtube-transcript-api pandas scikit-learn
```

2. Get a YouTube Data API key from the Google Cloud Console.

3. Replace `YOUR_YOUTUBE_API_KEY` with your actual API key.

The script does the following:
1. Extracts the video ID from the URL
2. Downloads available subtitles using `youtube_transcript_api`
3. Fetches all comments using the YouTube Data API
4. Uses TF-IDF vectorization and cosine similarity to compare comments with video content
5. Filters out irrelevant comments based on a similarity threshold
6. Saves relevant comments to a CSV file, sorted by number of likes

The output CSV includes:
- comment_id
- date (in datetime format)
- likes (number of likes)
- text (comment content)
- relevance_score (similarity to video content)

Adjust the `similarity_threshold` parameter to make the filtering more or less strict. A higher threshold means comments must be more relevant to be included.