# 🌐 Social Media Observatory

A powerful multi-platform social data scraper for research, sentiment analysis, and trend tracking.

## Overview

Social Media Observatory is an advanced tool designed for machine learning researchers, data scientists, and computational social scientists. This tool enables the systematic collection of topical data across Reddit, Twitter, and LinkedIn, providing a rich corpus for various analytical applications.

## 🚀 Features

- **Multi-platform Scraping**: Collect data simultaneously from Reddit, Twitter, and LinkedIn
- **Configurable Data Collection**: Specify topics, date ranges, and platforms via a simple configuration file
- **Structured Data Output**: All data is harmonized and exported in JSON format for easy integration with data processing pipelines
- **Rate-limit Safe**: Built-in random delays and error handling to respect platform API limitations
- **Flexible Authentication**: Support for both environment variables and credential files

## Organization
```
+---------------------------------------------------+
|                SocialMediaScraper                 |
|---------------------------------------------------|
| - config: dict                                    |
| - output_file: str                                |
| - credentials: dict                               |
| - results: list                                   |
|---------------------------------------------------|
| + __init__(config_file, output_file)              |
| + parse_config(config_file) -> dict               |
| + load_credentials() -> dict                      |
| + scrape_reddit()                                 |
| + scrape_twitter()                                |
| + scrape_linkedin()                               |
| + scrape()                                        |
+---------------------------------------------------+
```

## Workflow

```
+-------------------+       +-------------------+       +-------------------+
|   Configuration   |       |   Credentials     |       |   Results         |
|   File (config)   |       |   (env/file)      |       |   (output.json)   |
+-------------------+       +-------------------+       +-------------------+
          |                         |                         ^
          v                         v                         |
+-------------------+       +-------------------+             |
|  parse_config()   |       | load_credentials()|             |
+-------------------+       +-------------------+             |
          |                         |                         |
          v                         v                         |
+-------------------------------------------------------------+
|                SocialMediaScraper Class                     |
|-------------------------------------------------------------|
| - config: dict                                              |
| - output_file: str                                          |
| - credentials: dict                                         |
| - results: list                                             |
|-------------------------------------------------------------|
| + scrape()                                                  |
|   |                                                         |
|   |--> scrape_reddit()                                      |
|   |       |                                                 |
|   |       +--> Fetch Reddit posts                           |
|   |       +--> Store in results                             |
|   |                                                         |
|   |--> scrape_twitter()                                     |
|   |       |                                                 |
|   |       +--> Fetch Twitter tweets                         |
|   |       +--> Store in results                             |
|   |                                                         |
|   |--> scrape_linkedin()                                    |
|           |                                                 |
|           +--> Fetch LinkedIn posts                         |
|           +--> Store in results                             |
|                                                             |
|   +--> Save results to output_file                          |
+-------------------------------------------------------------+
```

## 📋 Requirements

- Python 3.7+
- Required packages: praw, tweepy, linkedin_api (automatically installed if missing)
- API credentials for each platform

## 📝 Configuration

Create a `config.txt` file with your search parameters:

```
topic=artificial intelligence
start_date=2023-01-01
end_date=2023-12-31
scrape_reddit=true
scrape_twitter=true
scrape_linkedin=true
```

Set up your credentials either as environment variables:

```bash
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
export REDDIT_USERNAME=your_username
export REDDIT_PASSWORD=your_password
# Add similar entries for Twitter and LinkedIn
```

Or create a `credentials.json` file:

```json
{
  "reddit_client_id": "your_client_id",
  "reddit_client_secret": "your_client_secret",
  "reddit_username": "your_username",
  "reddit_password": "your_password",
  "twitter_bearer_token": "your_bearer_token",
  "twitter_consumer_key": "your_consumer_key",
  "twitter_consumer_secret": "your_consumer_secret",
  "twitter_access_token": "your_access_token",
  "twitter_access_token_secret": "your_access_token_secret",
  "linkedin_username": "your_email",
  "linkedin_password": "your_password"
}
```

## 🚀 Usage

```bash
python socMedScraper.py -c config.txt -o results.json
```

## 🧠 Applications for Machine Learning Researchers

### 1. Cross-Platform Discourse Analysis
Track how topics evolve and spread across different social platforms. Identify which platforms lead in topic introduction and how narratives transform as they cross platform boundaries.

### 2. Multi-modal Sentiment Analysis
Build models that combine text, engagement metrics, and platform context to create more nuanced sentiment analysis tools that understand platform-specific communication patterns.

### 3. Public Response to Scientific Publications
Track how scientific papers are discussed across platforms after publication. Measure public sentiment, identify misunderstandings, and analyze how scientific communication could be improved.

### 4. Topic Evolution Visualization
Create dynamic visualizations showing how discussions around specific topics (like AI ethics or climate science) evolve over time across different platforms and communities.

### 5. Trend Prediction Systems
Develop early warning systems for emerging trends by analyzing cross-platform data patterns. Find signals that predict which topics might spread beyond niche communities to mainstream awareness.

### 6. Influence Network Mapping
Identify key influencers across platforms and visualize how information flows through cross-platform networks during emerging events or discussions.

### 7. Context-Aware LLM Training
Create platform-specific training data for fine-tuning language models to better understand the unique linguistic patterns and contexts of different social media platforms.

## 📊 Example Research Projects

- "Cross-Platform Discourse on Climate Change: A Comparative Analysis of Expert vs. Public Communication"
- "Predicting Research Impact: Social Media Mentions as Early Indicators of Scientific Paper Citations"
- "Platform-Specific Semantic Drift: How Technical Terms Change Meaning Across Social Media Ecosystems"
- "Temporal Dynamics of Public Response to AI Advancement Announcements"

## 📚 Ethical Considerations

This tool is designed for academic research purposes. Users should:
- Respect platform Terms of Service
- Consider privacy implications and anonymize data when appropriate
- Obtain IRB approval when necessary for human subjects research
- Consider the bias implications of platform-specific demographic skews
