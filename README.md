# Lazy Professor
## Overview
Lazy Professor is intended to be a research paper curator. Because of the large volumes of research papers that are uploaded to Arxiv every day, it is a task to find researech papers that are new and meaningful to you. On top of that there are no offical avenues for finding research papers; some of the popular ways are through Reddit, Twitter, articles on things like Medium, or the tedious search through Arxiv.

## Data Collection
Every day papers are collected at midnight and their data, such as title, authors, date published, and the abstract are all stored in a database. I used MongoDB for this project because it was the most farmiliar and suited all of my needs for a smaller application. In the future if this has to get scaled up then I will have to transfer all of the data to another database. 

## Modeling
### Data Representations
The only data I use for reccomendations pertaining to papers is the title, abstract, and categories. Abstracts and titles have to be cleaned up as their text contains artifacts of latex code and other unnecessary things like links that would impact embeddings. Categories are transformed into a one hot encoding. 

### Simulation
In order to make the neural reccomendation system work it is necessary to have a lot of click data which there isnt any current dataset keeping track of that. To compensate I simulated users by randomly selecting a paper and then chosing papers that are close in representation space filtered by category. The quality of the simulation has had a great impact on the reccomendation quality which is quite obvious.

### Neural Reccomendation Model
Using the simulation data, users clicks were aggregated using a mean of the paper representations. The models task was to take in the users average vector and the paper in question and as a classification task, predict wether or not the user would click on the paper. As more data becomes available, I will use a model similar to the YouTube reccomendation algorithm in order to predict user vectors. However as of now, I cant imagine the volume would be big enough to switch up the model archetecture.

## Website
I used Flask as the backend because I found it easier to integrate with the models which were also served in Python. There are many other features that I have to implement in the front end but am waiting for because I want to see if the modeling process returns results that are satisfactory.

## Future Work
- Collaberator Page
- Saved Papers
- Comments section
- Personal Notes on the paper
- Section where you can attach other content such as github links, youtube videos and so on
    - Also have a report feature that can moderate content
    - Likes, dislikes
