# MOVIE RECOMMENDER SYSTEM
 
 ### Authors
 * Jeremy Waiguru
 * Paul Ngatia
 * Winfred Bundi
 * Mercy Kiragu

 ## Business Overview
* Objective: Develop a versatile movie recommendation system using the following  techniques ;
* Simple Recommender: Ranks movies by popularity using metrics like ratings count and user reviews to generate a list of top-rated movies.
* Collaborative Filtering: The main recommender, analyzing user behavior and preferences to suggest movies based on similar users or movies.
* Hybrid Recommendation model: Addresses the cold-start problem by recommending movies based on intrinsic features like genre, director, and actors, useful for new users and items.
* Evaluation: Success measured using RMSE and MAE scores to ensure effectiveness and accuracy.

## Problem Statement
BingeMax needs to enhance user satisfaction and engagement by developing a robust movie recommendation system. This system will suggest the top 5 movies to each user based on their ratings of other films, utilizing the MovieLens "small" dataset. 
The effectiveness of this system is crucial for improving user interaction, retention, and overall engagement on the platform.

## Data Understanding
Dataset: MovieLens
Files: movies.csv, ratings.csv, links.csv, tags.csv
Key Variables:
userId: Identifier for the user
movieId: Identifier for the movie
rating: User rating (e.g., 1 to 5 stars)
timestamp: When the rating was given (Unix epoch format)
title: Movie title
genres: List of genres for the movie
Source: MovieLens Dataset

## Modelling Process
* For our modelling we used 

## Evaluation 
In the intial modeling, SVD produced an RMSE of 0.8153, KNN Baseline produced an RMSE of 0.8191 and KNN with Means had an RMSE of 0.8409 while KNN Basic produced an RMSE of 0.9081

Hybrid Recommender system was the best performing model with initial RMSE of 0.87.

By using the best parametres, SVD improved a bit and posted an RMSE of 0.86
