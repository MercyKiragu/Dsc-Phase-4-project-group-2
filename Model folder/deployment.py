import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict
import uvicorn
from surprise.prediction_algorithms import SVD
import pandas as pd
from surprise import SVD, KNNBasic, KNNWithMeans, Reader, Dataset,KNNBaseline
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from surprise.model_selection import GridSearchCV


#  loading datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Merge movies with ratings
movies_ratings = pd.merge(ratings, movies, on='movieId')

# Display the first few rows of the merged dataset
movies_ratings.sample(5)


# Define the Reader and load the data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(movies_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define and evaluate models
def evaluate_models(data):
    # Define a list of models to evaluate

    models = [
        SVD(),
        KNNBasic(sim_options={'name': 'cosine', 'user_based': True}),
        KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True}),
        # knns.KNNBaseline(sim_options={'name': 'cosine', 'user_based': True})
    ]

    results = []
    for model in models:
        # Perform cross-validation
        cv_results = cross_validate(model, data, measures=['RMSE'], cv=5, verbose=False)

        # Get the average RMSE from cross-validation
        rmse = cv_results['test_rmse'].mean()

        # Store the model and its performance
        results.append({'model': model.__class__.__name__, 'rmse': rmse})

    # Sort the results based on the RMSE in ascending order
    sorted_results = sorted(results, key=lambda x: x['rmse'])

    # Print the results
    for result in sorted_results:
        print(f"Model: {result['model']}, RMSE: {result['rmse']:.4f}")

    # Select the best performing model
    best_model = sorted_results[0]['model']
    print(f"Best performing model: {best_model}")
    print(sorted_results[0])

# Evaluate models
evaluate_models(data)

# Regularize the SVD model
params = {'n_factors': [20, 50, 100], 'reg_all': [0.02, 0.05, 0.1]}
g_s_svd = GridSearchCV(SVD, param_grid=params, n_jobs=-1)
g_s_svd.fit(data)
best_params = g_s_svd.best_params

print(g_s_svd.best_score)
print(g_s_svd.best_params)

n_factors = best_params['rmse']['n_factors']
reg_all = best_params['rmse']['reg_all']
# Train SVD model with the best RMSE parameters
svd = SVD(n_factors=n_factors, reg_all=reg_all)
svd.fit(trainset)


#  Deploying the model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd,f)

app = FastAPI()

# Load the trained model
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the request body
class PredictionRequest(BaseModel):
    userid: int
    movieid: int

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Predict the rating
    prediction = model.predict(request.user_id, request.movie_id)
    return {"user_id": request.user_id, "movie_id": request.movie_id, "prediction": prediction.est}

if __name__ == "__main__":
    #  freeze_support()
     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)