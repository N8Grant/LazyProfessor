import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from datetime import date
import re
import torch
import matplotlib.pyplot as plt
from pacmap import PaCMAP
from transformers import AutoTokenizer, AutoModel
import arxiv
from sklearn.neighbors import KNeighborsTransformer
from IPython.display import clear_output
from datetime import date
from datetime import timedelta
from pymongo import MongoClient
from datetime import datetime
from sklearn.neighbors import BallTree,KernelDensity
from torch import nn
import hashlib
import sys
device = 'cpu'
torch.device(device)


URI = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false'

def get_popular_papers(n_papers=4):
    client = MongoClient('mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false')

    db = client.papers.papers
    papers = db.find({"date" : {"$gt": datetime.now()-timedelta(days=2)}},{"title":1,
                                                                          "abstract":1,
                                                                          "date":1,
                                                                          "number_of_clicks":1})
    paper_list = []
    
    for p in list(papers):
        paper_list.append((p["_id"],p["number_of_clicks"]))
        
    paper_list = sorted(paper_list, key=lambda t: t[1],reverse=True)
    papers = db.find({"_id": {"$in" : [t[0] for t in paper_list[:n_papers]]}})
    return list(papers)

def get_papers_after_date(last_date):
    now = datetime.now()
    delta = (now - last_date).days
    
    number_of_searches = round((delta*300)/2000)
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = AutoModel.from_pretrained('allenai/longformer-base-4096')
    
    paper_data = []
    
    for i in number_of_searches:
        search = arxiv.Search(
          query = "computer science",
          max_results = delta*300,
          sort_by = arxiv.SortCriterion.SubmittedDate
        )

        for result in search.results():
            ## To be changed
            paper_date = (result.published+timedelta(days=1)).date()
            ##


            if paper_date < last_date.date():
                break

            category_vector = [result.categories[i] if i<len(result.categories) else "Nafin" for i in range(3)]


            abstract = result.summary.replace('\n',' ').replace('\\&','').replace('\\%','')

            temp = [hash(result.title),result.title,result.published,abstract,result.pdf_url]
            temp.extend(category_vector)

            temp.append(get_embedding(model,tokenizer,result.title).tolist())
            temp.append(get_embedding(model,tokenizer,abstract).tolist())
            temp.append(0)

            paper_data.append(temp)

    df = pd.DataFrame(data=paper_data, columns=['_id','title','date', 'abstract',"PDF URL",'category1','category2','category3',"title_embedding","abstract_embedding","number_of_clicks"])
    
    return df

def get_last_two_weeks():
    client = MongoClient(URI)

    db = client.papers.users
    user = db.find_one({'_id':0})
    
    # Get paperst hat had been missed
    df = get_papers_after_date(datetime.now()-timedelta(days=14))
    
    # Add them to the database
    db=client.papers.papers
    db.insert_many(df.to_dict('records'))
    
    # Remove old papers
    db.delete_many({"$and":[
                            {'$lt': {'date' : datetime.now()-timedelta(days=31)}},
                            {"$eq": {"number_of_clicks": 0} } ] 
                   })
    
def get_last_days_papers():
    client = MongoClient(URI)

    # Get paperst hat had been missed
    df = get_papers_after_date(datetime.now()-timedelta(days=1))
    
    # Add them to the database
    db=client.papers.papers
    db.insert_many(df.to_dict('records'))
    
    # Remove old papers
    db.delete_many({"$and":[
                            {'$lt': {'date' : datetime.now()-timedelta(days=31)}},
                            {"$eq": {"number_of_clicks": 0} } 
                           ] 
                   })
    
def dummy_user():
    client = MongoClient(URI)
    db = client.papers.users
    db.insert_one({"_id":0,"last_date_updated":datetime.now(),"recently_viewed":[],
                   "papers_read":0,"average_title":[[0 for i in range(768)]],"average_abstract":[[0 for i in range(768)]],
                   "title_weights":[.5]})
    
def add_user(email,username,password_hash):
    client = MongoClient(URI)
    db = client.papers.users
    
    email_exists = db.count_documents({"email":email},limit=1)
    if email_exists != 0:
        return "email_in_use"
    username_exists = db.count_documents({"username":username},limit=1)
    if username_exists != 0:
        return "username_in_use"
    
    result = db.insert_one({"email":email,
                           "username":username,
                           "password":password_hash,
                           "last_date_updated":datetime.now(),
                           "recently_viewed":[],
                           "papers_read":0,
                           "average_title":[[0 for i in range(768)]],
                           "average_abstract":[[0 for i in range(768)]],
                           "title_weights": [.5],
                           "saved_papers": [],
                           "following": [],
                           "followers": []})
    return str(result.inserted_id)
    
def delete_docs():
    client = MongoClient(URI)
    db = client.papers.papers
    db.delete_many({})

def get_abstracts_by_ids(paper_index_map):
    inv_paper_map = {v: k for k, v in paper_index_map.items()}
    
    client = MongoClient(URI)
    
    # get all users recently viewed
    db = client.papers.papers
    
    all_viewed = list(db.find({"_id":{"$in":list(paper_index_map.keys())}},{"abstract":1}))
    
    data = np.chararray(len(all_viewed),itemsize=10000)
    
    new_map = {}
    data = []
    for i in range(len(all_viewed)):
        paper_id = all_viewed[i]["_id"]
        new_map[paper_index_map[paper_id]] = i
        data.append(all_viewed[i]["abstract"])
    
    return np.array(data),new_map

def generate_interaction_matrix():
    client = MongoClient(URI)
    
    # get all users recently viewed
    db = client.papers.users
    
    all_viewed = db.find({},{"recently_viewed":1})
    users_viewed = list(all_viewed)
    
    # get all papers and their ids
    db = client.papers.papers
    
    all_paper_ids = db.find({"number_of_clicks":{"$gt":0}},{"_id":1})
    all_ids = list(all_paper_ids)
    
    # make dictionary between paper id and index in array
    paper_index_map = {}
    
    i=0
    for dic in all_ids:
        paper_index_map[dic["_id"]] = i
        i+=1
    
    # make array of zeroes with shape (users,papers)
    clicked = np.zeros((len(users_viewed),len(all_ids)))
    print(clicked.shape)
    # for all users recently viewed change 0 in array to a 1
    user_index_map = {}
    i=0
    for user in users_viewed:
        user_index_map[user["_id"]] = i
        
        for paper in user["recently_viewed"]:
            if paper in paper_index_map:
                clicked[i,paper_index_map[int(paper)]] = 1
        
        i+=1
    
    return clicked,user_index_map, paper_index_map

def get_paper_by_id(paper_id):
    client = MongoClient(URI)
    db = client.papers.papers
    paper = db.find({"_id": paper_id})
    return paper
    
    
def get_user_by_id(user_id):
    client = MongoClient(URI)
    db = client.papers.users
    user = db.find({"_id": user_id})
    return user

def curate_results(embeddings,average_title,average_abstract,title_weights,num_papers):
    num_results = 40
    ids = [e["_id"] for e in embeddings]
    title_embeddings = [e["title_embedding"][0] for e in embeddings]
    abstract_embeddings = [e["abstract_embedding"][0] for e in embeddings]
    
    ## Get closest title embeddings
    title_tree = BallTree(np.array(title_embeddings))
    title_dist, title_positions = title_tree.query(average_title,k=num_results)
    title_dist, title_positions = title_dist[0].tolist(), title_positions[0].tolist()
    
    ## Get closest abstract embeddings
    abstract_tree = BallTree(np.array(abstract_embeddings))
    abstract_dist, abstract_positions = abstract_tree.query(average_abstract,k=num_results)
    abstract_dist, abstract_positions = abstract_dist[0].tolist(), abstract_positions[0].tolist()
    
    # Get random weights
    if num_papers > 10:
        bandwith = (1/(len(title_weights)))/10
        kde = KernelDensity(bandwidth=bandwith).fit(title_weights)
        sampled_title_weights = kde.sample(len(title_embeddings))
        sampled_title_weights[sampled_title_weights<0] = 0
        sampled_title_weights[sampled_title_weights>1] = 1
    else:
        sampled_title_weights = np.full((len(title_embeddings),1),.5)
    
    ## Compute distances
    combined = {}
    weights = {}
    for i in range(100):
        # add title vector
        e = title_positions[i]
        weights[ids[e]] = sampled_title_weights[i]
        if e not in combined:
            combined[ids[e]] = title_dist[i]*sampled_title_weights[i]
        else:
            combined[ids[e]] += title_dist[i]
            
        # add abstract vector
        e = abstract_positions[i]
        
        if ids[e] not in weights:
            weights[ids[e]] = sampled_title_weights[i]
        
        if e not in combined:
            combined[ids[e]] = abstract_dist[i]*(1-sampled_title_weights[i])
        else:
            combined[ids[e]] += abstract_dist[i]*(1-sampled_title_weights[i])
    
    ## Sort by lowest distance
    combined_dist = [(k,v) for k,v in combined.items()]
    combined_dist = sorted(combined_dist,key=lambda t: t[1])
    return [c[0] for c in combined_dist], weights

def get_papers(user_id):
    client = MongoClient(URI)
    db = client.papers.users
    user = db.find_one({'_id':user_id})
    recently_viewed = user['recently_viewed']
    average_title = user['average_title']
    average_abstract = user['average_abstract']
    title_weights = user['title_weights']
    num_papers = user['papers_read']
    last_use = user['last_use'] b
    
    db = client.papers.papers
   # papers = db.find( {"_id": {"$nin":recently_viewed}})
    
    embeddings = db.find({"date" : {"$gt" : datetime.now()}},{"title_embedding":1,"abstract_embedding":1})
    
    # paper curation
    paper_list, weights = curate_results(list(embeddings), average_title, average_abstract,
                                         np.array(title_weights).reshape(-1, 1),num_papers)
    
    papers = db.aggregate([
    { "$match": {
        "_id": { "$in": paper_list, "$nin": recently_viewed },
    }},
    {"$addFields": {"__order": {"$indexOfArray": [paper_list, "$_id" ]}}},
             {"$sort": {"__order": 1}}])
    
    return {'result':list(papers)},weights

def search_results(query, database):
    client = MongoClient(URI)
    
    if database == "papers":
        db = client.papers.papers
    else:
        db = client.papers.users
        
    retults = db.find(
       { "$text": { "$search": query } },
       { "score": { "$meta": "textScore" } }
    ).sort( { "score": { "$meta": "textScore" } } )
    
    results_list = []
    
    for r in list(results):
        # need to return title or username information along with object ids
        # need to make an index oer titles and usernames
        
        if database =="papers":
            results_list.append({"name": r["title"],"ref":r["_id"]})
        else:
            results_list.append({"name": r["username"],"ref": r["_id"]})
            
    return {"results": results_list}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(model, tokenizer, text):   
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings.numpy()


    
if __name__ == "__main__":
    arg = str(sys.argv[1:][0])
    # task name is "Fetch Papers"
    if arg == "fetch":
        get_last_days_papers()
        print("Done!")
    
    
    