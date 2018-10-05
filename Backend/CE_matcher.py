"""
Basically we're going to receive the following from the data initially:

Database_Incoming ===> {name, age, gender, distance, interests, rating}  <=== young people

Database_Incoming ===> {location, interests} <=== old people

 >> glove function has to be multi-threaded in order to prevent bottle-neck in the fact that initially 20 names has
                                                                                                        to be loaded

 >> once interests have been matched and the profile has common interests we'll move to the matching part

 STEP 1: We need to find the ppl with the required gender and location <======> glean all the matching entries
                                                                                                        within 20kms

 STEP 2: Pass through interest_matcher based on the profiles that are having the common interests, we'll get the
                    profiles that need to be matched <==== over here we can now use multi-threading

 STEP 3: build json of the profiles that have matched and post

"""

from flask import Flask, request
import json
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
import sqlite3
from multiprocessing.dummy import Pool as ThreadPool
import queue
from api.util import most_similar
import numpy as np
import math

db_old = sqlite3.connect("old.db")  # database for old users
cursor_old = db_old.cursor()


@app.route('/', methods=['POST'])
def receive_old():
    global imei
    imei = request.json['imei']  # we need to assign userID to everyone as names may
# be common with other users
    return json_string


command_old_people = ('''SELECT latitude, longitude, interests, gender_preference
                         FROM Old
                         WHERE imei = ?;''')  # sql command

# I get a tuple here containing the required information based on the name received
row = cursor_old.execute(command_old_people, [imei])

# now based on the old people location, gender_preference and interests we need to find the matching profiles
old_gender_preference = row[3]
old_latitude = row[0]
old_longitude = row[1]

db_young = sqlite3.connect("young.db")  # database for companionship experts
cursor_young = db_young.cursor()
command_young_people = ('''SELECT imei, gender, interests, rating, name, latitude, longitude, profile_pic
                            id, (
                             6371 * acos( 
                             cos(radians(?))
                             *cos(radians(latitude))
                             *cos(radians(longitude) - radians(?))
                             +sin(radians(?))
                             *sin(radians(latitude))
                             )
                            ) AS distance
                            FROM Young
                            HAVING distance<20;
                            ORDER BY distance
                            LIMIT 0, 20;''')

row1 = cursor_young.execute(command_young_people, [old_latitude, old_longitude, old_latitude])
young_userID = row1[0]
young_gender = row1[1]
young_interests = row1[2]
young_ratings = row1[3]
young_name = row1[4]
young_lat = row1[5]
young_long = row1[6]
young_profile_pic = row1[7]
# display_index will store the indexes of the required profiles curated after comparing gender preferences
display_index = []
x = 0  # temporary variables to build display_index
for i in young_gender:
    if young_gender[i] == old_gender_preference:
        # making a nested list containing necessary items
        young_dist = math.sqrt(pow((old_latitude-young_lat[x]), 2) + pow((old_longitude-young_long[x]), 2))
        display_index.append([young_interests[x], young_userID[x], young_ratings[x], young_gender[i], young_name[x],
                              young_dist, young_profile_pic[x]])
    x = x + 1

pool = ThreadPool(8)
interests_list = []
for i in display_index:
    interests_list.append([i[0], i[1]])  # inserting the interests and user_id of every person into a list
# match_queue = queue.Queue(maxsize=20)  # this queue will store the final list of CEs to be displayed

"""Here on out the GloVe processing starts"""
data = np.load('glove.model.npz')
word_embeddings_array = data['word_embeddings_array']
word_to_index = data['word_to_index'].item()
index_to_word = data['index_to_word'].item()


def interest_matcher(interests):
    """
    This function uses GloVe to find common interests

    Algorithm:
            >> Every individual has 10 interests.
                interests ===> stores the list of interests per individual
                interest ===> an individual interest in the 'interests' list
            >> An interest is considered to match if the similarity is greater than or equal to 0.5
            >> if at least 3 interests match, the CE is added to the list
            >> the CEs are ordered in descending order, with most number of matches at the top of the list
            >> the number of matches will be stored in the 'display_index' list as 'score'
            >> the list will be sorted by 'score' and added to the queue
            >> the queue will be jsonified and sent to the app frontend
            >> add the distance and score to sort finally based upon the 2 considered together
    :param interests: list of interests of each individual Companionship Expert
    :return:
    """
    matched_CE = {"user_id", "name", "gender", "distance", "profile_pic", "interests", "rating"}
    index_tracker = 0
    common_interests = ""
    for interest in interests:
        index_tracker = index_tracker+1
        similarity_list = most_similar(word_embeddings_array, word_to_index, index_to_word, interest, result_num=5)
        summ = 0
        for item in similarity_list:
            if item[1] >= 0.5:
                summ = summ+1
        if summ >= 3:
            common_interests = common_interests + "," + interest
    matched_CE["user_id"] = display_index[1][index_tracker]  # haven't added the distance yet. might do it later
    matched_CE["name"] = display_index[4][index_tracker]
    matched_CE["gender"] = display_index[3][index_tracker]
    matched_CE["distance"] = display_index[0]
    matched_CE["profile_pic"] = display_index[6]
    matched_CE["interests"] = common_interests
    matched_CE["rating"] = display_index[2]
    return matched_CE


results = pool.map(interest_matcher, interests_list[0])  # multi-threading interest_matcher
json_string = json.dumps(results)
