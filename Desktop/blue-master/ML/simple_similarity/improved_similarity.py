import numpy as np
from scipy import stats

def find_user_similarity(input_user, all_users, attr_dict):
    """
        :param input_user: Represents input user data. Format (Gender, Age, Birth Country, Major, preferred joke 1, preferred joke 2, preferred joke type, music genre, movie genre)
        :param all_users: Represents all user data.
        :type input_user: (1 x 9) nparray.
        :type all_user: (m x 9) nparray.
        :return: List of integers represent similarity (hamming distance) between input user
        :rtype: (1 x m) nparray.
    """
    if len(input_user) != 9:
        raise ValueError("input_user has to be (1 x 9) nparray.")
    if all_users.shape[1] != 9:
        raise ValueError("all_user's has to be (n x 9) nparray.")

    all_ages = all_users[:,[attr_dict["age"]]].T[0]
    all_ages = [float(i) for i in all_ages]

    result = list()
    for i in all_users:
        hamming_distance = 0
        for j in range(len(input_user[0])):
            if j == attr_dict["age"]:
                input_percentile = stats.percentileofscore(all_ages, float(input_user[j]))/100
                user_percentile = stats.percentileofscore(all_ages, float(i[j]))/100
                hamming_distance += 1 - abs(input_percentile - user_percentile)
            elif j in attr_dict["preferred_joke"]:
                if input_user[0][j] in i[attr_dict["preferred_joke"]]:
                    hamming_distance += 1
            else:
                if input_user[0][j] == i[j]:
                    hamming_distance += 1
        result.append(hamming_distance)
    return (np.array(result))

if __name__ == "__main__":
    attr_dict = {"gender":0, "age":1, "birth_country":2, "major":3, "preferred_joke":[4, 5], "joke_type":6, "music":7, "movie": 8}
    i = list()
    i.append(['Male', 18, 'USA', 'CS', 121, 145, 'Nerd', 'Jazz', 'SIFI'])
    i.append(['Female', 23, 'USA', 'MATH', 100, 12, 'Puns', 'Classic', 'Comedy'])
    a = list()
    a.append(['Male', 21, 'JAP', 'PHY', 10, 32, 'Nerd', 'Pop'])
    #print(find_user_similarity(np.array(i), np.array(a), attr_dict))

    i = list()
    i.append(['Male', 18, 'USA', 'CS', 121, 145, 'Nerd', 'Jazz', 'SIFI'])
    #print(find_user_similarity(np.array(i), np.array(a), attr_dict))

    a = list()
    a.append(['Female', 22, 'USA', 'EE', 101, 24, 'Puns', 'Hip-Hop', 'Comedy'])
    a.append(['Female', 19, 'SPA', 'MATH', 10, 121, 'Lawyer', 'Pop', 'Romance'])
    a.append(['Male', 20, 'BRA', 'MATH', 145, 12, 'Nerd', 'Country', 'Horror'])
    print(find_user_similarity(np.array(i), np.array(a), attr_dict))
