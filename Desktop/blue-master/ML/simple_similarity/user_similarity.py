import numpy as np

def find_user_similarity(input_user, all_users):
    """
        :param input_user: Represents input user data.
        :param all_users: Represents all user data.
        :type input_user: (1 x n) nparray.
        :type all_user: (m x n) nparray.
        :return: List of integers represent similarity (hamming distance) between input user
        :rtype: (1 x m) nparray.
    """
    if input_user.shape[0] != 1:
        raise ValueError("input_user has to be (1 x n) nparray.")
    if input_user.shape[1] != all_users.shape[1]:
        raise ValueError("input_user's shape (1 x " + str(input_user.shape[1]) + "), all_user's shape (" + str(all_users.shape[0]) + " x " + str(all_users.shape[1]) + ")")

    result = list()
    for i in all_users:
        hamming_distance = 0
        for j in range(len(input_user[0])):
            if input_user[0][j] == i[j]:
                hamming_distance += 1
        result.append(hamming_distance)
    return (np.array(result))

if __name__ == "__main__":
    i = list()
    i.append(['USA', 18, 'CS'])
    i.append(['SPA', 22, 'EE'])
    a = list()
    a.append(['JAP', 20])
    #print(find_user_similarity(np.array(i), np.array(a)))

    i = list()
    i.append(['USA', 18, 'CS'])
    #print(find_user_similarity(np.array(i), np.array(a)))

    a = list()
    a.append(['JAP', 20, 'EE'])
    a.append(['BRA', 18, 'CS'])
    a.append(['USA', 21, 'EE'])
    print(find_user_similarity(np.array(i), np.array(a)))
