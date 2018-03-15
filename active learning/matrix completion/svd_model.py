from model import SVD_Model
import numpy as np



if __name__ == '__main__':
  print("Testing...")
  """ because it's hard to hit jokes present in the db, this just generates 50 random entries
      hit rate is about 10%, so this is leave 5 out
      pretty hacky, but just for testing before real dataset
  """
  model = SVD_Model()

  joke_ids = (350*np.random.randn(500) + 350).astype(int)
  mse, max_se = model.test(leave_one_out_user_id=5, leave_one_out_joke_ids=joke_ids)
  print("Mean Squared Error:", mse)
  print("Max Squared Error:", max_se)

