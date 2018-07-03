
import pandas as pd
import requests
import json
import pickle
import numpy as np

app = Flash(__name__)
model_path = os.path.join(os.path.pardir, os.path.pardir, 'models')
model_filepath = os.path.join(model_path, "lr_model.pkl")
scalar_filepath = os.path.join(model_path, "lr_scalar.pkl")

scalar = pickle.load(open(scalar_filepath))
model = pickle.load(open(model_filepath))
