import os
from pathlib import Path

list_of_files=[
    "src/__init__.py",
    "src/data_ingestion.py",
    "src/embedding.py",
    "src/model_api.py",
    "Experiments/experiments.ipynb",
    "StreamlitApp.py",
    "logger.py",
    "exception.py",
    "setup.py"
]


for filepath in list_of_files:
   filepath = Path(filepath)
   filedir, filename = os.path.split(filepath)

   if filedir !="":
      os.makedirs(filedir, exist_ok=True)

   if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
      with open(filepath, 'w') as f:
         pass