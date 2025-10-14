

!pip install -q trafilatura beautifulsoup4 hnswlib sentence-transformers torch transformers requests tqdm pandas pytest robotexclusionrulesparser accelerate

# Import libraries
import os
import time
import json
import requests
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Dict, Set, Tuple
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import trafilatura
from robotexclusionrulesparser import RobotExclusionRulesParser
from tqdm import tqdm
import pandas as pd

# Create project structure
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("✅ Setup complete!")
