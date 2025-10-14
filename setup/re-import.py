
import importlib
import src.ask
importlib.reload(src.ask)
from src.ask import QuestionAnswerer

qa = QuestionAnswerer(top_k=3, similarity_threshold=0.45)
