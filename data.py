import tensorflow as tf
from transformers import AutoTokenizer
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer

class Dataset:
    def __init__(self, model_path, d_model):
        self.tokenizer = SentenceTransformer(model_path)
        self.d_model = d_model


    def preprocess_dataset(self, df):
        def clean_text(text):
            text = str(text)
            text = text.lower().strip()

            return text


        tqdm.pandas(desc="Clean text")
        df["source"] = df["source"].progress_apply(clean_text)
        
        df['tokens'] = 0.
        embeddings = self.tokenizer.encode(list(df.source), batch_size=16)
        df['tokens'] = [e for e in embeddings]

        df = df.drop(['source', 'rank', 'ancestor_id'], axis=1)

        return df


    def get_notebook_token(self, df, max_cells, cell_pad):
        def create_tensor(col, desired_shape, dtype="int32"):
            out = np.full(shape=desired_shape, fill_value=cell_pad, dtype=dtype)
            
            count = 0
            for _, group in df.groupby("id"):
                value = group[col].tolist()
                value_shape = np.array(value).shape
                
                if len(value_shape) == 1:
                    out[count, :value_shape[0]] = value
                else:
                    out[count, :value_shape[0], :value_shape[1]] = value

                count += 1

            return out
        

        num_train = df.id.nunique()

        # input_ids
        tokens = create_tensor(
            "tokens", 
            (num_train, max_cells, self.d_model),
            dtype="float32"
        )
        
        # target
        target = create_tensor(
            "pct_rank", 
            (num_train, max_cells), 
            dtype="float32"
        )

        return tokens, target


    def build_dataset(self, batch_size, df=None, cell_pad=0., max_cells=128):
        def map_func(tokens, target):
            return ( 
                {
                    'tokens': tokens
                }, 
                target 
            )


        df = self.preprocess_dataset(df)
        
        tokens, target = self.get_notebook_token(df, max_cells, cell_pad)

        dataset = tf.data.Dataset.from_tensor_slices((
            tokens, 
            target
        ))
        dataset = dataset.map(map_func)
        dataset = dataset.batch(batch_size)

        return dataset
        