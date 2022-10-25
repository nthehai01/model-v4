import tensorflow as tf
from transformers import AutoTokenizer
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer

class Dataset:
    def __init__(self, model_path, d_model, max_len=128):
        self.tokenizer = SentenceTransformer(model_path)
        self.d_model = d_model
        self.max_len = max_len


    def preprocess_dataset(self, df):
        def truncate_text(text, max_len):
            words = text.split()
            if len(words) > max_len:
                half_len = max_len // 2
                words = words[:half_len] + words[-half_len:]

            return " ".join(words)

        def clean_text(text):
            text = str(text)
            text = text.lower().strip()

            return text


        df["rank"] = df["rank"] * 1.0

        # is code
        df['is_code'] = (df['cell_type'] == 'code') * 1.0

        # is markdown
        df['is_md'] = (df['cell_type'] == 'markdown') * 1.0

        # code rank
        df["code_rank"] = 0.
        code = df[df.cell_type == "code"]
        code_rank = code.groupby("id").cumcount()
        df.loc[code.index, "code_rank"] = code_rank
        del code, code_rank

        # markdown rank
        df["md_rank"] = 0.
        md = df[df.cell_type == "markdown"]
        md_rank = md.groupby("id")["pct_rank"].transform(lambda arr: arr.argsort().argsort())
        df.loc[md.index, "md_rank"] = md_rank
        del md, md_rank

        # {cell_type}_rank
        df["cell_type_rank"] = df["code_rank"] + df["md_rank"]
        
        # target 
        df["target"] = df[["rank", "cell_type_rank"]].values.tolist()

        # detect heading 1
        df["has_heading_1"] = 0.
        md = df[df.cell_type == "markdown"]
        heading_1 = "# "
        tqdm.pandas(desc="Detect heading 1")
        has_heading_1 =  md["source"].progress_apply(lambda x: x[:len(heading_1)] == heading_1) * 1.0
        df.loc[md.index, "has_heading_1"] = has_heading_1
        del md

        # detect heading 2
        df["has_heading_2"] = 0.
        md = df[df.cell_type == "markdown"]
        heading_2 = "## "
        tqdm.pandas(desc="Detect heading 2")
        has_heading_2 =  md["source"].progress_apply(lambda x: x[:len(heading_2)] == heading_2) * 1.0
        df.loc[md.index, "has_heading_2"] = has_heading_2
        del md

        # detect heading 3
        df["has_heading_3"] = 0.
        md = df[df.cell_type == "markdown"]
        heading_3 = "### "
        tqdm.pandas(desc="Detect heading 3")
        has_heading_3 =  md["source"].progress_apply(lambda x: x[:len(heading_3)] == heading_3) * 1.0
        df.loc[md.index, "has_heading_3"] = has_heading_3
        del md

        # truncate text
        tqdm.pandas(desc="Truncate text")
        df["source"] = df["source"].progress_apply(lambda x: truncate_text(x, self.max_len))

        # clean text
        tqdm.pandas(desc="Clean text")
        df["source"] = df["source"].progress_apply(clean_text)
        
        # get tokens
        df['tokens'] = 0.
        embeddings = self.tokenizer.encode(list(df.source), batch_size=16)
        df['tokens'] = [e for e in embeddings]
        del embeddings

        # counting
        df["count_by_type"] = df.groupby(["id", "cell_type"])["cell_id"].transform("count") * 1.0
        df["cell_count"] = df.groupby(["id"])["cell_id"].transform("count") * 1.0

        # additional features
        df["additional_features"] = df[["is_code", "is_md", "code_rank", "has_heading_1", "has_heading_2", "has_heading_3"]].values.tolist()
        
        df = df[["id", "tokens", "additional_features", "count_by_type", "cell_count", "target"]]

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

        # additional_features
        additional_features_len = len(df.additional_features.iloc[0])
        additional_features = create_tensor(
            "additional_features",
            (num_train, max_cells, additional_features_len),
        )

        # count_by_type
        count_by_type = create_tensor(
            "count_by_type",
            (num_train, max_cells),
        )
        count_by_type = np.expand_dims(count_by_type, axis=-1)

        # cell_count
        cell_count = create_tensor(
            "cell_count",
            (num_train, max_cells),
        )
        cell_count = np.expand_dims(cell_count, axis=-1)
        
        # target
        target_len = len(df.target.iloc[0])
        target = create_tensor(
            "target", 
            (num_train, max_cells, target_len), 
            dtype="float32"
        )

        return tokens, additional_features, count_by_type, cell_count, target


    def build_dataset(self, batch_size, df=None, cell_pad=0., max_cells=128):
        def map_func(tokens, additional_features, count_by_type, cell_count, target):
            return ( 
                {
                    'tokens': tokens,
                    'additional_features': additional_features,
                    'count_by_type': count_by_type,
                    'cell_count': cell_count,
                }, 
                target 
            )


        df = self.preprocess_dataset(df)
        
        tokens, additional_features, count_by_type, cell_count, target = self.get_notebook_token(df, max_cells, cell_pad)

        dataset = tf.data.Dataset.from_tensor_slices((
            tokens, additional_features, count_by_type, cell_count, 
            target
        ))
        del tokens, additional_features, count_by_type, cell_count, target
        
        dataset = dataset.map(map_func)
        dataset = dataset.batch(batch_size)

        return dataset
