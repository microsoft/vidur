import hashlib
import json
import logging
import os
import pickle
from abc import abstractmethod
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from simulator.config import Config
from simulator.entities import Batch
from simulator.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)

logger = logging.getLogger(__name__)


ALL_REDUCE_INPUT_FEATURES = [
    "prefill_batch_size",
    "decode_batch_size",
    "prefill_context_length",
]
SEND_RECV_INPUT_FEATURES = ["batch_size", "context_length"]


class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._cache_dir = f"{config.cache_dir}/execution_time_predictor"
        os.makedirs(self._cache_dir, exist_ok=True)

        self._no_cache = config.sklearn_execution_time_predictor_no_cache

        self._k_fold_cv_splits = (
            config.sklearn_execution_time_predictor_k_fold_cv_splits
        )
        self._num_q_heads = config.replica_num_q_heads
        self._num_kv_heads = config.replica_num_kv_heads
        self._embedding_dim = config.replica_embedding_dim
        self._mlp_hidden_dim = config.replica_mlp_hidden_dim
        self._use_gated_mlp = config.replica_use_gated_mlp
        self._vocab_size = config.replica_vocab_size
        self._block_size = config.replica_block_size

        self._max_batch_size = (
            config.sklearn_execution_time_predictor_prediction_max_batch_size
        )
        self._max_tokens_per_request = (
            config.sklearn_execution_time_predictor_prediction_max_tokens_per_request
        )
        self._max_tokens = self._max_tokens_per_request * self._max_batch_size
        self._prefill_chunk_size = config.replica_prefill_chunk_size

        self._compute_input_file = (
            config.sklearn_execution_time_predictor_compute_input_file
        )
        self._attention_input_file = (
            config.sklearn_execution_time_predictor_attention_input_file
        )
        self._all_reduce_input_file = (
            config.sklearn_execution_time_predictor_all_reduce_input_file
        )
        self._send_recv_input_file = (
            config.sklearn_execution_time_predictor_send_recv_input_file
        )

        self._kv_cache_prediction_granularity = (
            config.sklearn_execution_time_predictor_kv_cache_prediction_granularity
        )
        self._prediction_max_prefill_chunk_size = (
            config.sklearn_execution_time_predictor_prediction_max_prefill_chunk_size
        )

        self._models = self._train_models()
        self._predictions = self._predict_from_models()

    def _load_compute_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        return df[
            (df["n_head"] == self._num_q_heads)
            & (df["n_embd"] == self._embedding_dim)
            & (df["n_expanded_embd"] == self._mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._use_gated_mlp)
            & (df["vocab_size"] == self._vocab_size)
            & (df["num_tensor_parallel_workers"] == self._num_tensor_parallel_workers)
        ]

    def _load_attention_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        return df[
            (df["n_embd"] == self._embedding_dim)
            & (df["n_q_head"] == self._num_q_heads)
            & (df["n_kv_head"] == self._num_kv_heads)
            & (df["block_size"] == self._block_size)
            & (df["num_tensor_parallel_workers"] == self._num_tensor_parallel_workers)
        ]

    def _load_all_reduce_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        return df[
            (df["n_embd"] == self._embedding_dim)
            & (df["num_workers"] == self._num_tensor_parallel_workers)
        ]

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        filtered_df = df[
            (df["n_embd"] == self._embedding_dim)
            & (df["num_tensor_parallel_workers"] == self._num_tensor_parallel_workers)
        ]
        return filtered_df

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.dropna()
        df = df.drop_duplicates()
        return df

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()

        df_with_derived_features["decode"] = df["prefill_chunk_size"] == 1

        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def _get_scorer(self) -> Any:
        return make_scorer(
            SklearnExecutionTimePredictor.mean_absolute_percentage_error,
            greater_is_better=False,
        )

    @abstractmethod
    def _get_grid_search_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_estimator(self) -> BaseEstimator:
        pass

    def _get_model_name_hash(self, model_name: str) -> str:
        config_str = str(self.to_dict())
        model_name = f"{config_str}_{model_name}"
        model_name = model_name.encode("utf-8")
        return hashlib.md5(model_name).hexdigest()

    def _load_model_from_cache(self, model_name: str) -> BaseEstimator:
        if self._no_cache:
            return

        # use md5 hash of model name as cache key
        model_name_hash = self._get_model_name_hash(model_name)

        # check if model is in cache
        cache_file = f"{self._cache_dir}/{model_name_hash}.pkl"
        if not os.path.exists(cache_file):
            return

        logger.info(f"Found model {model_name} in cache")
        model = pickle.load(open(cache_file, "rb"))
        return model

    def _store_model_in_cache(self, model_name: str, model: BaseEstimator) -> None:
        model_name_hash = self._get_model_name_hash(model_name)

        # store model in cache
        cache_file = f"{self._cache_dir}/{model_name_hash}.pkl"
        pickle.dump(model, open(cache_file, "wb"))

    def _store_training_prediction_data(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: BaseEstimator,
    ) -> None:
        df = df.copy()

        # convert the df to list of tuples
        df["prediction"] = model.predict(df[feature_cols])

        # store the prediction data
        df[feature_cols + [target_col, "prediction"]].to_csv(
            f"{self._cache_dir}/{model_name}_training_predictions.csv",
            index=False,
        )

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        cached_model = self._load_model_from_cache(model_name)
        if cached_model:
            return cached_model

        model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=self._k_fold_cv_splits,
            n_jobs=-1,
        )

        # we don't create a train/test split, because we want to use all data for training
        # and we don't care about overfitting, because we only want to predict execution time within the same domain
        X, y = df[feature_cols], df[target_col]

        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(
            f"Trained model {model_name} and found best parameters: {grid_search.best_params_} "
            f"with mean absolute percentage error (MEAP) {-score}%"
        )

        self._store_model_in_cache(model_name, grid_search.best_estimator_)

        self._store_training_prediction_data(
            model_name=model_name,
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            model=grid_search.best_estimator_,
        )

        return grid_search.best_estimator_

    def _store_model_predication_cache(
        self, model_name: str, predictions: Dict[Tuple, float]
    ) -> None:
        model_name_hash = self._get_model_name_hash(model_name)
        cache_file = f"{self._cache_dir}/{model_name_hash}_predictions.pkl"
        json_file = f"{self._cache_dir}/{model_name}_{model_name_hash}_predictions.json"
        pickle.dump(predictions, open(cache_file, "wb"))
        # convert keys from tuple to string
        json_serializable_predictions = {str(x): y for x, y in predictions.items()}
        json.dump(json_serializable_predictions, open(json_file, "w"))

    def _load_model_predication_cache(self, model_name: str) -> Dict[Tuple, float]:
        if self._no_cache:
            return

        model_name_hash = self._get_model_name_hash(model_name)
        cache_file = f"{self._cache_dir}/{model_name_hash}_predictions.pkl"

        if not os.path.exists(cache_file):
            return

        predictions = pickle.load(open(cache_file, "rb"))
        return predictions

    def _get_model_prediction(
        self, model_name: str, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[Tuple, float]:
        X = X.copy()

        cached_predictions = self._load_model_predication_cache(model_name)
        if cached_predictions:
            return cached_predictions

        logger.info(f"Predicting execution time for model {model_name}")

        predictions_array = model.predict(X)

        # turn this into a dict, so we can store use it as a cache
        # the key is tuple for each row of X
        predictions = dict(zip([tuple(x) for x in X.values], predictions_array))

        self._store_model_predication_cache(model_name, predictions)

        X["prediction"] = predictions_array
        X.to_csv(
            f"{self._cache_dir}/{model_name}_predictions.csv",
            index=False,
        )

        return predictions

    def _train_models(self) -> Dict[str, BaseEstimator]:
        models = {}

        compute_df = self._load_compute_df(self._compute_input_file)
        compute_df = self._get_compute_df_with_derived_features(compute_df)

        models["attention_pre_proj"] = self._train_model(
            model_name="attention_pre_proj",
            df=compute_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.attn_pre_proj.median",
        )
        models["attention_post_proj"] = self._train_model(
            model_name="attention_post_proj",
            df=compute_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.attn_post_proj.median",
        )
        models["mlp"] = self._train_model(
            model_name="mlp",
            df=compute_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.mlp.median",
        )

        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)

        models["attention_attn_inner_decode"] = self._train_model(
            model_name="attention_attn_inner_decode",
            df=attention_df[attention_df["decode"] == True],
            feature_cols=["batch_size", "kv_cache_size"],
            target_col="time_stats.attention.median",
        )
        models["attention_attn_inner_prefill"] = self._train_model(
            model_name="attention_attn_inner_prefill",
            df=attention_df[attention_df["decode"] == False],
            feature_cols=["kv_cache_size", "prefill_chunk_size"],
            target_col="time_stats.attention.median",
        )

        send_recv_df = self._load_send_recv_df(self._send_recv_input_file)
        send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df)

        models["send_recv"] = self._train_model(
            model_name="send_recv",
            df=send_recv_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.send_recv.median",
        )

        if self._num_tensor_parallel_workers > 1:
            all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file)
            all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df)

            models["all_reduce"] = self._train_model(
                model_name="all_reduce",
                df=all_reduce_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.all_reduce.median",
            )

        return models

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = {}

        num_token_based_models = [
            "attention_pre_proj",
            "attention_post_proj",
            "mlp",
        ]

        num_token_based_models.append("send_recv")

        if self._num_tensor_parallel_workers > 1:
            num_token_based_models.append("all_reduce")

        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        for model_name in num_token_based_models:
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        model = self._models["attention_attn_inner_decode"]

        batch_size_range = np.arange(1, self._max_batch_size + 1)
        kv_cache_size_range = np.arange(
            0, self._max_tokens_per_request + 1, self._kv_cache_prediction_granularity
        )

        batch_size, kv_cache_size = zip(*product(batch_size_range, kv_cache_size_range))

        X = pd.DataFrame(
            {
                "batch_size": batch_size,
                "kv_cache_size": kv_cache_size,
            }
        )
        predictions["attention_attn_inner_decode"] = self._get_model_prediction(
            "attention_attn_inner_decode", model, X
        )

        kv_cache_size_range = np.arange(
            0, self._max_tokens_per_request + 1, self._kv_cache_prediction_granularity
        )
        prefill_chunk_size_range = np.arange(
            1, self._prediction_max_prefill_chunk_size + 1
        )

        # take the cartesian product of kv_cache_num_token_range and q_num_token_range
        # and then get seperate lists for kv_cache_num_tokens and q_num_tokens
        kv_cache_size, prefill_chunk_size = zip(
            *product(kv_cache_size_range, prefill_chunk_size_range)
        )

        model = self._models["attention_attn_inner_prefill"]
        X = pd.DataFrame(
            {
                "kv_cache_size": kv_cache_size,
                "prefill_chunk_size": prefill_chunk_size,
            }
        )
        predictions["attention_attn_inner_prefill"] = self._get_model_prediction(
            "attention_attn_inner_prefill", model, X
        )

        return predictions

    def _get_num_tokens(self, batch: Batch) -> float:
        return float(sum(batch.num_tokens))

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["attention_pre_proj"][(num_tokens,)]

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["attention_post_proj"][(num_tokens,)]

    def _get_attention_layer_flash_attention_execution_time(
        self, batch: Batch
    ) -> float:
        total_time = 0

        decode_kv_cache_sizes = []

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
            if request._is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)
            else:
                prefill_chunk_size = num_tokens_to_process
                kv_cache_size = (
                    request.num_processed_tokens
                    // self._kv_cache_prediction_granularity
                ) * self._kv_cache_prediction_granularity
                attention_time = self._predictions["attention_attn_inner_prefill"][
                    (
                        kv_cache_size,
                        prefill_chunk_size,
                    )
                ]
                total_time += attention_time

        if decode_kv_cache_sizes:
            decode_batch_size = len(decode_kv_cache_sizes)
            decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
            decode_avg_kv_cache_size = (
                decode_avg_kv_cache_size // self._kv_cache_prediction_granularity
            ) * self._kv_cache_prediction_granularity
            attention_time = self._predictions["attention_attn_inner_decode"][
                (
                    decode_batch_size,
                    decode_avg_kv_cache_size,
                )
            ]
            total_time += attention_time

        return total_time

    def _get_mlp_layer_mlp_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["mlp"][(num_tokens,)]

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["all_reduce"][(num_tokens,)]

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["send_recv"][(num_tokens,)]

    def to_dict(self) -> dict:
        return {
            "num_tensor_parallel_workers": self._num_tensor_parallel_workers,
            "k_fold_cv_splits": self._k_fold_cv_splits,
            "num_q_heads": self._num_q_heads,
            "num_kv_heads": self._num_kv_heads,
            "embedding_dim": self._embedding_dim,
            "mlp_hidden_dim": self._mlp_hidden_dim,
            "use_gated_mlp": self._use_gated_mlp,
            "vocab_size": self._vocab_size,
            "block_size": self._block_size,
            "max_tokens": self._max_tokens,
            "compute_input_file": self._compute_input_file,
            "all_reduce_input_file": self._all_reduce_input_file,
            "send_recv_input_file": self._send_recv_input_file,
            "prediction_max_prefill_chunk_size": self._prediction_max_prefill_chunk_size,
            "max_batch_size": self._max_batch_size,
        }
