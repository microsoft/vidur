import fasteners
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
        self._model_name = config.replica_model_name
        self._num_q_heads = config.replica_num_q_heads
        self._num_kv_heads = config.replica_num_kv_heads
        self._embedding_dim = config.replica_embedding_dim
        self._mlp_hidden_dim = config.replica_mlp_hidden_dim
        self._use_gated_mlp = config.replica_use_gated_mlp
        self._vocab_size = config.replica_vocab_size
        self._block_size = config.replica_block_size

        self._model_provider = config.execution_time_predictor_provider

        self._attention_decode_overhead_percentage = (
            config.sklearn_execution_time_predictor_attention_decode_overhead_percentage
        )
        self._nccl_cpu_launch_overhead_ms = (
            config.sklearn_execution_time_predictor_nccl_cpu_launch_overhead_ms
        )
        self._nccl_cpu_skew_overhead_per_device_ms = (
            config.sklearn_execution_time_predictor_nccl_cpu_skew_overhead_per_device_ms
        )

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
        self._cpu_overhead_input_file = (
            config.sklearn_execution_time_predictor_cpu_overhead_input_file
        )

        self._kv_cache_prediction_granularity = (
            config.sklearn_execution_time_predictor_kv_cache_prediction_granularity
        )
        self._prediction_max_prefill_chunk_size = (
            config.sklearn_execution_time_predictor_prediction_max_prefill_chunk_size
        )

        self._device_memory = (
            config.replica_total_memory_gb
        )

        self._models = self._train_models()
        self._predictions = self._predict_from_models()

    def _load_compute_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)

        logger.info(f"Length of complete compute df: {len(df)} {file_path}")
        logger.info(f"self._num_q_heads: {self._num_q_heads}")
        logger.info(f"self._embedding_dim: {self._embedding_dim}")
        logger.info(f"self._mlp_hidden_dim: {self._mlp_hidden_dim}")
        logger.info(f"self._use_gated_mlp: {self._use_gated_mlp}")
        logger.info(f"self._vocab_size: {self._vocab_size}")
        logger.info(f"self._num_tensor_parallel_workers: {self._num_tensor_parallel_workers}")
            
        return df[
            (df["n_head"] == self._num_q_heads)
            & (df["n_kv_head"] == self._num_kv_heads)
            & (df["n_embd"] == self._embedding_dim)
            & (df["n_expanded_embd"] == self._mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._use_gated_mlp)
            & (df["vocab_size"] == self._vocab_size)
            & (df["num_tensor_parallel_workers"] == self._num_tensor_parallel_workers)
        ]

    def _load_attention_df(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.fillna(0)
        df = df.drop_duplicates()

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

    def _load_cpu_overhead_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        print(len(df))
        print(self._model_name)
        print(self._num_tensor_parallel_workers)
        filtered_df = df[
            (df["model_name"] == self._model_name)
            & (df["tensor_parallel_degree"] == self._num_tensor_parallel_workers)
        ]
        return filtered_df

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        # df = df.dropna()
        df = df.drop_duplicates()
        return df

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = df_with_derived_features[["prefill_chunk_size", "batch_size"]].max(axis=1)
        df_with_derived_features["is_decode"] = df_with_derived_features["prefill_chunk_size"] == 0
        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_cpu_overhead_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Handling the case where y_true is 0 separately to avoid division by zero
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        # For non-zero true values, calculate the absolute percentage error
        error = np.zeros_like(y_true, dtype=float)  # using float instead of np.float
        error[non_zero_true_mask] = np.abs((y_true[non_zero_true_mask] - y_pred[non_zero_true_mask]) / y_true[non_zero_true_mask]) * 100
        
        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        # Return the mean of the absolute percentage errors
        return np.mean(error)

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
        with fasteners.InterProcessReaderWriterLock(f"{self._cache_dir}/{model_name}_model_lock.file").read_lock():
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
        with fasteners.InterProcessReaderWriterLock(f"{self._cache_dir}/{model_name}_model_lock.file").write_lock():
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

        if len(df) < self._k_fold_cv_splits:
            cv = 2
        else:
            cv = self._k_fold_cv_splits

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
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
        with fasteners.InterProcessReaderWriterLock(f"{self._cache_dir}/{model_name}_prediction_lock.file").write_lock():
            model_name_hash = self._get_model_name_hash(model_name)
            cache_file = f"{self._cache_dir}/{model_name_hash}_predictions.pkl"
            json_file = f"{self._cache_dir}/{model_name}_{model_name_hash}_predictions.json"
            pickle.dump(predictions, open(cache_file, "wb"))
            # convert keys from tuple to string
            json_serializable_predictions = {str(x): y for x, y in predictions.items()}
            json.dump(json_serializable_predictions, open(json_file, "w"))

    def _load_model_predication_cache(self, model_name: str) -> Dict[Tuple, float]:
        with fasteners.InterProcessReaderWriterLock(f"{self._cache_dir}/{model_name}_prediction_lock.file").read_lock():
            if self._no_cache:
                return

            model_name_hash = self._get_model_name_hash(model_name)
            cache_file = f"{self._cache_dir}/{model_name_hash}_predictions.pkl"

            if not os.path.exists(cache_file):
                return
            
            logger.info(f"Found model {model_name} predictions in cache")

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

    def _train_num_token_based_models(self) -> Dict[str, BaseEstimator]:
        compute_df = self._load_compute_df(self._compute_input_file)
        compute_df = self._get_compute_df_with_derived_features(compute_df)

        models = {}
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "rms_norm",
            "add_norm",
        ]

        for model_name in model_names:
            logger.info(f"Training model {model_name}, size of training data: {len(compute_df)}")
            models[model_name] = self._train_model(
                model_name=model_name,
                df=compute_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)

        model_names = [
            "attn_rope",
            "attn_kv_cache_save",
        ]

        for model_name in model_names:
            models[model_name] = self._train_model(
                model_name=model_name,
                df=attention_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
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

    def _train_batch_sized_based_models(self) -> Dict[str, BaseEstimator]:
        models = {}
        model_names = [
            "SCHEDULE",
            "SAMPLER_E2E",
            "PREPARE_INPUTS_E2E",
            "PROCESS_MODEL_OUTPUTS",
            "POST_PREPARE_INPUTS_BARRIER",
            "RAY_COMM_TIME"
        ]

        cpu_overhead_df = self._load_cpu_overhead_df(self._cpu_overhead_input_file)
        cpu_overhead_df = self._get_cpu_overhead_df_with_derived_features(cpu_overhead_df)

        print(len(cpu_overhead_df))

        for model_name in model_names:
            models[model_name] = self._train_model(
                model_name=model_name,
                df=cpu_overhead_df,
                feature_cols=["batch_size"],
                target_col=f"{model_name}_MEDIAN",
            )

        return models

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]

        models = {}

        models["attn_prefill_output_reshape_copy"] = self._train_model(
            model_name="attn_prefill_output_reshape_copy",
            df=prefill_df,
            feature_cols=["num_tokens"],
            target_col="time_stats.attn_prefill_output_reshape_copy.median",
        )

        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"] + chunked_prefill_df["prefill_chunk_size"]
        )

        models["attn_prefill_kv_cache_prep"] = self._train_model(
            model_name="attn_prefill_kv_cache_prep",
            df=chunked_prefill_df,
            feature_cols=["total_prefill_tokens"],
            target_col="time_stats.attn_prefill_kv_cache_prep.median",
        )
    
        models["attn_prefill"] = self._train_model(
            model_name="attn_prefill",
            df=prefill_df,
            feature_cols=["kv_cache_size", "prefill_chunk_size"],
            target_col="time_stats.attn_prefill.median",
        )

        models["attn_decode"] = self._train_model(
            model_name="attn_decode",
            df=decode_df,
            feature_cols=["batch_size", "kv_cache_size"],
            target_col="time_stats.attn_decode.median",
        )

        return models

    def _train_models(self) -> Dict[str, BaseEstimator]:
        models = self._train_num_token_based_models()
        models.update(self._train_batch_sized_based_models())
        models.update(self._train_attention_layer_models())

        return models

    def _predict_for_num_token_based_models(self) -> Dict[str, Any]:
        predictions = {}

        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "rms_norm",
            "add_norm",
        ]

        model_names.append("send_recv")

        if self._num_tensor_parallel_workers > 1:
            model_names.append("all_reduce")

        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        for model_name in model_names:
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        return predictions

    def _predict_for_batch_size_based_models(self) -> Dict[str, Any]:
        predictions = {}

        model_names = [
            "SCHEDULE",
            "SAMPLER_E2E",
            "PREPARE_INPUTS_E2E",
            "PROCESS_MODEL_OUTPUTS",
            "POST_PREPARE_INPUTS_BARRIER",
            "RAY_COMM_TIME"
        ]

        batch_size_range = np.arange(1, self._max_batch_size + 1)
        X = pd.DataFrame({"batch_size": batch_size_range})

        for model_name in model_names:
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        return predictions

    def _predict_for_attention_layer_models(self) -> Dict[str, Any]:
        predictions = {}

        decode_batch_size_range = np.arange(1, self._max_batch_size + 1)
        decode_kv_cache_size_range = np.arange(
            0, self._max_tokens_per_request + 1, self._kv_cache_prediction_granularity
        )
        decode_prefill_chunk_size_range = [0]
        decode_batch_size, decode_kv_cache_size, decode_prefill_chunk_size = \
            zip(
                *product(
                    decode_batch_size_range,
                    decode_kv_cache_size_range,
                    decode_prefill_chunk_size_range
                )
            )

        prefill_batch_size_range = [1]
        prefill_kv_cache_size_range = np.arange(
            0, self._max_tokens_per_request + 1, self._kv_cache_prediction_granularity
        )
        prefill_prefill_chunk_size_range = np.arange(
            1, self._prediction_max_prefill_chunk_size + 1
        )
        prefill_batch_size, prefill_kv_cache_size, prefill_prefill_chunk_size = \
            zip(
                *product(
                    prefill_batch_size_range,
                    prefill_kv_cache_size_range,
                    prefill_prefill_chunk_size_range
                )
            )

        attention_df = pd.DataFrame(
            {
                "batch_size": decode_batch_size + prefill_batch_size,
                "kv_cache_size": decode_kv_cache_size + prefill_kv_cache_size,
                "prefill_chunk_size": decode_prefill_chunk_size + prefill_prefill_chunk_size,
            }
        )

        attention_df["is_decode"] = attention_df["prefill_chunk_size"] == 0
        attention_df["num_tokens"] = attention_df[["prefill_chunk_size", "batch_size"]].max(axis=1)    

        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]
        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"] + chunked_prefill_df["prefill_chunk_size"]
        )
    
        predictions["attn_prefill_output_reshape_copy"] = self._get_model_prediction(
            "attn_prefill_output_reshape_copy", 
            self._models["attn_prefill_output_reshape_copy"],
            prefill_df[["num_tokens"]]
        )

        predictions["attn_prefill_kv_cache_prep"] = self._get_model_prediction(
            "attn_prefill_kv_cache_prep", 
            self._models["attn_prefill_kv_cache_prep"],
            chunked_prefill_df[["total_prefill_tokens"]]
        )

        predictions["attn_prefill"] = self._get_model_prediction(
            "attn_prefill", 
            self._models["attn_prefill"],
            prefill_df[["kv_cache_size", "prefill_chunk_size"]]
        )

        predictions["attn_decode"] = self._get_model_prediction(
            "attn_decode", 
            self._models["attn_decode"],
            decode_df[["batch_size", "kv_cache_size"]]
        )

        return predictions

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = self._predict_for_num_token_based_models()
        predictions.update(self._predict_for_batch_size_based_models())
        predictions.update(self._predict_for_attention_layer_models())

        return predictions

    def _get_num_tokens(self, batch: Batch) -> float:
        if hasattr(batch, "__num_tokens"):
            return batch.__num_tokens

        num_tokens = sum(batch.num_tokens)
        # round up to the nearest multiple of 8
        num_tokens = int(np.ceil(num_tokens / 8) * 8)

        batch.__num_tokens = num_tokens

        return num_tokens

    def _get_batch_decode_attention_params(self, batch: Batch) -> Tuple[int, int]:
        if hasattr(batch, "_decode_params"):
            return batch._decode_params

        decode_kv_cache_sizes = []

        for request in batch.requests:
            if request._is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)

        if not decode_kv_cache_sizes:
            return (0, 0)

        decode_batch_size = len(decode_kv_cache_sizes)
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
        decode_avg_kv_cache_size = (
            decode_avg_kv_cache_size // self._kv_cache_prediction_granularity
        ) * self._kv_cache_prediction_granularity

        return (decode_batch_size, decode_avg_kv_cache_size)

    def _get_batch_prefill_attention_params(self, batch: Batch) -> List[Tuple[int, int]]:
        if hasattr(batch, "_prefill_params"):
            return batch._prefill_params

        prefill_params = []

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
            if request._is_prefill_complete:
                continue

            prefill_chunk_size = num_tokens_to_process
            kv_cache_size = (
                request.num_processed_tokens
                // self._kv_cache_prediction_granularity
            ) * self._kv_cache_prediction_granularity

            prefill_params.append((kv_cache_size, prefill_chunk_size))

        batch._prefill_params = prefill_params

        return prefill_params

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["attn_pre_proj"][(num_tokens,)] * 1.05

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["attn_post_proj"][(num_tokens,)]

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["mlp_up_proj"][(num_tokens,)]

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["mlp_down_proj"][(num_tokens,)]

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["mlp_act"][(num_tokens,)]

    def _get_rms_norm_layer_act_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["rms_norm"][(num_tokens,)]

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["add_norm"][(num_tokens,)]

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return (
            self._predictions["all_reduce"][(num_tokens,)]
            + self._nccl_cpu_launch_overhead_ms
            + self._nccl_cpu_skew_overhead_per_device_ms  * self._num_tensor_parallel_workers ** 1.25
        )

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        num_tokens = self._get_num_tokens(batch)
        return self._predictions["send_recv"][(num_tokens,)]

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        num_tokens = sum(batch.num_tokens)
        return self._predictions["attn_rope"][(num_tokens,)]

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        num_tokens = sum(batch.num_tokens)

        # special case A40 llama 7b TP1 for now
        if self._device_memory == 45 and self._num_layers == 32 and self._num_kv_heads == 32 and self._num_tensor_parallel_workers == 1:
            return self._predictions["attn_kv_cache_save"][(num_tokens,)] * 0.3

        return self._predictions["attn_kv_cache_save"][(num_tokens,)]

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        decode_batch_size, decode_avg_kv_cache_size = self._get_batch_decode_attention_params(
            batch
        )
        if decode_batch_size == 0:
            return 0

        return (
            self._predictions["attn_decode"][(decode_batch_size, decode_avg_kv_cache_size)]
            * (1 + self._attention_decode_overhead_percentage)
        )

    def _get_attention_prefill_kv_cache_prep_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)

        total_time = 0

        for kv_cache_size, prefill_chunk_size in prefill_params:
            if kv_cache_size == 0:
                continue

            total_time += self._predictions["attn_prefill_kv_cache_prep"][
                (kv_cache_size + prefill_chunk_size,)
            ]            

        return total_time

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)

        total_time = 0

        for kv_cache_size, prefill_chunk_size in prefill_params:
            total_time += self._predictions["attn_prefill"][
                (kv_cache_size, prefill_chunk_size)
            ]

        return total_time

    def _get_attention_prefill_output_reshape_copy_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)
        total_current_prefill_tokens = sum([x[1] for x in prefill_params])

        if total_current_prefill_tokens == 0:
            return 0

        if total_current_prefill_tokens > self._prediction_max_prefill_chunk_size:
            return self._predictions["attn_prefill_output_reshape_copy"][(
                self._prediction_max_prefill_chunk_size,
            )] * (total_current_prefill_tokens / self._prediction_max_prefill_chunk_size)

        return self._predictions["attn_prefill_output_reshape_copy"][(
            total_current_prefill_tokens,
        )]

    def _get_schedule_time(self, batch: Batch) -> float:
        return self._predictions["SCHEDULE"][(batch.size,)]

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        return self._predictions["SAMPLER_E2E"][(batch.size,)]

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        return self._predictions["PREPARE_INPUTS_E2E"][(batch.size,)]

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        return self._predictions["PROCESS_MODEL_OUTPUTS"][(batch.size,)]

    def _get_post_prepare_inputs_barrier_time(self, batch: Batch) -> float:
        return self._predictions["POST_PREPARE_INPUTS_BARRIER"][(batch.size,)]

    def _get_ray_comm_time(self, batch: Batch) -> float:
        return self._predictions["RAY_COMM_TIME"][(batch.size,)]

    def to_dict(self) -> dict:
        return {
            "model_provider": self._model_provider,
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
            "cpu_overhead_input_file": self._cpu_overhead_input_file,
            "prediction_max_prefill_chunk_size": self._prediction_max_prefill_chunk_size,
            "max_batch_size": self._max_batch_size,
        }
