from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from ..config import EnvConfig, PPOConfig
from ..env.rewards import RewardStrategy
from ..env.single_asset_env import SingleAssetTradingEnv


def make_env(
    df: pd.DataFrame,
    env_cfg: EnvConfig,
    reward_strategy: RewardStrategy | None = None,
) -> SingleAssetTradingEnv:
    return Monitor(
        SingleAssetTradingEnv(
            df=df,
            window_size=env_cfg.window_size,
            initial_cash=env_cfg.initial_cash,
            commission_pct=env_cfg.commission_pct,
            slippage_bps=env_cfg.slippage_bps,
            reward_strategy=reward_strategy,
        )
    )


class _EnvFactory:
    """SubprocVecEnv用のpicklable環境ファクトリ。

    Windowsのmultiprocessing（spawn方式）はローカル関数/ラムダをpickle化できないため、
    トップレベルの呼び出し可能クラスとして定義する。
    """

    def __init__(self, df: pd.DataFrame, env_cfg: EnvConfig, reward_strategy: RewardStrategy | None):
        self.df = df
        self.env_cfg = env_cfg
        self.reward_strategy = reward_strategy

    def __call__(self) -> SingleAssetTradingEnv:
        return make_env(self.df, self.env_cfg, reward_strategy=self.reward_strategy)


def make_vec_env(
    df: pd.DataFrame,
    env_cfg: EnvConfig,
    reward_strategy: RewardStrategy | None = None,
    n_envs: int = 1,
) -> SingleAssetTradingEnv | VecEnv:
    """n_envs>1ならSubprocVecEnvでロールアウト収集を並列化する。

    `model.learn(total_timesteps=N)` の N は n_envs に関わらず総経験量（環境ステップ数）の
    意味を保つため、収集される学習データの総量・学習の精度には影響しない。
    並列化によって短縮されるのは壁時計時間のみ。
    """
    if n_envs <= 1:
        return make_env(df, env_cfg, reward_strategy=reward_strategy)

    env_fns = [_EnvFactory(df, env_cfg, reward_strategy) for _ in range(n_envs)]
    return SubprocVecEnv(env_fns)


def train_one_iteration(
    train_df: pd.DataFrame,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    model_path_in: Path | None,
    tensorboard_log: Path | None = None,
    reward_strategy: RewardStrategy | None = None,
) -> PPO:
    """train_df上でPPOを学習する。既存チェックポイントがあれば継続学習する。

    継続学習することで growth_loop の各イテレーションが
    「前回までの学習成果の上に積む」形になる。
    """
    env = make_env(train_df, env_cfg, reward_strategy=reward_strategy)

    if model_path_in is not None and model_path_in.exists():
        model = PPO.load(str(model_path_in), env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            gamma=ppo_cfg.gamma,
            verbose=0,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        )

    model.learn(total_timesteps=ppo_cfg.timesteps_per_iteration)
    return model
