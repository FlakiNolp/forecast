from __future__ import annotations
import abc
import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Any
import backoff
import httpx
import pandas as pd
from openai import AsyncClient
from tqdm.auto import tqdm

from expressions import regex_dict

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


# =========================
# Публичный интерфейс
# =========================

class BaseModel(ABC):
    @abstractmethod
    def fit(self, news_df: pd.DataFrame, candles_df: pd.DataFrame) -> None: ...
    @abstractmethod
    def predict(self) -> pd.DataFrame: ...


@dataclass
class ModelConfig:
    proxy: str
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "API_KEY"
    model_name: str = "openai/gpt-4o-mini"
    timeout_sec: float = 10
    max_retries: int = 5
    stall_timeout_sec: float = 15.0
    max_concurrency: int = 30
    # только один прогресс-бар нужен
    bar_pos_sentiment: int = 0
    bar_pos_tickers: int = 1
    prediction_length: int = 20



def create_model(cfg: ModelConfig) -> BaseModel:
    """
    Единственная публичная фабрика. Возвращает объект с интерфейсом BaseModel.
    Все внутренние классы и “билдеры” скрыты.
    """
    return PipelineBuilder(cfg).build()


# =========================
# Внутренние типы/утилиты
# =========================

class LLMQuotaError(Exception):
    """LLM вернул 402/403 — надо остановить LLM-ветку."""


async def gather_with_progress(coros: Iterable[asyncio.Future], desc: str, position: int = 0):
    """Собираем результаты с прогрессом по мере готовности (as_completed) и сохраняем порядок."""
    async def _wrap(idx: int, c):
        try:
            res = await c
            return idx, res, None
        except Exception as e:
            return idx, None, e

    tasks = [asyncio.create_task(_wrap(i, c)) for i, c in enumerate(coros)]
    results: list[Any] = [None] * len(tasks)

    with tqdm(
        total=len(tasks),
        desc=desc,
        position=position,
        leave=True,
        dynamic_ncols=True,
        miniters=1,
        mininterval=0.1,
    ) as bar:
        for fut in asyncio.as_completed(tasks):
            idx, res, err = await fut
            results[idx] = err if err is not None else res
            bar.update(1)

    return results



# =========================
# Внутренний LLM-сервис (с backoff)
# =========================

@dataclass
class LLMConfig:
    base_url: str
    api_key_env: str
    model_name: str
    proxy: str
    timeout: float
    max_retries: int


class LLMService:
    """Единый LLM-сервис с общим семафором и backoff."""

    def __init__(self, cfg: LLMConfig, semaphore: asyncio.Semaphore):
        self.cfg = cfg
        self.semaphore = semaphore
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.cfg.timeout), proxy=self.cfg.proxy)
        self._client = AsyncClient(
            base_url=self.cfg.base_url,
            api_key=os.getenv(self.cfg.api_key_env, ""),
            http_client=self._http,
        )

    async def aclose(self):
        await self._client.close()
        await self._http.aclose()

    @staticmethod
    def _giveup(exc: Exception) -> bool:
        status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "status_code", None)
        if status in (402, 403):  # квоты/запрет
            return True
        if status == 429:  # оставить на ретраи
            return False
        return status is not None and not (500 <= int(status) <= 599)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=lambda self, *a, **k: self.cfg.max_retries,
        jitter=backoff.full_jitter,
        giveup=lambda e: LLMService._giveup(e),
    )
    async def chat_json(self, system_prompt: str, response_format: dict, text: str) -> dict:
        async with self.semaphore:
            try:
                resp = await self._client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                    response_format=response_format,
                )
            except Exception as e:
                status = getattr(e, "status_code", None)
                if hasattr(e, "response") and getattr(e, "response", None) is not None:
                    try:
                        status = e.response.status_code
                    except Exception:
                        pass
                if status in (402, 403):
                    raise LLMQuotaError(str(e)) from e
                raise

        msg = resp.choices[0].message
        parsed = getattr(msg, "parsed", None)
        return parsed if parsed is not None else json.loads(msg.content)


# =========================
# Сентимент (LLM)
# =========================

class SentimentExtractor(abc.ABC):
    @abc.abstractmethod
    async def add_sentiment(self, news_df: pd.DataFrame, position: int) -> None: ...


class SentimentLLMExtractor(SentimentExtractor):
    SYSTEM_PROMPT = (
        "Ты — экстрактор сентимента, работающий с экономическими новостями. "
        "Верни класс 0 (нейтрально), 1 (положительно) или 2 (отрицательно). "
        "Только структурированный вывод."
    )
    RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_extraction",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"sentiment": {"type": "integer", "enum": [0, 1, 2]}},
                "required": ["sentiment"],
            },
            "strict": True,
        },
    }

    def __init__(self, llm: LLMService):
        self._llm = llm

    async def add_sentiment(self, news_df: pd.DataFrame, position: int) -> None:
        texts = news_df["publication"].tolist()
        coros = (self._llm.chat_json(self.SYSTEM_PROMPT, self.RESPONSE_FORMAT, t) for t in texts)
        results = await gather_with_progress(coros, desc="Сентимент", position=position)
        out: list[int | None] = []
        for r in results:
            out.append(None if isinstance(r, Exception) else r.get("sentiment"))
        news_df["sentiment"] = out


# =========================
# Тикеры: стратегии + оркестратор с as_completed
# =========================

class TickerStrategy(abc.ABC):
    @abc.abstractmethod
    async def extract_one(self, text: str) -> list[str]: ...


class RegexTickerStrategy(TickerStrategy):
    def __init__(self):
        self._cache: dict[str, list[re.Pattern]] = regex_dict

    async def extract_one(self, text: str) -> list[str]:
        text_str = str(text).lower()
        found_tickers = set()
        # 1) Порядок: сначала OZPH (если есть), затем остальные
        keys = (["OZPH"] if "OZPH" in regex_dict else []) + [
            k for k in regex_dict if k != "OZPH"
        ]

        # 2) По каждому тикеру проверяем: есть ли хоть одно совпадение его паттернов
        for ticker in keys:
            patterns = regex_dict[ticker]
            has_match = False
            for pat in patterns:
                if re.search(pat, text_str):
                    has_match = True
                    break
            if has_match and ticker not in found_tickers:
                found_tickers.add(ticker)

        # 3) Если OZPH найден — OZON исключаем
        if "OZPH" in found_tickers:
            found_tickers = [t for t in found_tickers if t != "OZON"]

        return list(found_tickers)


class TickerExtractor:
    def __init__(self, strategy: TickerStrategy):
        self._strategy = strategy

    async def add_tickers(
        self,
        news_df: pd.DataFrame,
        llm_bar_pos: int,
    ) -> None:
        texts = news_df["publication"].tolist()
        coros = [self._strategy.extract_one(text) for text in texts]
        results = await gather_with_progress(coros, desc="Тикеры (Regex)", position=llm_bar_pos)
        news_df["tickers"] = results


# =========================
# Внутренний Pipeline + билдер
# =========================

class SmartModel(BaseModel):
    def __init__(
        self,
        sentiment: SentimentExtractor,
        tickers: TickerExtractor,
        cfg: ModelConfig,
    ):
        self._sent = sentiment
        self._tix = tickers
        self._cfg = cfg
        self._news: pd.DataFrame | None = None
        self._candles: pd.DataFrame | None = None

        self.predictor = TimeSeriesPredictor(
            prediction_length=cfg.prediction_length,
            target="close",
            freq='D',
            eval_metric="WQL"
        )

        self.data = None

    def __resolve_daily_sentiment(self, group: pd.DataFrame) -> str:
        """
        Выбирает итоговый сентимент для группы (один тикер в один день)
        по правилам:
          1) мода;
          2) если ничья — выбираем не 'neutral';
          3) если снова ничья — сентимент самой поздней новости.
        """
        # считаем частоты по исходным (как есть) меткам
        counts = group['sentiment'].value_counts()
        max_count = counts.max()
        top = counts[counts == max_count].index.tolist()

        if len(top) == 1:
            return top[0]

        # ничья: попробуем выбрать не-нейтральную метку
        top_non_neutral = [s for s in top if str(s).strip().lower() != 'neutral']
        if len(top_non_neutral) == 1:
            return top_non_neutral[0]

        # если всё ещё не однозначно — берем сентимент самой последней новости дня
        last_row = group.sort_values('publish_date', kind='mergesort').iloc[-1]
        return last_row['sentiment']

    def fit(self, news_df: pd.DataFrame, candles_df: pd.DataFrame) -> None:
        # if set(candles_df.columns) != {'open', 'close', 'high', 'low', 'volume', 'begin', 'ticker'}:
        #     raise ValueError('cancles_df must contains {"open", "close", "high", "low", "volume", "begin", "ticker"}')
        # if set(news_df.columns) != {'publish_date', 'title', 'publication'}:
        #     raise ValueError('news_df must contains {"publication_date", "title", "publication"}')
        self._news = news_df.copy()
        self._candles = candles_df.copy()

        # ждем завершения всех задач
        # асинхронный вызов с прогресс-баром
        async def process():
            await asyncio.gather(
                self._sent.add_sentiment(self._news, position=self._cfg.bar_pos_sentiment),
                self._tix.add_tickers(self._news, llm_bar_pos=self._cfg.bar_pos_tickers),
            )

        asyncio.run(process())

        exploded_tickers_df = (
            self._news.explode('tickers', ignore_index=True)  # одна строка = один тикер
            .rename(columns={'tickers': 'ticker'})  # переименуем в единственное число (по желанию)
        )

        exploded_tickers_df = exploded_tickers_df.dropna(subset=['ticker'])
        exploded_tickers_df['date'] = pd.to_datetime(exploded_tickers_df['publish_date']).dt.date

        aggregated_news_daily = (
            exploded_tickers_df.sort_values(['ticker', 'publish_date'], kind='mergesort')
            .groupby(['ticker', 'date'], as_index=False)
            .apply(lambda g: pd.Series({'sentiment_daily': self.__resolve_daily_sentiment(g)}))
        )
        aggregated_news_daily = aggregated_news_daily[['ticker', 'date', 'sentiment_daily']]
        aggregated_news_daily['date'] = pd.to_datetime(aggregated_news_daily['date'])

        candles_df = self._candles.copy()
        candles_df['date'] = pd.to_datetime(pd.to_datetime(candles_df['begin']).dt.date)
        candles_df = candles_df[['date', 'volume', 'close', 'ticker']]

        final = candles_df.merge(aggregated_news_daily, on=['ticker', 'date'], how='left')
        # final.to_csv('FINAL.csv', index=False)

        # TRAIN
        df = final.rename(columns={'date': 'timestamp'})
        data = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column='ticker',
        )
        data = data.convert_frequency(freq="D")
        self.data = data
        data.to_csv('YEEEES.csv', index=False)

        self.predictor.fit(
            data,
            hyperparameters={
                "TemporalFusionTransformerModel": [{}]
            },
            enable_ensemble=False,
        )


    def predict(self) -> pd.DataFrame:
        predictions = self.predictor.predict(self.data)
        predictions = predictions.to_data_frame().reset_index()
        predictions['close'] = predictions['0.5']
        predictions['ticker'] = predictions['item_id']
        predictions = predictions.drop(columns=['item_id', 'mean', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
        predictions['begin'] = predictions['timestamp']
        data = self.data.groupby(['item_id']).tail(1).rename(columns={'item_id': 'ticker'})
        return pd.concat([data, predictions], ignore_index=True).sort_values([
            'ticker', 'begin'
        ]).reset_index(drop=True)


class PipelineBuilder:
    def __init__(self, cfg: ModelConfig):
        self._cfg = cfg

    def build(self) -> BaseModel:
        # сентимент через LLM (оставим как есть)
        semaphore = asyncio.Semaphore(self._cfg.max_concurrency)
        llm_cfg = LLMConfig(
            base_url=self._cfg.api_base,
            api_key_env=self._cfg.api_key_env,
            model_name=self._cfg.model_name,
            proxy=self._cfg.proxy,
            timeout=self._cfg.timeout_sec,
            max_retries=self._cfg.max_retries,
        )
        llm_service = LLMService(llm_cfg, semaphore)
        sentiment = SentimentLLMExtractor(llm_service)

        # ТОЛЬКО Regex стратегия
        ticker_strategy = RegexTickerStrategy()
        tickers = TickerExtractor(strategy=ticker_strategy)

        return SmartModel(
            sentiment=sentiment,
            tickers=tickers,
            cfg=self._cfg,
        )


# =========================
# 🔧 ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================
def main():
    cfg = ModelConfig(
        proxy="http://localhost:12334",
        max_concurrency=15,
        stall_timeout_sec=60,
        api_key_env="API_KEY",
    )
    model = create_model(cfg)
    logging.basicConfig(level=logging.INFO)

    news = pd.read_csv('news.csv')
    news['publish_date'] = pd.to_datetime(news['publish_date'])

    candles = pd.read_csv('candles.csv')
    model.fit(news, candles)

    result_df = model.predict()
    for i in range(1, 21):
        future_close = result_df.groupby(['ticker'])['close'].shift(-i)
        result_df[f'p{i}'] = future_close / result_df['close'] - 1
    result_df = result_df.drop(
        columns=['timestamp', 'begin', 'close', 'volume', 'sentiment_daily'],
    ).groupby(['ticker']).head(1)
    print(result_df)
    result_df.to_csv('result_df.csv')


if __name__ == '__main__':
    main()
