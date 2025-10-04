from __future__ import annotations
import abc
import asyncio
import html
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
import backoff
import httpx
import pandas as pd
from openai import AsyncClient
from tqdm.auto import tqdm
import concurrent.futures as cf
import multiprocessing as mp
import os


from expressions import regex_dict


# =========================
# Публичный интерфейс
# =========================

class BaseModel(ABC):
    @abstractmethod
    def fit(self, news_df: pd.DataFrame, candles_df: pd.DataFrame) -> None: ...
    @abstractmethod
    def predict(self, news_df: pd.DataFrame, candles_df: pd.DataFrame, path_to_save: Path) -> pd.DataFrame: ...


@dataclass
class ModelConfig:
    proxy: str
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "API_KEY"
    model_name: str = "openai/gpt-4.0-mini"
    timeout_sec: float = 10
    max_retries: int = 5
    stall_timeout_sec: float = 15.0
    max_concurrency: int = 30
    # только один прогресс-бар нужен
    bar_pos_sentiment: int = 0
    bar_pos_tickers: int = 1



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
        # можно скомпилировать, если у вас строки, но и так сработает
        self._cache: dict[str, list[re.Pattern | str]] = regex_dict

    def extract_one_sync(self, text: str) -> list[str]:
        text_str = str(text).lower()
        found_tickers: set[str] = set()

        keys = (["OZPH"] if "OZPH" in self._cache else []) + [k for k in self._cache if k != "OZPH"]
        for ticker in keys:
            patterns = self._cache[ticker]
            if any((p.search(text_str) if hasattr(p, "search") else re.search(p, text_str)) for p in patterns):
                found_tickers.add(ticker)

        if "OZPH" in found_tickers:
            found_tickers.discard("OZON")
        return list(found_tickers)

    async def extract_one(self, text: str) -> list[str]:
        # выносим CPU-bound regex в thread pool
        return await asyncio.to_thread(self.extract_one_sync, text)



class TickerExtractor:
    """
    Мультипроцессная извлекалка тикеров.
    Воркерные функции находятся внутри класса и доступны как статические методы.
    """

    # Глобальное состояние на процесс-воркер (class variable)
    _WORKER_REGEX: dict[str, list[re.Pattern]] | None = None

    def __init__(
        self,
        strategy: TickerStrategy,
        mp_workers: int | None = None,
        regex_src: dict[str, list[re.Pattern | str]] | None = None,
    ):
        self._strategy = strategy
        self._mp_workers = mp_workers or (os.cpu_count() or 4)

        # Источник паттернов (по умолчанию ваш regex_dict)
        src = regex_src if regex_src is not None else regex_dict

        # Подготовим сериализуемую версию (только строки), чтобы безболезненно передать в воркеры
        self._regex_src_serializable: dict[str, list[str]] = {
            k: [(p.pattern if isinstance(p, re.Pattern) else str(p)) for p in pats]
            for k, pats in src.items()
        }

    # ---- статические воркерные функции (доступны по пути module.TickerExtractor._init_regex_worker) ----
    @staticmethod
    def _init_regex_worker(regex_src_serializable: dict[str, list[str]]) -> None:
        """Инициализируем паттерны один раз на процесс-воркер."""
        compiled: dict[str, list[re.Pattern]] = {
            k: [re.compile(p) for p in pats] for k, pats in regex_src_serializable.items()
        }
        TickerExtractor._WORKER_REGEX = compiled

    @staticmethod
    def _extract_one_worker(text: str) -> list[str]:
        """CPU-bound: извлечь тикеры из одного текста, используя состояние воркера."""
        rx = TickerExtractor._WORKER_REGEX
        assert rx is not None, "Worker regex not initialized"
        text_str = str(text).lower()
        found: set[str] = set()

        # OZPH приоритетом
        keys = (["OZPH"] if "OZPH" in rx else []) + [k for k in rx if k != "OZPH"]
        for ticker in keys:
            for pat in rx[ticker]:
                if pat.search(text_str):
                    found.add(ticker)
                    break

        # OZPH -> исключаем OZON
        if "OZPH" in found:
            found.discard("OZON")
        return list(found)
    # ----------------------------------------------------------------------------------------------------

    def _extract_many_mp(
        self,
        texts: list[str],
        position: int,
        submit_backlog_factor: int = 4,
    ) -> list[list[str] | Exception]:
        """
        Синхронная обвязка вокруг ProcessPoolExecutor с прогрессом и ограничением очереди.
        Вызывается из async-кода через asyncio.to_thread(...), чтобы не блокировать event loop.
        """
        total = len(texts)
        results: list[list[str] | Exception] = [None] * total

        ctx = mp.get_context("spawn")  # Windows-safe
        max_workers = self._mp_workers
        max_outstanding = max_workers * submit_backlog_factor

        with cf.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=TickerExtractor._init_regex_worker,
            initargs=(self._regex_src_serializable,),
        ) as pool, tqdm(
            total=total,
            desc="Тикеры",
            position=position,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1,
        ) as bar:
            submitted: dict[cf.Future, int] = {}
            i = 0
            completed = 0

            while completed < total:
                # Досабмитим задачи, но не раздуваем очередь
                while i < total and len(submitted) < max_outstanding:
                    fut = pool.submit(TickerExtractor._extract_one_worker, texts[i])
                    submitted[fut] = i
                    i += 1

                if not submitted:
                    break  # на случай total == 0

                done, _ = cf.wait(list(submitted.keys()), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    idx = submitted.pop(fut)
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        results[idx] = e
                    completed += 1
                    bar.update(1)

        return results

    async def add_tickers(
        self,
        news_df: pd.DataFrame,
        llm_bar_pos: int,
    ) -> None:
        texts = news_df["publication"].tolist()
        # не блокируем event loop — выполняем синхронный мультипроцессинг в отдельном треде
        results = await asyncio.to_thread(self._extract_many_mp, texts, llm_bar_pos)
        news_df["ticker"] = results


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

    def fit(self, news_df: pd.DataFrame, candles_df: pd.DataFrame) -> None:
        # if set(candles_df.columns) != {'open', 'close', 'high', 'low', 'volume', 'begin', 'ticker'}:
        #     raise ValueError('cancles_df must contains {"open", "close", "high", "low", "volume", "begin", "ticker"}')
        # if set(news_df.columns) != {'publish_date', 'title', 'publication'}:
        #     raise ValueError('news_df must contains {"publication_date", "title", "publication"}')
        self._news = news_df.copy()
        self._candles = candles_df.copy()

        # асинхронный вызов с прогресс-баром
        async def process():
            await asyncio.gather(
                # self._sent.add_sentiment(self._news, position=self._cfg.bar_pos_sentiment),
                self._tix.add_tickers(self._news, llm_bar_pos=self._cfg.bar_pos_tickers),
            )

        asyncio.run(process())

        self._candles = (
            self._candles[["close", "begin", "ticker"]]
            .drop_duplicates(['ticker', 'begin'])
            .sort_values(by=["ticker", "begin"])
            .reset_index(drop=True)
        )
        self._candles["begin"] = pd.to_datetime(self._candles["begin"])

        self._news = self._news.copy()
        self._news["publish_date"] = pd.to_datetime(self._news["publish_date"])
        self._news = (
            self._news.explode('ticker', ignore_index=True)
        )

        self._candles["_date"] = self._candles["begin"].dt.date
        self._news["_date"] = self._news["publish_date"].dt.date
        self._news['text'] = self._news['title'] + ';' + self._news['publication']
        self._news["text"] = self._news["text"].apply(lambda x: html.unescape(x))

        # 3) For each (ticker, date), collect news texts ordered by published_timestamp
        self._news = (
            self._news.sort_values(
                ["ticker", "_date", "publish_date"]
            )  # ensures correct order inside lists
            .groupby(["ticker", "_date"])["text"]
            .apply(list)
            .reset_index()
            .rename(columns={"text": "news"})
        )

        # 4) Left-merge back; fill missing with empty list
        out = self._candles.merge(self._news, how="left", on=["ticker", "_date"])
        out["news"] = out["news"].apply(lambda x: x if isinstance(x, list) else [])

        # 5) (Optional) drop helper column
        out = out.drop(columns=["_date"])
        out.to_parquet('out_df.parquet')

    def predict(self, news_df: pd.DataFrame, candles_df: pd.DataFrame, path_to_save: Path | None) -> pd.DataFrame:
        if self._news is None:
            raise RuntimeError("Call fit(...) first.")
        if path_to_save:
            pass
        pass


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
        proxy="http://localhost:2080",
        max_concurrency=15,
        stall_timeout_sec=60,
    )
    model = create_model(cfg)
    news = pd.read_csv('news.csv')
    candles = pd.read_csv('candles.csv')
    model.fit(news_df=news, candles_df=candles)
    # result_df = model.predict(news_df=..., candles_df=..., path_to_save=Path('result.csv'))


if __name__ == '__main__':
    main()
