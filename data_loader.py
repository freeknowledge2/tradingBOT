"""
Plik: data_loader.py
Opis: Moduł dostarcza klasę MT5DataFeed, która łączy się z MetaTrader5,
      pobiera dane historyczne i aktualizuje je w tle.
      Jest zaprojektowany do łatwego importu w innych skryptach analitycznych.
"""
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Brak biblioteki MetaTrader5. Zainstaluj: pip install MetaTrader5")
    sys.exit(1)

# --- Konfiguracja ---
TIMEFRAME_BASE = mt5.TIMEFRAME_M1
HISTORY_DAYS = 1

class MT5DataFeed:
    def __init__(self, symbols: List[str]):
        load_dotenv()
        self.mt5_login = int(os.getenv("MT5_LOGIN"))
        self.mt5_password = os.getenv("MT5_PASSWORD")
        self.mt5_server = os.getenv("MT5_SERVER")

        self.symbols = symbols
        self.data_m1: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self.lock = threading.Lock()
        self.running = False

    def connect(self):
        if not mt5.initialize():
            raise RuntimeError(f"Nie udało się zainicjalizować MT5: {mt5.last_error()}")
        authorized = mt5.login(self.mt5_login, self.mt5_password, self.mt5_server)
        if not authorized:
            raise RuntimeError(f"Logowanie nieudane: {mt5.last_error()}")
        print(f"✅ Zalogowano: {self.mt5_login} @ {self.mt5_server}")
        self._ensure_symbols()

    def shutdown(self):
        self.stop_updating()
        mt5.shutdown()
        print("👋 Rozłączono z MT5")

    def _ensure_symbols(self):
        for s in self.symbols:
            if mt5.symbol_info(s) is None:
                print(f"[WARN] Symbol {s} nie istnieje.")
                continue
            if not mt5.symbol_info(s).visible:
                if not mt5.symbol_select(s, True):
                    print(f"[WARN] Nie udało się włączyć {s} w Market Watch.")

    def fetch_m1_history(self, symbol: str, since: datetime, until: datetime) -> pd.DataFrame:
        rates = mt5.copy_rates_range(symbol, TIMEFRAME_BASE, since, until)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'tick_volume']]

    def initial_load(self):
        print("Pobieram dane wstecz...")
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=HISTORY_DAYS)
        for s in self.symbols:
            with self.lock:
                self.data_m1[s] = self.fetch_m1_history(s, since, now)
        print("Dane historyczne załadowane.")

    def _update_once(self):
        now = datetime.now(timezone.utc)
        for s in self.symbols:
            with self.lock:
                df = self.data_m1[s]
                last_time = df.index.max() if not df.empty else now - timedelta(days=HISTORY_DAYS)
                since = last_time - pd.Timedelta(minutes=1)
                new_df = self.fetch_m1_history(s, since, now)
                if not new_df.empty:
                    self.data_m1[s] = pd.concat([df, new_df]).sort_index()
                    self.data_m1[s] = self.data_m1[s][~self.data_m1[s].index.duplicated(keep='last')]

    def start_updating(self):
        self.running = True
        def loop():
            while self.running:
                try:
                    self._update_once()
                except Exception as e:
                    print(f"[ERR update] {e}")
                time.sleep(60)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        print("Aktualizacje danych M1 w tle zostały uruchomione.")

    def stop_updating(self):
        self.running = False
        
    # --- POPRAWIONA LINIA ---
    # Usunięto błędną adnotację typu -> Optional[mt5.SymbolInfoTick]
    def get_tick(self, symbol: str):
        """Zwraca najnowszy tick dla danego symbolu."""
        return mt5.symbol_info_tick(symbol)

    def get_m1_data(self, symbol: str) -> pd.DataFrame:
        with self.lock:
            return self.data_m1.get(symbol, pd.DataFrame()).copy()

    def get_h1_data(self, symbol: str) -> pd.DataFrame:
        with self.lock:
            df_m1 = self.data_m1.get(symbol)
            if df_m1 is None or df_m1.empty:
                return pd.DataFrame()
            return df_m1.resample("1h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last"
            }).dropna()

    def get_live_h1_data(self, symbol: str) -> pd.DataFrame:
        completed_h1 = self.get_h1_data(symbol)
        if not completed_h1.empty:
            completed_h1 = completed_h1.iloc[:-1]
            
        m1_data = self.get_m1_data(symbol)
        tick = self.get_tick(symbol)
        if m1_data.empty or not tick:
            return completed_h1

        last_price = tick.bid
        now_utc = datetime.now(timezone.utc)
        current_candle_start_time = now_utc.replace(minute=0, second=0, microsecond=0)
        m1_current_hour = m1_data.loc[m1_data.index >= current_candle_start_time]

        if m1_current_hour.empty:
            return completed_h1
        
        live_candle = pd.DataFrame([{
            "open": m1_current_hour['open'].iloc[0],
            "high": max(m1_current_hour['high'].max(), last_price),
            "low": min(m1_current_hour['low'].min(), last_price),
            "close": last_price
        }], index=[current_candle_start_time])
        
        return pd.concat([completed_h1, live_candle])