"""
Plik: plotter.py
Opis: Przykład użycia klasy MT5DataFeed do wyświetlania
      dynamicznego wykresu świecowego H1 dla symbolu EURUSD.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# Importujemy naszą klasę z drugiego pliku
from data_loader import MT5DataFeed

# --- Konfiguracja ---
SYMBOL = "EURUSD"
UPDATE_EVERY_SEC = 1
CHART_RIGHT_PADDING = 5 # Liczba pustych świec po prawej stronie

def run_chart_demo(symbol: str):
    # 1. Inicjalizacja i połączenie
    feed = MT5DataFeed([symbol])
    try:
        feed.connect()
        feed.initial_load()
        feed.start_updating()
    except RuntimeError as e:
        print(f"Błąd krytyczny: {e}")
        sys.exit(1)

    # 2. Ustawienia wykresu - styl profesjonalny
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Definiujemy kolory świec i ogólny styl wykresu
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    style = mpf.make_mpf_style(
        base_mpf_style='charles', 
        marketcolors=mc,
        gridstyle='--',
        gridcolor='#303030'
    )
    # Ustawiamy tło całej figury (zewnętrzne) na czarne
    fig.patch.set_facecolor('black')

    # 3. Pętla odświeżająca wykres
    while plt.fignum_exists(fig.number):
        try:
            plot_df = feed.get_live_h1_data(symbol)
            tick = feed.get_tick(symbol)
            
            if plot_df.empty:
                plt.pause(UPDATE_EVERY_SEC)
                continue

            # Ustalenie ostatniej ceny
            last_price = tick.bid if tick else plot_df['close'].iloc[-1]

            # Dodanie pustego miejsca po prawej
            last_time = plot_df.index[-1]
            future_index = pd.date_range(
                start=last_time + pd.Timedelta(hours=1),
                periods=CHART_RIGHT_PADDING,
                freq='h'
            )
            padding_df = pd.DataFrame(index=future_index, columns=plot_df.columns)
            final_plot_df = pd.concat([plot_df, padding_df])

            # Rysowanie - za każdym razem czyścimy i ustawiamy parametry osi
            ax.clear()
            ax.set_facecolor('black') # Tło samego wykresu

            mpf.plot(
                final_plot_df,
                type="candle",
                style=style,
                ax=ax,
                xrotation=20,
                datetime_format="%m-%d %H:%M"
            )

            # --- DODATKI WIZUALIZACYJNE ---
            # 1. Pozioma linia aktualnej ceny
            ax.axhline(last_price, color="yellow", linestyle="--", linewidth=1)

            # 2. Etykieta z ceną po prawej stronie osi Y
            ax.text(
                1.01, last_price, f"{last_price:.5f}",
                color="black",
                fontsize=9,
                va="center",
                ha="left",
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor="yellow", edgecolor="none", boxstyle="round,pad=0.3")
            )

            # 3. Ustawienie kolorów etykiet i tytułu
            ax.set_title(f"{symbol} — H1", color="white")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            
            # Pauza, aby wykres mógł się odświeżyć
            plt.pause(UPDATE_EVERY_SEC)

        except KeyboardInterrupt:
            print("\nZatrzymywanie na żądanie użytkownika...")
            break
        except Exception as e:
            print(f"[ERR plot] {e}")
            break

    # 4. Zamykanie połączenia
    feed.shutdown()

if __name__ == "__main__":
    run_chart_demo(SYMBOL)