"""
SIMULAZIONE LADDER VS ETF - VERSION 2.1
Calibration based on Real-World Data (Research Jan 2026)

RESEARCH SUMMARY & PARAMETER RATIONALE:
1. ETF COSTS (TER):
   - Government (EU): ~0.07% (Ref: iShares Core Euro Government Bond IE00B4WXJJ64)
   - Corporate (EU): ~0.09% (Ref: iShares Core Euro Corp Bond IE00B3F81R35)
   - Global Govt: ~0.20% (Ref: iShares Global Govt Bond IE00B3F81K65)
   - Global Agg/Corp: ~0.10% (Ref: Vanguard Global Aggregate Bond IE00BGCZ0337)

2. ETF INTERNAL EFFICIENCY:
   - Cash Drag: Research shows very low cash exposure in large ETFs (~0.2% to 0.5%) for 
     operational liquidity. Previous 1.5% was overly penalizing.
   - Turnover Cost: Implicit transaction costs in large index ETFs are extremely low 
     due to institutional pricing and internal crossing (~0.01% reported).

3. LADDER COSTS (RETAIL):
   - BTP Spreads: Tight on MOT (~0.05% for liquid maturities).
   - Retail Commissions: Bank fees (Directa/Fineco/Webank) range from 2.5€ to 19€ 
     per trade, or ~0.19%. 
   - Calibration: Used ~0.15% for EU Gov (Spread + Comm) and higher for Corporate/Global.

4. DIVERSIFICATION:
   - Large Corporate/Global ETFs hold 4,000 to 10,000+ individual bonds.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Parametri di simulazione
YEARS = 30
N_SIM = 30000

# ==========================
# Parametri Curva dei Rendimenti (Yield Curve)
# ==========================
# Modello stilizzato: Tasso(t, m) = ShortRate(t) + Slope(t) * log(m)
# Short Rate: il tasso a brevissimo (es. 1 anno o cash)
# Slope: ripidezza della curva (spread tra lungo e breve)

# ==========================
# Selezione Scenario
# ==========================
# Options: "EU_GOV", "EU_CORP", "GLOBAL_GOV", "GLOBAL_CORP"
SELECTED_SCENARIO = "EU_GOV" 

SCENARIOS = {
    "EU_GOV": {
        "DESC": "Bond Governativi Europei (es. BTP/Bund)",
        # Market Yields (Base Rates)
        "SHORT_RATE_MEAN": 0.025, "SHORT_RATE_VOL": 0.015,
        "SLOPE_MEAN": 0.012, "SLOPE_VOL": 0.008,
        # Credit Risk (Safe)
        "DEFAULT_PROB_NORMAL": 0.0002, "DEFAULT_PROB_CRISIS": 0.0050,
        "RECOVERY_MIN": 0.50, "RECOVERY_MAX": 0.95,
        "CRISIS_PROB": 0.05,
        # ETF Params
        "ETF_TER": 0.0007, # 0.07% (iShares Core Euro Gov)
        "ETF_TER": 0.0007, # 0.07% (iShares Core Euro Gov)
        "N_ETF_ISSUERS": 20, # Eurozone countries approx.
        # Costs ETF (Liquid)
        "ETF_SPREAD": 0.0004, "ETF_SLIPPAGE": 0.0001, "ETF_STRESS_SPREAD": 0.0020,
        # Costs Ladder (Liquid on MOT: Spread ~0.05% + Comm ~0.10%)
        "LADDER_SPREAD": 0.0015, "LADDER_SLIPPAGE": 0.0002, "LADDER_STRESS_SPREAD": 0.0100,
        "N_LADDER_ISSUERS": 3,
        # Tax (White List 12.5%)
        "TAX_COUPON": 0.125, "TAX_CG": 0.125
    },
    "EU_CORP": {
        "DESC": "Bond Corporate Europei (Investment Grade)",
        # Market Yields (Higher Yield = Risk Free + Spread)
        "SHORT_RATE_MEAN": 0.035, "SHORT_RATE_VOL": 0.018, 
        "SLOPE_MEAN": 0.015, "SLOPE_VOL": 0.010,
        # Credit Risk (Moderate)
        "DEFAULT_PROB_NORMAL": 0.0020, "DEFAULT_PROB_CRISIS": 0.0300, 
        "RECOVERY_MIN": 0.30, "RECOVERY_MAX": 0.60,
        "CRISIS_PROB": 0.10, 
        # ETF Params
        "ETF_TER": 0.0009, # 0.09% (iShares Core Euro Corp)
        "N_ETF_ISSUERS": 4059,
        # Costs ETF (Relatively Liquid)
        "ETF_SPREAD": 0.0007, "ETF_SLIPPAGE": 0.0001, "ETF_STRESS_SPREAD": 0.0030,
        # Costs Ladder (OTC/TLX + Commissions)
        "LADDER_SPREAD": 0.0060, "LADDER_SLIPPAGE": 0.0005, "LADDER_STRESS_SPREAD": 0.0250,
        "N_LADDER_ISSUERS": 1, 
        # Tax (Standard 26%)
        "TAX_COUPON": 0.26, "TAX_CG": 0.26
    },
    "GLOBAL_GOV": {
        "DESC": "Bond Governativi Globali (G7 Hedged)",
        # Market Yields
        "SHORT_RATE_MEAN": 0.030, "SHORT_RATE_VOL": 0.020,
        "SLOPE_MEAN": 0.010, "SLOPE_VOL": 0.008,
        # Credit Risk (Low)
        "DEFAULT_PROB_NORMAL": 0.0005, "DEFAULT_PROB_CRISIS": 0.0100,
        "RECOVERY_MIN": 0.40, "RECOVERY_MAX": 0.90,
        "CRISIS_PROB": 0.08,
        # ETF Params
        "ETF_TER": 0.0020, # 0.20% (iShares Global Govt)
        "N_ETF_ISSUERS": 879,
        # Costs ETF (Very Liquid but Hedged)
        "ETF_SPREAD": 0.0010, "ETF_SLIPPAGE": 0.0001, "ETF_STRESS_SPREAD": 0.0025,
        # Costs Ladder (Retail access to global bonds is expensive)
        "LADDER_SPREAD": 0.0100, "LADDER_SLIPPAGE": 0.0010, "LADDER_STRESS_SPREAD": 0.0300,
        "N_LADDER_ISSUERS": 1,
        # Tax (White List 12.5% mostly)
        "TAX_COUPON": 0.125, "TAX_CG": 0.125
    },
    "GLOBAL_CORP": {
        "DESC": "Bond Corporate Globali (Aggregate Mix)",
        # Market Yields (High)
        "SHORT_RATE_MEAN": 0.045, "SHORT_RATE_VOL": 0.025,
        "SLOPE_MEAN": 0.020, "SLOPE_VOL": 0.012,
        # Credit Risk (Moderate to High)
        "DEFAULT_PROB_NORMAL": 0.0050, "DEFAULT_PROB_CRISIS": 0.0800, 
        "RECOVERY_MIN": 0.20, "RECOVERY_MAX": 0.50,
        "CRISIS_PROB": 0.15,
        # ETF Params
        "ETF_TER": 0.0010, # 0.10% (Vanguard Global Aggregate)
        "N_ETF_ISSUERS": 10000,
        # Costs ETF (Liquidish)
        "ETF_SPREAD": 0.0010, "ETF_SLIPPAGE": 0.0002, "ETF_STRESS_SPREAD": 0.0050,
        # Costs Ladder (Logistically hard/expensive for retail)
        "LADDER_SPREAD": 0.0200, "LADDER_SLIPPAGE": 0.0030, "LADDER_STRESS_SPREAD": 0.0500,
        "N_LADDER_ISSUERS": 1,
        # Tax (Standard 26%)
        "TAX_COUPON": 0.26, "TAX_CG": 0.26
    }
}

# --- Apply Selection ---
params = SCENARIOS[SELECTED_SCENARIO]
print(f"LOADING SCENARIO: {SELECTED_SCENARIO} - {params['DESC']}")

# Market
SHORT_RATE_MEAN = params["SHORT_RATE_MEAN"]
SHORT_RATE_VOL = params["SHORT_RATE_VOL"]
SLOPE_MEAN = params["SLOPE_MEAN"]
SLOPE_VOL = params["SLOPE_VOL"]

# Credit
DEFAULT_PROB_NORMAL = params["DEFAULT_PROB_NORMAL"]
DEFAULT_PROB_CRISIS = params["DEFAULT_PROB_CRISIS"]
RECOVERY_MIN = params["RECOVERY_MIN"]
RECOVERY_MAX = params["RECOVERY_MAX"]
CRISIS_PROB = params["CRISIS_PROB"]

# Tax
TAX_RATE_COUPON = params["TAX_COUPON"]
TAX_RATE_CG = params["TAX_CG"]

# Ladder Costs
LADDER_BID_ASK_SPREAD = params["LADDER_SPREAD"]
LADDER_SLIPPAGE = params["LADDER_SLIPPAGE"]
LADDER_STRESS_SPREAD = params["LADDER_STRESS_SPREAD"]
N_LADDER_ISSUERS_PER_RUNG = params["N_LADDER_ISSUERS"]

# ETF Costs
ETF_TER = params["ETF_TER"]
ETF_BID_ASK_SPREAD = params["ETF_SPREAD"]
ETF_SLIPPAGE = params["ETF_SLIPPAGE"]
ETF_STRESS_SPREAD = params["ETF_STRESS_SPREAD"]
N_ETF_ISSUERS_TOTAL = params["N_ETF_ISSUERS"]

# N_SIM definita sopra (30000)
INF_MEAN = 0.02
INF_VOL = 0.01
DF_T = 5

# Correlazioni
RHO_RATE_INF = 0.4 
RHO_RATE_SLOPE = -0.5

# Parametri Flussi
SPEND_PROB = 0.3
SPEND_MIN = 1000.0
SPEND_MAX = 15000.0
ANNUAL_SAVINGS = 12_000.0 
INITIAL_WEALTH_DEC = 100_000.0
INITIAL_WEALTH_ACC = 10_000.0

# --- Nuovi Parametri per Bilanciamento Ladder vs ETF ---
# Punto 2: Fiscale
INITIAL_TAX_LOSSES = 0.0   # 0k di minusvalenze pregresse recuperabili solo dai bond (zainetto fiscale)
# Punto 3: Costi occulti ETF
ETF_INTERNAL_TURNOVER_COST = 0.0001 # 0.01% di drag annuale per trading interno e spread del fondo
ETF_INTERNAL_CASH_DRAG = 0.003      # 0.3% del fondo è tenuto in cash (non rende nulla)

# Simulation Modes
SIMULATION_MODES_TO_RUN = ["accumulation", "decumulation", "mixed"]

LADDER_STRATEGIES = {
    "Ladder": [2, 4, 6, 8, 10],   
    "Ladder (Short 2-4y)": [2, 3, 4],
    "Ladder (long 6-10y)": [6, 8, 10]
}

ETF_STRATEGIES = {
    "ETF_LadderLike": [2, 4, 6, 8, 10],   
    "ETF_Short (2y)": [2],
    "ETF_Short (4y)": [4],                
    "ETF_Medium (6y)": [6],               
    "ETF_Long (10y)": [10],               
    "ETF_Barbell (2+10)": [2, 10]         
}

CASH_THRESHOLD = 5_000.0
CASH_REINVEST_FRACTION = 0.5
SELL_STRATEGY_LADDER = "shortest_first"
SELL_STRATEGY_ETF = "shortest_first"

# Parametri "Human Factor" (Pigrizia dell'investitore)
LAZY_DAILY_AWARENESS_PROB = 0.008 # ~0.8% prob. giornaliera di investire (~80% di operare entro l'anno, ~20% skip)
# Con 0.008, il tempo medio di attesa è 1/0.008 = 125 giorni lavorativi.
LAZY_CASH_THRESHOLD = 10000.0    # Aspetta di avere 15k prima di muoversi (efficienza cash)
LAZY_SLIPPAGE_PENALTY = 0.0015   # +0.15% di slippage extra (ordini a mercato vs limit)

# Costanti Magic Numbers
ROLL_DOWN_EFFICIENCY = 0.9  # Approssimazione duration × yield spread
RUIN_THRESHOLD = 0.95       # 5% buffer per costi/tasse prima di rovina
TRADING_DAYS_PER_YEAR = 200

np.random.seed(42)


# ==========================
# Funzioni di utilità
# ==========================

def get_rate_at_maturity(short_rate, slope, maturity):
    """
    Restituisce il tasso spot (o array di tassi) per una data maturità.
    Supporta input vettoriali (array NumPy).
    """
    if isinstance(maturity, (np.ndarray, list)):
        maturity = np.array(maturity)
        
        # Gestione broadcasting: se short_rate è (N,) e maturity è (N, M)
        sr = np.asanyarray(short_rate)
        sl = np.asanyarray(slope)
        
        if sr.ndim == 1 and maturity.ndim == 2:
            sr = sr[:, np.newaxis]
            sl = sl[:, np.newaxis]
            
        res = np.where(maturity <= 1, sr, sr + sl * np.log10(np.maximum(1.0, maturity)))
        return res
    else:
        if maturity <= 1:
            return short_rate
        # DOC LOG10 vs LN:
        # Usiamo Log10 per convenzione di calibrazione parametri SLOPE_MEAN/VOL trovati in letteratura
        # o semplicemente per appiattire la curva più velocemente sulle lunghe scadenze rispetto a Ln.
        # Parametri calibrati coerentemente.
        return short_rate + slope * np.log10(maturity)

def sample_brownian_bridge(start_val, end_val, fraction, vol):
    """
    Campiona un valore intra-year usando un Brownian Bridge.
    X(t) ~ InterpolazioneLineare + Rumore(t)
    Rumore(t) ha var = t * (1-t) * vol^2
    """
    # Linear interpolation
    mu = start_val + (end_val - start_val) * fraction
    
    # Noise (Bridge Standard Deviation)
    # std = vol * sqrt(t * (1-t))
    # fraction è array (N_SIM,)
    std = vol * np.sqrt(np.clip(fraction * (1.0 - fraction), 0.0, 1.0))
    
    noise = np.random.normal(0, 1, size=start_val.shape) * std
    return mu + noise

# ==========================
# Funzioni di utilità Pricing & Tax
# ==========================

def calculate_trade_impact(amount, direction, spread, slippage):
    """
    Calcola l'impatto dei costi di transazione. Supporta array NumPy.
    """
    cost_rate = (spread / 2.0) + slippage
    cost = amount * cost_rate
    
    # Se direction è 1 (compra), il costo si aggiunge (paghi di più)
    # Se direction è -1 (vendi), il costo si sottrae (incassi meno)
    # Lavoriamo in modo vettoriale se amount è un array
    return amount + (direction * cost), cost

def calc_fiscale_gain_tax(sell_price, buy_price, quantity):
    """
    Calcola Capital Gain e Tasse.
    """
    gain = (sell_price - buy_price) * quantity
    tax = 0.0
    if gain > 0:
        tax = gain * TAX_RATE_CG
    return gain, tax

def get_bond_price(face, coupon, maturity, current_rate):
    """
    Calcola il prezzo di mercato di un bond (Vanilla). Supporta array NumPy.
    """
    # Gestione maturità <= 0 (vettoriale)
    # Usiamo np.where se gli input sono array
    is_array = isinstance(maturity, np.ndarray) or isinstance(current_rate, np.ndarray)
    
    if not is_array:
        if maturity <= 0: return face
        if abs(current_rate) < 1e-9: return face * (1 + coupon * maturity)
        df = (1 + current_rate)**(-maturity)
        c_val = face * coupon * (1 - df) / current_rate
        f_val = face * df
        return c_val + f_val
    else:
        # Versione vettoriale
        # Assicuriamoci che siano tutti array
        face = np.asanyarray(face)
        coupon = np.asanyarray(coupon)
        maturity = np.asanyarray(maturity)
        current_rate = np.asanyarray(current_rate)
        
        # Maschere per casi speciali
        res = np.zeros_like(maturity, dtype=float)
        
        # Caso maturità <= 0
        mask_m0 = (maturity <= 0)
        res[mask_m0] = face[mask_m0] if face.shape else face
        
        # Caso rate ~ 0
        mask_r0 = (~mask_m0) & (np.abs(current_rate) < 1e-9)
        if np.any(mask_r0):
            # Calcolo del valore a tasso zero: face + tutte le cedole (face * coupon * maturity)
            term = face * (1 + coupon * maturity)
            if np.isscalar(term):
                res[mask_r0] = term
            else:
                res[mask_r0] = term[mask_r0]
        
        # Caso standard
        mask_std = (~mask_m0) & (~mask_r0)
        if np.any(mask_std):
            m = maturity[mask_std]
            r = current_rate[mask_std]
            # Se face è scalare, non posso indicizzarlo con la maschera.
            f = face if np.isscalar(face) or face.ndim == 0 else face[mask_std]
            c = coupon if np.isscalar(coupon) or coupon.ndim == 0 else coupon[mask_std]
            
            df = (1 + r)**(-m)
            cv = f * c * (1 - df) / r
            fv = f * df
            res[mask_std] = cv + fv
            
        return res


def simulate_market_scenarios(years, n_sim):
    """
    Genera scenari per:
    - Short Rate (tasso a breve)
    - Slope (inclinazione curva)
    - Inflazione
    
    Usa una matrice di covarianza per legare le 3 variabili.
    """
    # 3 variabili: [ShortRate, Slope, Inflazione]
    means = np.array([SHORT_RATE_MEAN, SLOPE_MEAN, INF_MEAN])
    stds = np.array([SHORT_RATE_VOL, SLOPE_VOL, INF_VOL])
    
    # Matrice di correlazione
    # RHO_RATE_SLOPE: Tassi alti -> Slope basso (inversione)
    # RHO_RATE_INF: Tassi alti -> Inflazione alta (risposta banche centrali o causa)
    # Slope vs Inf: ipotizziamo 0 per semplicità, o negativa (recessione -> taglio tassi -> curva ripida)
    corr_matrix = np.array([
        [1.0,            RHO_RATE_SLOPE, RHO_RATE_INF],
        [RHO_RATE_SLOPE, 1.0,            -0.2         ],
        [RHO_RATE_INF,   -0.2,           1.0          ]
    ])
    
    # Costruiamo covarianza
    cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
    
    # Generazione rumore (Multivariate Student-t sarebbe ideale, qui usiamo Normale per semplicità di correlazione 
    # oppure Student-t scorrelata e poi colorata via Cholesky, facciamo quest'ultima per mantenere le code grasse)
    
    # 1. Z standard (t-student)
    z = np.random.standard_t(DF_T, size=(n_sim, years, 3))
    # Normalizziamo Z per avere std=1 approx (la t-student ha var > 1)
    z = z / np.sqrt(DF_T / (DF_T - 2))
    
    # 2. Correlazione via Cholesky decomposition della matrice di correlazione
    L = np.linalg.cholesky(corr_matrix)
    
    # Applichiamo rotazione: z_corr[sim, t] = z[sim, t] @ L.T
    z_corr = np.einsum('ijk,lk->ijl', z, L)
    
    # 3. Scaling e Mean
    # scenarios[sim, t, 0] -> Short Rate
    # scenarios[sim, t, 1] -> Slope
    # scenarios[sim, t, 2] -> Inflation
    scenarios = np.zeros_like(z_corr)
    for i in range(3):
        scenarios[:, :, i] = means[i] + z_corr[:, :, i] * stds[i]
        
    # Clipping per evitare valori assurdi
    # Rate > -2%, Slope tra -3% e +5%, Inflation > -5%
    scenarios[:, :, 0] = np.clip(scenarios[:, :, 0], -0.02, 0.15)
    scenarios[:, :, 1] = np.clip(scenarios[:, :, 1], -0.05, 0.08)
    scenarios[:, :, 2] = np.clip(scenarios[:, :, 2], -0.05, 0.20)
    
    return scenarios


def init_ladder_portfolio(initial_wealth, maturities, short_rate, slope):
    """
    Inizializza Ladder: Compra bond al prezzo Ask (inclusi costi).
    """
    n_bonds = len(maturities)
    per_bond_cash = initial_wealth / n_bonds
    bonds = []
    
    for m in maturities:
        coupon = get_rate_at_maturity(short_rate, slope, m)
        # 1. Calcolo Prezzo Fair
        fair_price_unit = get_bond_price(1.0, coupon, m, coupon) # Yield=Coupon -> Price=1.0 approx (al netto convessità)
        #    Attenzione: get_bond_price usa un tasso di sconto. Qui usiamo yield=coupon quindi prezzo ~100.
        #    Tuttavia `get_bond_price` prende 'current_rate' che è un tasso.
        #    Se usiamo 'coupon' come discount rate, p=1.0. Correct.
        
        # 2. Prezzo Acquisto (Ask)
        #    Ask = Fair + Costs
        buy_price_unit_gross, _ = calculate_trade_impact(fair_price_unit, 1, LADDER_BID_ASK_SPREAD, LADDER_SLIPPAGE)
        
        # 3. Quanti Face Value compro con `per_bond_cash`?
        #    Cash = Face * BuyPrice
        face = per_bond_cash / buy_price_unit_gross
        
        # RAFFINAMENTO FISCALE V2: Memorizziamo cost_basis totale
        cost_basis = face * buy_price_unit_gross
        
        bonds.append({
            "maturity": m, 
            "face": face, 
            "coupon": coupon,
            "cost_basis": cost_basis, # Costo fiscale totale del lotto
            "n_issuers": N_LADDER_ISSUERS_PER_RUNG
        })
    cash = 0.0 # Tutto investito (salvo spiccioli decimali ignorati)
    return bonds, cash


def ladder_value(bonds, cash, current_rate):
    """
    Valore Mark-to-Market del ladder.
    """
    total = cash
    for b in bonds:
        total += get_bond_price(b["face"], b["coupon"], b["maturity"], current_rate)
    return total


def roll_ladder_one_year(bonds, cash, short_rate, slope, spread, slippage):
    """
    Avanza di un anno:
    - Incasso Cedole (Taxed)
    - Bond Scaduti (Redemption - Taxed on CG)
    - Reinvestimento (New Buy - Costs)
    """
    new_bonds = []
    
    # --- Incasso Cedole ---
    for b in bonds:
        # Cedola lorda
        coupon_gross = b["face"] * b["coupon"]
        # Tassazione immediata
        tax = coupon_gross * TAX_RATE_COUPON
        cash += (coupon_gross - tax)
        
    # --- Rollover ---
    for b in bonds:
        b["maturity"] -= 1
        if b["maturity"] <= 0:
            # --- Scadenza ---
            # Rimborso a 100 (Face)
            redemption_value = b["face"] # * 1.0
            
            # Calcolo Tasse Capital Gain (su rimborso finale)
            # Gain = Valore Rimborso (Face) - Costo Carico Totale (cost_basis)
            # Se cost_basis mancante (vecchi bond), stimiamo da face (approx par)
            cost_basis = b.get("cost_basis", b["face"]) 
            
            gain = redemption_value - cost_basis
            
            # Applicazione Tassa CG solo se gain > 0
            tax_cg = max(0.0, gain * TAX_RATE_CG)
            
            cash += (redemption_value - tax_cg)
        else:
            new_bonds.append(b)

    # --- Reinvestimento (Nuovi Bond) ---
    if len(bonds) > 0:
        max_mat = max(b["maturity"] for b in bonds) + 1  
    else:
        max_mat = 10 

    n_original = len(bonds)
    n_new_needed = n_original - len(new_bonds)
    
    if n_new_needed > 0 and cash > 0:
        per_bond_cash = cash / (n_new_needed + 1e-8)
        
        # Tasso mercato attuale per i nuovi bond
        rate_at_mat = get_rate_at_maturity(short_rate, slope, max_mat)
        
        # Prezzo Fair (usiamo lo yield 'rate_at_mat')
        # Se Coupon = Yield -> Price = 1.0
        # Quindi emettiamo alla pari, ma paghiamo costi transazione
        fair_price_unit = 1.0 
        buy_price_unit, _ = calculate_trade_impact( fair_price_unit, 1, spread, slippage )
        
        for _ in range(n_new_needed):
            # Cash = Face * BuyPrice
            face = per_bond_cash / buy_price_unit
            if face > 0:
                # Scalo il cash effettivamente usato
                # CostBasis = Cash Usato (che include costi transazione impliciti nel calcolo di face)
                # face = cash / buy_price -> cash = face * buy_price. 
                # Esatto: per_bond_cash è quello che abbiamo speso.
                actual_cost = face * buy_price_unit
                
                cash -= actual_cost 
                new_bonds.append({
                    "maturity": max_mat, 
                    "face": face, 
                    "coupon": rate_at_mat,
                    "cost_basis": actual_cost,
                    "n_issuers": N_LADDER_ISSUERS_PER_RUNG
                })

    return new_bonds, cash


def sell_from_ladder(bonds, cash, amount, strategy, short_rate, slope, spread, slippage):
    """
    Vende dal ladder (MTM) applicando Bid-Ask spread e Tasse Capital Gain.
    """
    if amount <= 0:
        return bonds, cash, 0.0

    if cash >= amount:
        cash -= amount
        return bonds, cash, amount

    amount_remaining = amount - cash
    cash = 0.0

    # Lavoriamo su copie per non modificare lista in caso di errore non handled, 
    # e per semplicità di logica in-place.
    temp_bonds = [b.copy() for b in bonds]

    if strategy == "shortest_first":
        sorted_indices = np.argsort([b["maturity"] for b in temp_bonds])
    else: 
        sorted_indices = range(len(temp_bonds))
    
    generated_cash = 0.0
    
    for i in sorted_indices:
        b = temp_bonds[i]
        
        if generated_cash >= amount_remaining:
            # Ho già coperto il bisogno, non tocco questo bond
            continue
        
        needed = amount_remaining - generated_cash
        
        # 1. Prezzo Fair
        r_bond = get_rate_at_maturity(short_rate, slope, b["maturity"])
        fair_price_unit = get_bond_price(1.0, b["coupon"], b["maturity"], r_bond)
        
        # 2. Prezzo Bid (Vendita)
        bid_price_unit, _ = calculate_trade_impact(fair_price_unit, -1, spread, slippage)
        
        # 3. Tasse
        # Calcolo Prezzo Carico Unitario Implicito dal Cost Basis totale
        if b["face"] > 1e-9:
            avg_load_price = b["cost_basis"] / b["face"]
        else:
            avg_load_price = 1.0 # fallback

        # GainUnit
        gain_unit = bid_price_unit - avg_load_price
        tax_unit = max(0, gain_unit * TAX_RATE_CG)
        net_proceeds_unit = bid_price_unit - tax_unit
        
        # 4. Quanta Face devo vendere?
        face_to_sell = needed / (net_proceeds_unit + 1e-9)
        
        if face_to_sell >= b["face"]:
            # Vendo tutto
            proceeds = b["face"] * net_proceeds_unit
            generated_cash += proceeds
            b["face"] = 0.0
            b["cost_basis"] = 0.0
        else:
            # Vendo parziale
            proceeds = face_to_sell * net_proceeds_unit
            generated_cash += proceeds
            
            # Reduce Cost Basis Pro-Rata
            fraction_sold = face_to_sell / b["face"]
            b["cost_basis"] *= (1.0 - fraction_sold)
            
            b["face"] -= face_to_sell

    # Ricostruisco lista finale tenendo solo i sopravvissuti
    final_bonds = [b for b in temp_bonds if b["face"] > 1e-6]
    
    cash += generated_cash
    return final_bonds, cash, amount - max(0.0, amount_remaining - generated_cash)


def init_etf_portfolio(initial_wealth, durations, short_rate, slope):
    """
    Inizializza portafoglio ETF. Include Slippage e Spread iniziale (costo ingresso).
    """
    n = len(durations)
    per_etf_cash = initial_wealth / n
    etfs = []
    
    # Costo ingresso su ETF
    # Value = Cash investito - Costi
    # Compri a Ask = Fair * (1 + costs)
    # Valore 'Book' (NAV) è Fair. 
    # Quindi Units = Cash / Ask. Value = Units * Fair = Cash / (1+costs)
    
    # Calculate the effective purchase price per unit (assuming NAV starts at 1.0)
    # This is the price paid including transaction costs
    purchase_price_per_unit, _ = calculate_trade_impact(1.0, 1, ETF_BID_ASK_SPREAD, ETF_SLIPPAGE) # Assuming NAV is 1.0 initially
    
    for d in durations:
        y = get_rate_at_maturity(short_rate, slope, d)
        
        # Units acquired: total cash allocated / purchase price per unit
        units_acquired = per_etf_cash / purchase_price_per_unit
        
        # Initial value of the ETF (NAV * units)
        # Assuming NAV starts at 1.0, so value = units_acquired
        initial_val = units_acquired * 1.0 # Current NAV is 1.0
        
        etfs.append({
            "value": initial_val, 
            "duration": d, 
            "yield": y, 
            "avg_buy_price": purchase_price_per_unit, # This is the fiscal cost basis per unit
            "units": units_acquired  # Number of units held
        })
    cash = 0.0
    return etfs, cash


def etf_value(etfs, cash):
    # Valore di liquidazione (Bid Price)
    # Valore NAV = value.
    # Se vendo tutto ora: value * (1 - costs)
    # Qui ritorniamo il NAV puro per coerenza contabile, ma 
    # se serve MTM di liquidazione bisogna applicare spread.
    # Teniamo NAV puro per i grafici
    return cash + sum(e["value"] for e in etfs)


def evolve_etfs_one_year(etfs, cash, short_rate_curr, slope_curr, short_rate_prev, slope_prev):
    """
    Evoluzione ETF con Yield Drag, Price Change, Roll Down, e TER.
    """
    
    for e in etfs:
        d = e["duration"]
        
        rate_prev = get_rate_at_maturity(short_rate_prev, slope_prev, d)
        rate_curr = get_rate_at_maturity(short_rate_curr, slope_curr, d)
        dr = rate_curr - rate_prev
        
        # 1. Income return
        income_ret = e["yield"]
        
        # 2. Price return
        price_ret = -d * dr
        
        # 3. Roll Down Return
        y_now = get_rate_at_maturity(short_rate_curr, slope_curr, d)
        y_shorter = get_rate_at_maturity(short_rate_curr, slope_curr, d - 1 if d > 1 else 0.5)
        roll_down_ret = d * (y_now - y_shorter) * ROLL_DOWN_EFFICIENCY

        # TER (Costo annuo)
        ter_cost = ETF_TER
        
        total_ret = income_ret + price_ret + roll_down_ret - ter_cost
        
        # Applico rendimento al valore (NAV)
        e["value"] *= (1 + total_ret)
        # Units restano uguali (Accumulazione interna aumenta il NAV unitario)
        # Ma nel nostro modello semplificato "value" e "units" divergono.
        # Value = Units * NAV.
        # Qui value sale, Units fisse -> NAV Sale.
        # Avg Buy Price (fiscale) resta fisso.
        
        # 3. Turnover Yield Update
        if d > 0:
            turnover_fraction = 1.0 / d
        else:
            turnover_fraction = 1.0
        e["yield"] = e["yield"] * (1 - turnover_fraction) + rate_curr * turnover_fraction

    return etfs, cash


def sell_from_etfs(etfs, cash, amount, strategy, spread, slippage):
    """
    Vendita ETF con Costi e Tasse Capital Gain.
    """
    if amount <= 0:
        return etfs, cash, 0.0

    if cash >= amount:
        cash -= amount
        return etfs, cash, amount

    amount_remaining = amount - cash
    generated_cash = 0.0
    
    # Ordine
    if strategy == "shortest_first":
         etfs_sorted_idx = np.argsort([e["duration"] for e in etfs])
    else: 
         etfs_sorted_idx = range(len(etfs))

    for idx in etfs_sorted_idx:
        e = etfs[idx]
        if generated_cash >= amount_remaining:
            continue
            
        needed = amount_remaining - generated_cash
        
        # Valutazione Unit NAV attuale
        # Current Value = e["value"]
        # Units = e["units"]
        # NAV attuale = Value / Units
        current_nav = e["value"] / (e["units"] + 1e-9)
        
        # Prezzo Bid (Vendita)
        bid_nav, _ = calculate_trade_impact(current_nav, -1, spread, slippage)
        
        # Tasse Unit
        # Gain = Bid - AvgBuy
        gain_unit = bid_nav - e["avg_buy_price"]
        tax_unit = max(0, gain_unit * TAX_RATE_CG)
        net_proceeds_unit = bid_nav - tax_unit
        
        # Quante Units vendere?
        units_to_sell = needed / (net_proceeds_unit + 1e-9)
        
        if units_to_sell >= e["units"]:
            # Vendo tutto
            units_sold = e["units"]
            proceeds = units_sold * net_proceeds_unit
            generated_cash += proceeds
            
            e["units"] = 0.0
            e["value"] = 0.0
        else:
            # Vendo parziale
            units_sold = units_to_sell
            proceeds = units_sold * net_proceeds_unit
            generated_cash += proceeds
            
            e["units"] -= units_sold
            # Value si riduce proporzionalmente
            e["value"] -= (units_sold * current_nav) # Riduzione valore lordo portafoglio

    cash += generated_cash
    return etfs, cash, amount - max(0.0, amount_remaining - generated_cash)


def reinvest_cash_ladder(bonds, cash, short_rate, slope, threshold, fraction, spread, slippage):
    """
    Reinvestimento ladder con Costi.
    """
    if cash <= threshold:
        return bonds, cash
    investable = (cash - threshold) * fraction
    if investable <= 0:
        return bonds, cash
    
    if len(bonds) > 0:
        max_mat = max(b["maturity"] for b in bonds)
    else:
        max_mat = 10
        
    rate_at_mat = get_rate_at_maturity(short_rate, slope, max_mat)
    
    # Costi acquisto
    fair_price = 1.0
    ask_price, _ = calculate_trade_impact(fair_price, 1, spread, slippage)
    
    face = investable / ask_price
    actual_cost = face * ask_price
    cash -= actual_cost # Usa costo esatto
    
    bonds.append({
        "maturity": max_mat, 
        "face": face, 
        "coupon": rate_at_mat,
        "cost_basis": actual_cost,
        "n_issuers": N_LADDER_ISSUERS_PER_RUNG
    })
    
    return bonds, cash


def reinvest_cash_etfs(etfs, cash, threshold, fraction, spread, slippage):
    """
    Reinvestimento ETF: compra units, aggiorna prezzo medio carico.
    """
    if cash <= threshold:
        return etfs, cash
    investable = (cash - threshold) * fraction
    if investable <= 0:
        return etfs, cash

    total_value = sum(e["value"] for e in etfs)
    
    # Distribuzione
    if total_value <= 0:
         # Equamente
         allocations = [investable / len(etfs)] * len(etfs)
    else:
         allocations = [(e["value"] / total_value) * investable for e in etfs]
    
    cash -= investable
    
    for i, e in enumerate(etfs):
        amount = allocations[i]
        if amount < 1e-2: continue
            
        current_nav = e["value"] / (e["units"] + 1e-9)
        ask_nav, cost = calculate_trade_impact(current_nav, 1, spread, slippage)
        
        if ask_nav < 1e-6:
            # ETF collassato a zero, impossibile reinvestire
            continue

        new_units = amount / ask_nav
        
        # Aggiornamento Prezzo Medio Ponderato (Weighted Avg Price)
        # NewAvg = (OldUnits * OldAvg + NewUnits * BuyPrice) / (OldUnits + NewUnits)
        old_units = e["units"]
        old_avg = e["avg_buy_price"]
        
        new_avg = (old_units * old_avg + new_units * ask_nav) / (old_units + new_units + 1e-9)
        
        e["units"] += new_units
        e["avg_buy_price"] = new_avg
        # Removed dead code: e["value"] += amount
        # No, value = units * current_nav.
        # Abbiamo comprato new_units. Quindi value += new_units * current_nav.
        # amount speso includeva costi. Value aumenta di (amount - costi).
        e["value"] = e["units"] * current_nav
        
    return etfs, cash


# ==========================
# Simulazione Monte Carlo
# ==========================

def run_simulation(mode):
    try:
        scenarios = simulate_market_scenarios(YEARS, N_SIM)
        
        initial_w = INITIAL_WEALTH_ACC if mode in ["accumulation", "mixed"] else INITIAL_WEALTH_DEC

        # --- Pre-generazione Eventi Casuali ---
        is_crisis = np.random.rand(N_SIM, YEARS) < CRISIS_PROB
        
        # Simulazione pigrizia: Giorno di reazione (1-200). Se > 200, l'investitore non opera quell'anno.
        # Usiamo una distribuzione geometrica per simulare i tentativi giornalieri
        lazy_days_to_react = np.random.geometric(LAZY_DAILY_AWARENESS_PROB, size=(N_SIM, YEARS))
        
        # Simulazione Giorno di Spesa (Decumulo): random uniform durante l'anno (per mico-oscillazioni)
        spending_days = np.random.uniform(1, TRADING_DAYS_PER_YEAR, (N_SIM, YEARS))
        
        spend_events = np.random.rand(N_SIM, YEARS) < SPEND_PROB
        spend_amounts_base = np.random.uniform(SPEND_MIN, SPEND_MAX, (N_SIM, YEARS))

        # --- Inizializzazione Vettoriale ---
        ladder_datas = {}
        sr0, sl0 = scenarios[:, 0, 0], scenarios[:, 0, 1]
        
        for name, maturities in LADDER_STRATEGIES.items():
            m_arr = np.array(maturities)
            n_mat = len(m_arr)
            face = np.zeros((N_SIM, n_mat))
            coupon = np.zeros((N_SIM, n_mat))
            cost_basis = np.zeros((N_SIM, n_mat))
            mats = np.tile(m_arr, (N_SIM, 1)).astype(float)
            
            per_bond_cash = initial_w / n_mat
            for i in range(n_mat):
                m = m_arr[i]
                c = get_rate_at_maturity(sr0, sl0, m)
                buy_p, _ = calculate_trade_impact(1.0, 1, LADDER_BID_ASK_SPREAD, LADDER_SLIPPAGE)
                face[:, i] = per_bond_cash / buy_p
                coupon[:, i] = c
                cost_basis[:, i] = face[:, i] * buy_p

            ladder_datas[name] = {
                "face": face, "coupon": coupon, "cost": cost_basis, "mats": mats,
                "cash": np.zeros(N_SIM), "ruined": np.zeros(N_SIM, dtype=bool), "hist": [],
                "mats_orig": m_arr,
                "zainetto": np.full(N_SIM, INITIAL_TAX_LOSSES) # Zainetto fiscale per ogni simulazione
            }
            
            # Crea versione Lazy per ogni Ladder
            lazy_name = f"{name} (Lazy)"
            ladder_datas[lazy_name] = {
                "face": face.copy(), "coupon": coupon.copy(), "cost": cost_basis.copy(), "mats": mats.copy(),
                "cash": np.zeros(N_SIM), "ruined": np.zeros(N_SIM, dtype=bool), "hist": [],
                "mats_orig": m_arr,
                "zainetto": np.full(N_SIM, INITIAL_TAX_LOSSES)
            }

        # ETFs
        etf_datas = {}
        for name, durations in ETF_STRATEGIES.items():
            n_e = len(durations)
            e_vals = np.zeros((N_SIM, n_e))
            e_units = np.zeros((N_SIM, n_e))
            e_avg_buy = np.zeros((N_SIM, n_e))
            e_yields = np.zeros((N_SIM, n_e))
            
            buy_p, _ = calculate_trade_impact(1.0, 1, ETF_BID_ASK_SPREAD, ETF_SLIPPAGE)
            per_etf_cash = initial_w / n_e
            for i, d in enumerate(durations):
                y = get_rate_at_maturity(sr0, sl0, d)
                e_units[:, i] = per_etf_cash / buy_p
                e_vals[:, i] = e_units[:, i] # Navigator NAV=1.0
                e_avg_buy[:, i] = buy_p
                e_yields[:, i] = y
                
            etf_datas[name] = {"val": e_vals, "units": e_units, "avg_buy": e_avg_buy, "yields": e_yields, "cash": np.zeros(N_SIM), "ruined": np.zeros(N_SIM, dtype=bool), "hist": []}

        # ETF Lazy (LadderLike)
        el_durations = ETF_STRATEGIES["ETF_LadderLike"]
        n_el = len(el_durations)
        EL_val = np.zeros((N_SIM, n_el))
        EL_units = np.zeros((N_SIM, n_el))
        EL_avg_buy = np.zeros((N_SIM, n_el))
        EL_yields = np.zeros((N_SIM, n_el))
        EL_cash = np.zeros(N_SIM)
        EL_ruined = np.zeros(N_SIM, dtype=bool)
        # Init Lazy ETF (stessa logica)
        buy_p_e, _ = calculate_trade_impact(1.0, 1, ETF_BID_ASK_SPREAD, ETF_SLIPPAGE)
        for i, d in enumerate(el_durations):
            EL_units[:, i] = (initial_w / n_el) / buy_p_e
            EL_val[:, i] = EL_units[:, i]
            EL_avg_buy[:, i] = buy_p_e
            EL_yields[:, i] = get_rate_at_maturity(sr0, sl0, d)

        # Risultati
        L_hist, LL_hist, EL_hist = [], [], []
        cum_infl = np.ones(N_SIM)
        prev_sr, prev_sl = sr0.copy(), sl0.copy()

        # --- Init Tracking for CAGR ---
        # "prev_wealth" tracks the wealth at the START of the annual loop (before savings)
        for name, p in ladder_datas.items():
            r_init = get_rate_at_maturity(sr0, sl0, p["mats"])
            w_init = p["cash"] + get_bond_price(p["face"], p["coupon"], p["mats"], r_init).sum(axis=1)
            p["prev_wealth"] = w_init
            p["cum_yield"] = np.ones(N_SIM)

        for name, p in etf_datas.items():
             w_init = p["cash"] + p["val"].sum(axis=1)
             p["prev_wealth"] = w_init
             p["cum_yield"] = np.ones(N_SIM)
        
        # EL
        EL_prev_wealth = EL_cash + EL_val.sum(axis=1)
        EL_cum_yield = np.ones(N_SIM)

        # --- LOOP ANNUALE (GIA' VETTORIZZATO SU N_SIM) ---
        for t in range(YEARS):
            curr_sr, curr_sl, curr_inf = scenarios[:, t, 0], scenarios[:, t, 1], scenarios[:, t, 2]
            cum_infl *= (1 + curr_inf)
            
            # Scelta parametri crisi
            crisis_mask = is_crisis[:, t]
            l_spread = np.where(crisis_mask, LADDER_STRESS_SPREAD, LADDER_BID_ASK_SPREAD)
            e_spread = np.where(crisis_mask, ETF_STRESS_SPREAD, ETF_BID_ASK_SPREAD)
            def_prob = np.where(crisis_mask, DEFAULT_PROB_CRISIS, DEFAULT_PROB_NORMAL)
            
            # 1. Risparmi (Accumulo)
            flow_in_savings = np.zeros(N_SIM)
            flow_out_map = {} # Tracks spending per strategy
            if mode in ["accumulation", "mixed"]:
                savings = ANNUAL_SAVINGS * cum_infl
                flow_in_savings = savings # Recorded for CAGR calc
                for name in ladder_datas:
                    ladder_datas[name]["cash"] += np.where(~ladder_datas[name]["ruined"], savings, 0)
                EL_cash += np.where(~EL_ruined, savings, 0)
                for name in etf_datas:
                    etf_datas[name]["cash"] += np.where(~etf_datas[name]["ruined"], savings, 0)

            # 2. Defaults
            for name, p in ladder_datas.items():
                m_arr = p["mats_orig"]
                # Il Ladder subisce meno default grazie alla selezione di qualità (Cherry Picking)
                effective_ladder_def_prob = def_prob
                for i in range(len(m_arr)):
                    n_def = np.random.binomial(N_LADDER_ISSUERS_PER_RUNG, effective_ladder_def_prob)
                    loss_sev = 1.0 - np.random.uniform(RECOVERY_MIN, RECOVERY_MAX, N_SIM)
                    loss_frac = (n_def / N_LADDER_ISSUERS_PER_RUNG) * loss_sev
                    p["face"][:, i] *= (1 - loss_frac)
                    p["cost"][:, i] *= (1 - loss_frac)

            etf_shock = 1.0 - (def_prob * (1.0 - (RECOVERY_MIN + RECOVERY_MAX)/2))
            for name in etf_datas: etf_datas[name]["val"] *= etf_shock[:, None]
            EL_val *= etf_shock[:, None]

            # 3. Evoluzione Ladder (Rollover)
            for name, p in ladder_datas.items():
                if np.any(~p["ruined"]):
                    # Cedole
                    p["cash"] += np.where(~p["ruined"], (p["face"] * p["coupon"]).sum(axis=1) * (1 - TAX_RATE_COUPON), 0)
                    p["mats"] -= 1
                    
                    m_arr = p["mats_orig"]
                    max_mat = m_arr.max()
                    
                    for i in range(len(m_arr)):
                        expired = (p["mats"][:, i] <= 0) & (~p["ruined"])
                        if np.any(expired):
                            gain = p["face"][:, i] - p["cost"][:, i]
                            
                            # --- Recupero Minusvalenze (Fiscale Fix 13) ---
                            # Se gain < 0 (Loss), aumento zainetto. Se gain > 0, uso zainetto.
                            loss_mask = (gain < 0)
                            gain_mask = (gain > 0)
                            
                            # Caso Loss: Aggiungo a zainetto (valore assoluto)
                            p["zainetto"][loss_mask & expired] += np.abs(gain[loss_mask & expired])
                            taxable_gain = np.zeros_like(gain)
                            
                            # Caso Gain: Uso zainetto
                            # Gain coperto = min(gain, zainetto)
                            gain_covered = np.minimum(gain, p["zainetto"])
                            # Riduco zainetto della parte usata
                            p["zainetto"][gain_mask & expired] -= gain_covered[gain_mask & expired]
                            
                            taxable_gain[gain_mask & expired] = gain[gain_mask & expired] - gain_covered[gain_mask & expired]
                            
                            tax = np.where(taxable_gain > 0, taxable_gain * TAX_RATE_CG, 0)
                            p["cash"][expired] += (p["face"][expired, i] - tax[expired])
                            p["mats"][expired, i] = max_mat
                            # Applicazione Bonus Selezione (Cherry Picking)
                            p["mats"][expired, i] = max_mat
                            # Removed unused LADDER_SELECTION_PREMIUM
                            new_coupon = get_rate_at_maturity(curr_sr[expired], curr_sl[expired], max_mat)
                            p["coupon"][expired, i] = new_coupon
                            p["face"][expired, i] = 0
                            p["cost"][expired, i] = 0

                    # Reinvestimento
                    is_lazy = "Lazy" in name
                    thresh = LAZY_CASH_THRESHOLD if is_lazy else CASH_THRESHOLD
                    slip = LADDER_SLIPPAGE + (LAZY_SLIPPAGE_PENALTY if is_lazy else 0)
                    
                    # Logica Intra-year Drag
                    if is_lazy:
                        days = lazy_days_to_react[:, t]
                        skip_mask = (days > 200)
                    else:
                        days = np.zeros(N_SIM) # Reazione istantanea (giorno 0)
                        skip_mask = np.zeros(N_SIM, dtype=bool)

                    reinvest_mask = (p["cash"] > (thresh * cum_infl)) & (~p["ruined"]) & (~skip_mask)
                    if np.any(reinvest_mask):
                        idx_new = np.argmax(p["mats"], axis=1)
                        invest = (p["cash"][reinvest_mask] - (thresh * cum_infl)[reinvest_mask]) * CASH_REINVEST_FRACTION
                        buy_p, _ = calculate_trade_impact(1.0, 1, l_spread[reinvest_mask], slip)
                        rows = np.where(reinvest_mask)[0]
                        cols = idx_new[reinvest_mask]
                        
                        # Tasso mercato attuale per i nuovi bond
                        # Fix 20: Use Intra-Year Rates (Micro-oscillations)
                        # Reinvestimento avviene al giorno 'days'.
                        frac_year = np.clip(days / TRADING_DAYS_PER_YEAR, 0.01, 0.99)
                        
                        # Campionamento Brownian Bridge per i Ladder attivi
                        # Nota: curr_sr e curr_sl sono vettori (N_SIM,), reinvest_mask filtra.
                        # Dobbiamo calcolare rates solo per [reinvest_mask]
                        sr_reinv = sample_brownian_bridge(
                            prev_sr[reinvest_mask], curr_sr[reinvest_mask], 
                            frac_year[reinvest_mask], SHORT_RATE_VOL
                        )
                        sl_reinv = sample_brownian_bridge(
                            prev_sl[reinvest_mask], curr_sl[reinvest_mask], 
                            frac_year[reinvest_mask], SLOPE_VOL
                        )
                        
                        rate_at_mat = get_rate_at_maturity(sr_reinv, sl_reinv, max_mat)

                        # --- Penale di Ritardo (Cash Drag) ---
                        # Se compro al giorno D, perdo l'interesse per quei giorni.
                        # Lo simuliamo riducendo il nominale acquistato.
                        # Fix 1: Exponential delay penalty
                        delay_penalty = (1.0 + rate_at_mat) ** (days[reinvest_mask] / TRADING_DAYS_PER_YEAR)
                        
                        
                        # Assert Broadcating Safety (Fix 9)
                        if buy_p.shape != () and len(buy_p.shape) > 0: # se non scalare
                             assert buy_p.shape == (reinvest_mask.sum(),), f"Shape mismatch buy_p: {buy_p.shape}"
                        
                        p["face"][rows, cols] += invest / (buy_p * delay_penalty)
                        p["cost"][rows, cols] += invest
                        p["coupon"][rows, cols] = rate_at_mat 
                        p["cash"][rows] -= invest

            # 4. Evoluzione ETFs
            for name in etf_datas:
                p = etf_datas[name]
                if np.any(~p["ruined"]):
                    for i, d in enumerate(ETF_STRATEGIES[name]):
                        # NOTA IMPLEMENTATIVA (Fix 8): 
                        # La duration 'd' resta costante nel tempo (Rolling).
                        # Questo simula il comportamento reale degli ETF obbligazionari che fanno rebalancing continuo
                        # per mantenere la duration target (es. "7-10 Year" resta sempre 7-10 Year).
                        # Non e' un bond che scade ("pull to par"), ma un portafoglio rolling.
                        r_prev = get_rate_at_maturity(prev_sr, prev_sl, d)
                        r_curr = get_rate_at_maturity(curr_sr, curr_sl, d)
                        dr = r_curr - r_prev
                        # Fix 10: Roll Down Return Coerente
                        # Uso get_rate_at_maturity invece della formula manuale per coerenza con d <= 1
                        y_shorter_rate = get_rate_at_maturity(curr_sr, curr_sl, d - 1 if d > 1 else 0.5)
                        
                        # total_ret = income + price_change + roll_down - costs
                        # roll_down_ret = d * (y_now - y_shorter) * 0.9 -- approx duration * delta_yield
                        # y_now e' r_curr (gia' calcolato)
                        roll_down_ret = d * (r_curr - y_shorter_rate) * ROLL_DOWN_EFFICIENCY
                        
                        total_ret = p["yields"][:, i] - d * dr + roll_down_ret
                        total_ret -= ETF_TER 
                        
                        # Fix 18: Applicazione Costi Occulti (Turnover e Cash Drag)
                        total_ret -= ETF_INTERNAL_TURNOVER_COST
                        # Il Cash Drag riduce il rendimento in proporzione alla quota tenuta in cash
                        # Se ho 0.3% in cash, il rendimento totale si riduce dello 0.3% approx (essendo r_cash ~0)
                        total_ret *= (1.0 - ETF_INTERNAL_CASH_DRAG)
                        
                        p["val"][:, i] = np.maximum(0, p["val"][:, i] * (1 + total_ret)) 
                        # N.B. p["val"] qui cresce per performance. Le units restano fisse.
                        
                        turnover = 1.0 / max(1, d)
                        p["yields"][:, i] = p["yields"][:, i] * (1 - turnover) + r_curr * turnover
                    
                    reinvest_etf = (p["cash"] > (CASH_THRESHOLD * cum_infl)) & (~p["ruined"])
                    if np.any(reinvest_etf):
                        invest = (p["cash"][reinvest_etf] - (CASH_THRESHOLD * cum_infl)[reinvest_etf]) * CASH_REINVEST_FRACTION
                        p["cash"][reinvest_etf] -= invest
                        per_e_invest = invest / len(ETF_STRATEGIES[name])
                        for i in range(len(ETF_STRATEGIES[name])):
                            nav = p["val"][reinvest_etf, i] / (p["units"][reinvest_etf, i] + 1e-9)
                            ask_nav, _ = calculate_trade_impact(nav, 1, e_spread[reinvest_etf], ETF_SLIPPAGE)
                            new_units = per_e_invest / (ask_nav + 1e-9)
                            p["avg_buy"][reinvest_etf, i] = (p["units"][reinvest_etf, i] * p["avg_buy"][reinvest_etf, i] + new_units * ask_nav) / (p["units"][reinvest_etf, i] + new_units + 1e-9)
                            p["units"][reinvest_etf, i] += new_units
                            # Removed dead code line: e["value"] += amount 
                            p["val"][reinvest_etf, i] = p["units"][reinvest_etf, i] * nav

            # ETF Lazy
            if np.any(~EL_ruined):
                for i, d in enumerate(el_durations):
                    r_prev, r_curr = get_rate_at_maturity(prev_sr, prev_sl, d), get_rate_at_maturity(curr_sr, curr_sl, d)
                    dr = r_curr - r_prev
                    y_shorter = get_rate_at_maturity(curr_sr, curr_sl, d - 1 if d > 1 else 0.5)
                    
                    # Fix 10: Roll Down Coerente anche per Lazy
                    roll_down_ret = d * (r_curr - y_shorter) * ROLL_DOWN_EFFICIENCY
                    
                    ret = EL_yields[:, i] - d * dr + roll_down_ret - ETF_TER
                    ret -= ETF_INTERNAL_TURNOVER_COST
                    ret *= (1.0 - ETF_INTERNAL_CASH_DRAG)
                    
                    EL_val[:, i] = np.maximum(0, EL_val[:, i] * (1 + ret))

                    EL_yields[:, i] = EL_yields[:, i] * (1 - (1/d)) + r_curr * (1/d)
                
                # Logica Intra-year Drag ETF
                el_days = lazy_days_to_react[:, t]
                el_skip = (el_days > 200)
                
                reinvest_el = (EL_cash > (LAZY_CASH_THRESHOLD * cum_infl)) & (~EL_ruined) & (~el_skip)
                if np.any(reinvest_el):
                    invest = (EL_cash[reinvest_el] - (LAZY_CASH_THRESHOLD * cum_infl)[reinvest_el]) * CASH_REINVEST_FRACTION
                    EL_cash[reinvest_el] -= invest
                    for i, d in enumerate(el_durations):
                        p_invest = invest / n_el
                        nav = EL_val[reinvest_el, i] / (EL_units[reinvest_el, i] + 1e-9)
                        ask_nav, _ = calculate_trade_impact(nav, 1, e_spread[reinvest_el], ETF_SLIPPAGE + LAZY_SLIPPAGE_PENALTY)
                        
                        # Calcolo rendimento corrente per il penalty
                        # Fix 20: Intra-year rates per ETF Reinvest
                        el_frac = np.clip(el_days / TRADING_DAYS_PER_YEAR, 0.01, 0.99)
                        sr_el = sample_brownian_bridge(prev_sr[reinvest_el], curr_sr[reinvest_el], el_frac[reinvest_el], SHORT_RATE_VOL)
                        sl_el = sample_brownian_bridge(prev_sl[reinvest_el], curr_sl[reinvest_el], el_frac[reinvest_el], SLOPE_VOL)
                        
                        # Usiamo i tassi micro-oscillati per definire yield corrente (e quindi prezzo teorico di penalty)
                        r_curr = get_rate_at_maturity(sr_el, sl_el, d)
                        # Fix 1: Exponential delay penalty
                        delay_penalty = (1.0 + r_curr) ** (el_days[reinvest_el] / TRADING_DAYS_PER_YEAR)
                        
                        # Fix 2: Lazy logic - Capital is reduced/notgrown, Price is ASK.
                        # Capital effective = Invest / delay_penalty
                        invest_effective = p_invest / delay_penalty
                        
                        new_u = invest_effective / (ask_nav + 1e-9)
                        
                        # Avg Buy Price Update: uses real transaction price (ask_nav), not penalised one.
                        EL_avg_buy[reinvest_el, i] = (EL_units[reinvest_el, i] * EL_avg_buy[reinvest_el, i] + new_u * ask_nav) / (EL_units[reinvest_el, i] + new_u + 1e-9)
                        EL_units[reinvest_el, i] += new_u
                        EL_val[reinvest_el, i] = EL_units[reinvest_el, i] * nav

            # 5. Spese
            # Initialize map for this year if not already
            for name in ladder_datas: flow_out_map[name] = np.zeros(N_SIM)
            flow_out_map["ETF_LadderLike (Lazy)"] = np.zeros(N_SIM)
            for name in etf_datas: flow_out_map[name] = np.zeros(N_SIM)

            if mode in ["decumulation", "mixed"]:
                spending_nominal = spend_amounts_base[:, t] * cum_infl
                active_spend = spend_events[:, t]
                
                # Fix 20: Generazione Tassi Spending Intra-Year (Micro-oscillazioni)
                # Calcoliamo i tassi di mercato al momento della spesa per tutte le sim (vettorializzato)
                # Utile per rivalutare Bond/ETF al momento esatto della vendita
                sp_frac = np.clip(spending_days[:, t] / TRADING_DAYS_PER_YEAR, 0.01, 0.99)
                sr_spend = sample_brownian_bridge(prev_sr, curr_sr, sp_frac, SHORT_RATE_VOL)
                sl_spend = sample_brownian_bridge(prev_sl, curr_sl, sp_frac, SLOPE_VOL)
                
                for name, p in ladder_datas.items():
                    is_lazy = "Lazy" in name
                    slip = LADDER_SLIPPAGE + (LAZY_SLIPPAGE_PENALTY if is_lazy else 0)
                    
                    m = active_spend & (~p["ruined"])
                    if np.any(m):
                        amt = spending_nominal[m]
                        
                        # Prelievo da Cash
                        c_take = np.minimum(p["cash"][m], amt)
                        rem = amt - c_take
                        
                        # Track spending (Flow Out) for CAGR
                        flow_out_map[name][m] += amt
                        
                        # Prelievo (Vendita) da Bond per la parte rimanente (rem > 0)
                        # Fix 11 & 12: Vendita vera con Costi e Tasse, e riduzione Cost Basis
                        needs_sell = (rem > 1.0) # soglia minima
                        if np.any(needs_sell):
                            # Indices relativi al sotto-insieme m che ha bisogno di vendere
                            sub_idx = np.where(needs_sell)[0] 
                            
                            # Logica vettoriale complessa: iteriamo per semplicita' o approx vettoriale?
                            # Approx vettoriale: vendiamo pro-rata da tutti i bond
                            # MTM totale dei bond
                            # Fix Broadcasting: curr_sr e curr_sl hanno shape (N_SIM,) ma p["mats"][m] ha shape (M, cols)
                            # Dobbiamo filtrare sr/sl con m
                            # Fix 20: Use Intra-Year Spenidng Rates for MTM logic
                            # Prezzo al momento della vendita (Spending Day)
                            sr_s = sr_spend[m]
                            sl_s = sl_spend[m]
                            
                            r_b = get_rate_at_maturity(sr_s, sl_s, p["mats"][m]) # shape (M, cols)
                            prices = get_bond_price(1.0, p["coupon"][m], p["mats"][m], r_b)
                            face_vals = p["face"][m]
                            mtm_vals = face_vals * prices
                            tot_mtm = mtm_vals.sum(axis=1) # shape (M,)
                            
                            # Check rovina
                            p["ruined"][m] |= (rem > (tot_mtm * RUIN_THRESHOLD)) # buffer 5% per costi/tasse
                            
                            # Quota da vendere (Target Netto = rem)
                            # Stimiamo costi totali (Spread + Tax) in modo dinamico (Fix 9)
                            # Calcolo gain medio pesato del sotto-portafoglio
                            # Gain = MTM - Cost
                            curr_mtm = tot_mtm # shape (S,)
                            curr_cost = p["cost"][m].sum(axis=1) # shape (S,)
                            
                            # Gain rate per unita' di valore venduto
                            avg_gain_pct = (curr_mtm - curr_cost) / (curr_mtm + 1e-9)
                            
                            # Se gain > 0, si paga tax. Se gain < 0, niente tax (e si accumula zainetto, ignorato in questa stima costi)
                            tax_impact_rate = np.maximum(0, avg_gain_pct) * TAX_RATE_CG
                            
                            # Gross Factor: quanto devo vendere in piu' per coprire spread e tasse?
                            # Net = Gross * (1 - spread/2 - tax_impact)
                            # Gross = Net / (1 - spread/2 - tax_impact)
                            # Approx lineare per piccole %: Gross ~ Net * (1 + spread/2 + tax_impact)
                            
                            gross_sell_factor = 1.0 + (LADDER_BID_ASK_SPREAD / 2.0) + tax_impact_rate + 0.01 # 1% extra buffer
                            
                            fraction_to_sell = (rem / (tot_mtm + 1e-9)) * gross_sell_factor
                            fraction_to_sell = np.clip(fraction_to_sell, 0, 1.0)
                            
                            # Applichiamo riduzione:
                            # Fix 12: Riduco Face E Cost Basis
                            multiplier = 1.0 - fraction_to_sell
                            
                            # Applicazione
                            # Attenzione: m seleziona righe. sub_idx filtra ulteriormente.
                            # Usiamo indices globali per chiarezza
                            global_idx = np.where(m)[0]
                            # Se needs_sell e' un array bool su m, allora global_idx[needs_sell] sono quelli che vendono
                            
                            sellers_idx = global_idx[needs_sell]
                            mult_vals = multiplier[needs_sell] # shape (S,)
                            
                            p["face"][sellers_idx] *= mult_vals[:, None]
                            p["cost"][sellers_idx] *= mult_vals[:, None] # Fix 12
                            
                            # Aggiornamento Zainetto nelle Spese (Fix 5)
                            # Calcolo Gain/Loss Realizzato
                            # MTM Unitario Sold = curr_mtm * fraction_to_sell
                            # Cost Unitario Sold = curr_cost * fraction_to_sell
                            mtm_sold = curr_mtm[needs_sell] * fraction_to_sell[needs_sell]
                            cost_sold = curr_cost[needs_sell] * fraction_to_sell[needs_sell]
                            realized_gain = mtm_sold - cost_sold
                            
                            # Loss: Incremento zainetto
                            loss_mask = realized_gain < 0
                            # gain_mask = realized_gain >= 0 # Per i gain, si paga tassa nel gross factor, zainetto potrebbe essere usato ma approx
                            
                            # Attenzione: zainetto ha shape [N_SIM], qui stiamo lavorando su sub-idx di m
                            # sellers_idx sono gli indici globali delle sim che vendono
                            loss_indices = sellers_idx[loss_mask]
                            if len(loss_indices) > 0:
                                p["zainetto"][loss_indices] += np.abs(realized_gain[loss_mask])

                m_el = active_spend & (~EL_ruined)
                if np.any(m_el):
                    # Stessa logica per ETF Lazy
                    amt = spending_nominal[m_el]
                    c_take = np.minimum(EL_cash[m_el], amt)
                    EL_cash[m_el] -= c_take
                    rem = amt - c_take
                    
                    flow_out_map["ETF_LadderLike (Lazy)"][m_el] += amt
                    
                    camt = amt
                    
                    # Fix 20: ETF Intra-Year Valuation for Spending
                    # Stimiamo il NAV al momento della spesa
                    # NAV_spend ~ NAV_end * (1 + duration * (Rate_End - Rate_Spend)) roughly
                    # Piu preciso: NAV_spend = Unit * Price(Spend).
                    # Price(Spend) approx: Price(End) + Price(End) * Duration * (Rate_End - Rate_Spend)
                    
                    # Calcoliamo Rate End e Rate Spend per i vari ETF duration
                    # EL_yields[:, i] contiene yield corrente? No, contiene yield storico del portafoglio.
                    # Usiamo yield di mercato per Delta Pricing.
                    
                    current_val_end = EL_val[m_el].copy() # Valore a fine anno
                    # Lo aggiustiamo per la differenza tassi
                    
                    # sr_spend[m_el], sl_spend[m_el] vs curr_sr[m_el], curr_sl[m_el]
                    # Delta Rate = Rate_Spend - Rate_End (Attenzione ai segni)
                    # Se Rate Spend < Rate End -> Prezzo Spend > Prezzo End.
                    # Change = - Duration * (Rate_Spend - Rate_End)
                    
                    v_tot_est = 0.0
                    adjusted_navs = [] # Store per uso dopo
                    
                    for i, d in enumerate(el_durations):
                         r_end = get_rate_at_maturity(curr_sr[m_el], curr_sl[m_el], d)
                         r_spd = get_rate_at_maturity(sr_spend[m_el], sl_spend[m_el], d)
                         dr = r_spd - r_end
                         
                         # Adj Value
                         val_end = EL_val[m_el, i]
                         val_spd = val_end * (1.0 - d * dr) # Approx Price sensitivity
                         v_tot_est += val_spd
                         adjusted_navs.append(val_spd)
                    
                    EL_ruined[m_el] |= (rem > (v_tot_est * RUIN_THRESHOLD))
                    
                    # Fix 12: Vendita ETF Lazy
                    # Fix 6: Dynamic Tax Calc also for ETF
                    
                    # Calcolo Avg Buy Price e Gain Medio Pesato
                    # EL_units shape: (N_SIM_EL, N_DUR)
                    # EL_avg_buy shape: (N_SIM_EL, N_DUR)
                    # Abbiamo bisogno del costo totale e valore totale
                    units_sub = EL_units[m_el]
                    buy_price_sub = EL_avg_buy[m_el]
                    
                    current_cost = (units_sub * buy_price_sub).sum(axis=1)
                    current_cost = (units_sub * buy_price_sub).sum(axis=1)
                    current_val = v_tot_est # Usiamo Valore Stimato Intra-Year
                    
                    avg_gain_pct = (current_val - current_cost) / (current_val + 1e-9)
                    tax_impact_rate = np.maximum(0, avg_gain_pct) * TAX_RATE_CG
                    
                    gross_factor = 1.0 + (ETF_BID_ASK_SPREAD / 2.0) + tax_impact_rate + 0.01

                    gross_factor = 1.0 + (ETF_BID_ASK_SPREAD / 2.0) + tax_impact_rate + 0.01

                    fraction = (rem / (v_tot_est + 1e-9)) * gross_factor
                    fraction = np.clip(fraction, 0, 1.0)
                    multiplier = 1.0 - fraction
                    
                    EL_units[m_el] *= multiplier[:, None]
                    # EL_val deve essere aggiornato? 
                    # EL_val rappresenta il valore a FINE anno. 
                    # Se vendo il 10% delle unit a meta anno, a fine anno avro' il 10% in meno di units
                    # e quindi il 10% in meno di valore (rispetto al controfattuale).
                    # Quindi ridurre EL_val pro-quota è corretto.
                    EL_val[m_el] *= multiplier[:, None]
                    # Avg Buy Price (unitario) NON cambia se vendo pro-quota


                for name in etf_datas:
                    p = etf_datas[name]
                    m = active_spend & (~p["ruined"])
                    if np.any(m):
                        amt = spending_nominal[m]
                        c_take = np.minimum(p["cash"][m], amt)
                        p["cash"][m] -= c_take
                        rem = amt - c_take
                        rem = amt - c_take
                        
                        flow_out_map[name][m] += amt
                        
                        # Fix 20: Standard ETF Intra-Year (copia logica Lazy ETF)
                        durations = ETF_STRATEGIES[name]
                        v_tot_est = 0.0
                        for i, d in enumerate(durations):
                             r_end = get_rate_at_maturity(curr_sr[m], curr_sl[m], d)
                             r_spd = get_rate_at_maturity(sr_spend[m], sl_spend[m], d)
                             dr = r_spd - r_end
                             v_tot_est += p["val"][m, i] * (1.0 - d * dr)

                        p["ruined"][m] |= (rem > (v_tot_est * 0.9))
                        multiplier = np.clip(1 - (rem / (v_tot_est + 1e-9)), 0, 1)
                        p["val"][m] *= multiplier[:, None]


            # Storico
            for name, p in ladder_datas.items():
                r = get_rate_at_maturity(curr_sr, curr_sl, p["mats"])
                v = p["cash"] + get_bond_price(p["face"], p["coupon"], p["mats"], r).sum(axis=1)
                v[p["ruined"]] = 0
                p["hist"].append(v)
                
            for name in etf_datas:
                v_e = etf_datas[name]["cash"] + etf_datas[name]["val"].sum(axis=1)
                v_e[etf_datas[name]["ruined"]] = 0
                etf_datas[name]["hist"].append(v_e)
                
            v_el = EL_cash + EL_val.sum(axis=1)
            v_el[EL_ruined] = 0
            EL_hist.append(v_el)

            prev_sr, prev_sl = curr_sr.copy(), curr_sl.copy()

            # --- CAGR Calculation Step ---
            # 1. Savings (Flow In) - defined earlier as 'savings' (vector) or 0
            # 2. Spending (Flow Out) - stored in flow_out_map
            # 3. Wealth End (w_curr) - from hist[-1]
            
            # Ladder
            for name, p in ladder_datas.items():
                w_curr = p["hist"][-1]
                inflow = flow_in_savings
                outflow = flow_out_map.get(name, np.zeros(N_SIM))
                
                base = p["prev_wealth"] + inflow
                final_gross = w_curr + outflow
                
                # Yield
                # Se base è zero (già rovinato o vuoto), rendimento è 0 (o invariato 1.0)
                # Se w_curr è 0 e base > 0 -> si è rovinato ora. yield=0.
                y_factor = np.divide(final_gross, base, out=np.zeros_like(base), where=base>1e-4)
                # Handle cases where base is small (avoid infinity)
                
                p["cum_yield"] *= y_factor
                p["prev_wealth"] = w_curr
            
            # ETFs
            for name, p in etf_datas.items():
                w_curr = p["hist"][-1]
                inflow = flow_in_savings
                outflow = flow_out_map.get(name, np.zeros(N_SIM))
                
                base = p["prev_wealth"] + inflow
                final_gross = w_curr + outflow
                
                y_factor = np.divide(final_gross, base, out=np.zeros_like(base), where=base>1e-4)
                p["cum_yield"] *= y_factor
                p["prev_wealth"] = w_curr
                
            # EL
            w_curr_el = EL_hist[-1]
            inflow_el = flow_in_savings
            outflow_el = flow_out_map["ETF_LadderLike (Lazy)"]
            
            base_el = EL_prev_wealth + inflow_el
            final_gross_el = w_curr_el + outflow_el
            
            y_factor_el = np.divide(final_gross_el, base_el, out=np.zeros_like(base_el), where=base_el>1e-4)
            EL_cum_yield *= y_factor_el
            EL_prev_wealth = w_curr_el

        final_values, final_values_real, ruin_counts, paths, final_cagr = {}, {}, {}, {}, {}
        
        for name, p in ladder_datas.items():
            h = np.array(p["hist"]).T
            final_values[name] = h[:, -1]
            final_values_real[name] = h[:, -1] / cum_infl
            ruin_counts[name] = p["ruined"]
            paths[name] = h
            final_cagr[name] = (p["cum_yield"] ** (1/YEARS)) - 1.0

        for name in etf_datas:
            h = np.array(etf_datas[name]["hist"]).T
            final_values[name] = h[:, -1]
            final_values_real[name] = h[:, -1] / cum_infl
            ruin_counts[name] = etf_datas[name]["ruined"]
            paths[name] = h
            final_cagr[name] = (etf_datas[name]["cum_yield"] ** (1/YEARS)) - 1.0

        h_el = np.array(EL_hist).T
        final_values["ETF_LadderLike (Lazy)"] = h_el[:, -1]
        final_values_real["ETF_LadderLike (Lazy)"] = h_el[:, -1] / cum_infl
        ruin_counts["ETF_LadderLike (Lazy)"] = EL_ruined
        paths["ETF_LadderLike (Lazy)"] = h_el
        final_cagr["ETF_LadderLike (Lazy)"] = (EL_cum_yield ** (1/YEARS)) - 1.0

        return {
            "final_values": final_values, 
            "final_values_real": final_values_real, 
            "ruin_counts": ruin_counts, 
            "paths": paths,
            "final_cagr": final_cagr 
        }

    except Exception as e:
        import traceback
        print(f"ERRORE NELLA SIMULAZIONE {mode.upper()}:")
        traceback.print_exc()
        raise e


def run_full_analysis():
    results = {}
    for mode in SIMULATION_MODES_TO_RUN:
        print(f"Avvio simulazione: {mode.upper()}...")
        results[mode] = run_simulation(mode)
    return results


def summarize_and_plot_combined(full_results):
    modes = list(full_results.keys())
    
    # --- Tabella Unica ---
    summary_rows = []
    
    for mode in modes:
        res = full_results[mode]
        for strat in res["final_values"].keys():
            fv_nom = np.array(res["final_values"][strat])
            fv_real = np.array(res["final_values_real"][strat])
            ruin = np.array(res["ruin_counts"][strat])
            
            # Retrieve CAGR (already calculated in data gathering)
            cagrs = np.array(res["final_cagr"][strat]) * 100.0 # Convert to %
            
            # CAGR Statistics
            mean_cagr = cagrs.mean()
            median_cagr = np.median(cagrs)
            min_cagr = cagrs.min()
            std_cagr = cagrs.std()
            
            # Efficiency CAGR (Return / Risk)
            # Sharpe-like ratio: Mean CAGR / Std Dev CAGR
            eff_cagr = mean_cagr / (std_cagr + 1e-9) if std_cagr > 1e-9 else 0.0

            # Tail Risk on Real Wealth (Min Real) matches "Safety" better than CAGR Min (which is rate)
            # But specific "Min CAGR" helps understand worst-case compounding.
            
            row = {
                "Modalità": mode,
                "Strategia": strat,
                "CAGR Mean %": mean_cagr,
                "CAGR Med %": median_cagr,
                "CAGR Min %": min_cagr,
                "Min Real (Eu)": fv_real.min(), # Keep absolute safety check
                "Rovina %": ruin.mean() * 100,
                "Eff.(CAGR)": eff_cagr
            }
            summary_rows.append(row)
            
    df = pd.DataFrame(summary_rows)
    # Formattazione per la stampa
    pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x) if x > 100 else '{:.1f}'.format(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Ordiniamo per Modalità poi per Min Real (o CAGR Min)
    df = df.sort_values(by=["Modalità", "Min Real (Eu)"], ascending=[True, False])
    
    print("\n=== RISULTATI CONSOLIDATI (CAGR & REAL) ===")
    print(df.to_string(index=False))
    
    # --- Analisi Perdita di Efficienza (Pigrizia) ---
    print("\n=== ANALISI COSTO DELLA PIGRIZIA (LAZY PENALTY) ===")
    lazy_analysis = []
    
    # Identificazione dinamica delle coppie Strategia - Strategia (Lazy)
    all_strats = list(full_results[modes[0]]["final_values"].keys())
    comparisons = []
    for s in all_strats:
        if "(Lazy)" not in s and f"{s} (Lazy)" in all_strats:
            comparisons.append((s, f"{s} (Lazy)"))
    
    for mode in modes:
        res = full_results[mode]
        for opt, lazy in comparisons:
            if opt in res["final_values"] and lazy in res["final_values"]:
                wealth_opt = np.mean(res["final_values_real"][opt])
                wealth_lazy = np.mean(res["final_values_real"][lazy])
                
                # Retrieve CAGR stats for efficiency
                cagr_opt = np.array(res["final_cagr"][opt]) * 100.0
                cagr_lazy = np.array(res["final_cagr"][lazy]) * 100.0
                
                eff_opt = cagr_opt.mean() / (cagr_opt.std() + 1e-9)
                eff_lazy = cagr_lazy.mean() / (cagr_lazy.std() + 1e-9)
                
                # Perdita di capitale finale (Reale) - Resta utile per capire "quanti soldi persi"
                wealth_loss_pct = (1 - (wealth_lazy / (wealth_opt + 1e-9))) * 100
                
                # Perdita di Efficienza (su misura CAGR)
                eff_loss_pct = (1 - (eff_lazy / (eff_opt + 1e-9))) * 100
                
                lazy_analysis.append({
                    "Modalità": mode,
                    "Confronto": f"{opt} vs Lazy",
                    "Perdita Capitale (%)": wealth_loss_pct,
                    "Riduzione Efficienza (%)": eff_loss_pct
                })
    
    df_lazy = pd.DataFrame(lazy_analysis)
    pd.options.display.float_format = '{:.2f}'.format
    print(df_lazy.to_string(index=False))
    
    df.to_csv('results.csv', index=False)
    
    # --- Grafici Andamento Temporale (N righe, 2 colonne) ---
    n_rows = len(modes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    fig.suptitle("Confronto Strategie: Ladder (Black) vs ETFs (Colored)", fontsize=16, y=0.99)
    
    # Gestione caso single row (axes non è 2D array)
    if n_rows == 1:
        axes = np.array([axes])
    
    t = np.arange(1, YEARS + 1)
    
    # --- Color Palette & Styles ---
    # Definiamo uno stile fisso per ogni strategia per coerenza tra i grafici
    # --- Color Palette & Styles ---
    STRATEGY_STYLES = {
        "Ladder": {"color": "#e31a1c", "lw": 2.5, "ls": "-", "z": 10},       # Rosso
        "Ladder (Lazy)": {"color": "#000000", "lw": 2.5, "ls": "-", "z": 15}, # Nero (Evidenziato)
        "Ladder (Short 2-4y)": {"color": "#6a3d9a", "lw": 2.5, "ls": "-", "z": 9},
        "Ladder (Short 2-4y) (Lazy)": {"color": "#6a3d9a", "lw": 1.5, "ls": "--", "z": 8},
        "Ladder (long 6-10y)": {"color": "#33a02c", "lw": 2.5, "ls": "-", "z": 9},
        "Ladder (long 6-10y) (Lazy)": {"color": "#33a02c", "lw": 1.5, "ls": "--", "z": 8},
        "ETF_LadderLike": {"color": "#1f78b4", "lw": 1.5, "ls": "-", "z": 8},
        "ETF_LadderLike (Lazy)": {"color": "#1f78b4", "lw": 1.5, "ls": "--", "z": 7},
        "ETF_Short (2y)": {"color": "#33a02c", "lw": 1.5, "ls": "--", "z": 5},
        "ETF_Short (4y)": {"color": "#fb9a99", "lw": 1.5, "ls": "--", "z": 5},
        "ETF_Medium (6y)": {"color": "#cab2d6", "lw": 1.5, "ls": "--", "z": 5},
        "ETF_Long (10y)": {"color": "#6a3d9a", "lw": 1.5, "ls": "--", "z": 5},
        "ETF_Barbell (2+10)": {"color": "#b15928", "lw": 1.5, "ls": "--", "z": 5}
    }

    for i, mode in enumerate(modes):
        res = full_results[mode]
        paths_dict = res["paths"]
        res = full_results[mode]
        paths_dict = res["paths"]
        # final_vals_dict = res["final_values"] # Unused for plotting now
        
        # 1. Plot Andamento Mediano (Colonna SX)
        ax_time = axes[i, 0]
        for strat, paths in paths_dict.items():
            arr_paths = np.array(paths)
            p50 = np.percentile(arr_paths, 50, axis=0)
            
            style = STRATEGY_STYLES.get(strat, {"color": None, "lw": 1.5, "ls": "-", "z": 5})
            
            ax_time.plot(t, p50, label=strat, 
                         color=style["color"], 
                         linestyle=style["ls"], 
                         linewidth=style["lw"], 
                         zorder=style["z"],
                         alpha=1.0)

            
        ax_time.set_title(f"Andamento Mediano ({mode.capitalize()})", fontsize=12, fontweight='bold')
        ax_time.set_ylabel("Euro")
        ax_time.grid(True, alpha=0.3)
        if i == 0: # Legend solo in alto
             ax_time.legend(fontsize='small', loc='upper left')

        # 2. Plot Distribuzione CAGR (Colonna DX)
        # Sostituiamo il grafico Final Value con CAGR
        ax_dist = axes[i, 1]
        cagr_dict = res["final_cagr"]
        
        for strat, vals in cagr_dict.items():
            vals = np.array(vals) * 100 # Convert to percentage
            # Filter outliers for visualization (-100% is ruin)
            # User Request: Cut below -2
            vals_clean = vals[(vals > -2.0) & (vals < 15.0)]
            
            if len(vals_clean) < 2: continue

            style = STRATEGY_STYLES.get(strat, {"color": None, "lw": 1.5, "ls": "-", "z": 5})
            
            if HAS_SCIPY:
                try:
                    density = gaussian_kde(vals_clean)
                    # Custom range centered on data
                    x_min, x_max = vals_clean.min(), vals_clean.max()
                    x_range = np.linspace(x_min, x_max, 200)
                    y_range = density(x_range)
                    
                    ax_dist.plot(x_range, y_range, label=strat, 
                                color=style["color"],
                                linestyle=style["ls"],
                                linewidth=style["lw"], 
                                zorder=style["z"])
                    ax_dist.fill_between(x_range, y_range, alpha=0.1, color=style["color"], zorder=style["z"]-1)
                except Exception:
                     # Fallback if singular matrix
                     ax_dist.hist(vals_clean, bins=50, density=True, histtype='step', 
                             label=strat, color=style["color"], 
                             linewidth=style["lw"], zorder=style["z"])
            else:
                ax_dist.hist(vals_clean, bins=50, density=True, histtype='step', 
                             label=strat, color=style["color"], 
                             linewidth=style["lw"], zorder=style["z"])
            
        ax_dist.set_title(f"Distribuzione Rendimento Annuale Composto (CAGR) % - ({mode.capitalize()})", fontsize=12, fontweight='bold')
        
        # Center view around median of all strategies (approx)
        # Assuming most strategies are around 1-4%, so -2 to 8 is a safe 10% range.
        ax_dist.set_xlim(left=-2.0, right=8.0) 
        
        # Rimuovo label asse X eccetto ultima riga per pulizia
        if i == n_rows - 1:
            ax_time.set_xlabel("Anno")
            ax_dist.set_xlabel("CAGR (%)")
        
        ax_dist.grid(True, alpha=0.3)
        # ax_dist.legend(fontsize='small') 

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3) # Spazio extra per titolo e tra righe
    
    # Salvataggio su file prima della visualizzazione
    #plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    #print("\nGrafico salvato come 'simulation_results.png'")
    
    plt.show()


# ==========================
# Main Execution: Run All Scenarios
# ==========================

if __name__ == "__main__":
    SCENARIOS_TO_TEST =  ["EU_GOV"] #list(SCENARIOS.keys()) # ["EU_GOV", "EU_CORP", "GLOBAL_GOV", "GLOBAL_CORP"]
    
    # Per semplicità di visualizzazione e rapidità, eseguiamo solo EU_GOV e EU_CORP come esempi principali nel post reddit
    # Ma il codice supporta tutto.
    # SCENARIOS_TO_TEST = ["EU_GOV", "EU_CORP"] 
    
    all_experiments = {}

    for scen in SCENARIOS_TO_TEST:
        print(f"\n\n{'='*50}")
        print(f"STARTING EXPERIMENT: {scen}")
        print(f"{'='*50}")
        
        # 1. Update Global Parameters based on Scenario
        # Questo è un hack necessario perché il codice usa variabili globali.
        # Riassegnamo le globali leggendo da SCENARIOS[scen]
        p = SCENARIOS[scen]
        
        # GLOBAL VARIABLES OVERWRITE
        globals()["SHORT_RATE_MEAN"] = p["SHORT_RATE_MEAN"]
        globals()["SHORT_RATE_VOL"] = p["SHORT_RATE_VOL"]
        globals()["SLOPE_MEAN"] = p["SLOPE_MEAN"]
        globals()["SLOPE_VOL"] = p["SLOPE_VOL"]
        globals()["DEFAULT_PROB_NORMAL"] = p["DEFAULT_PROB_NORMAL"]
        globals()["DEFAULT_PROB_CRISIS"] = p["DEFAULT_PROB_CRISIS"]
        globals()["RECOVERY_MIN"] = p["RECOVERY_MIN"]
        globals()["RECOVERY_MAX"] = p["RECOVERY_MAX"]
        globals()["CRISIS_PROB"] = p["CRISIS_PROB"]
        globals()["TAX_RATE_COUPON"] = p["TAX_COUPON"]
        globals()["TAX_RATE_CG"] = p["TAX_CG"]
        globals()["LADDER_BID_ASK_SPREAD"] = p["LADDER_SPREAD"]
        globals()["LADDER_SLIPPAGE"] = p["LADDER_SLIPPAGE"]
        globals()["LADDER_STRESS_SPREAD"] = p["LADDER_STRESS_SPREAD"]
        globals()["N_LADDER_ISSUERS_PER_RUNG"] = p["N_LADDER_ISSUERS"]
        globals()["ETF_TER"] = p["ETF_TER"]
        globals()["ETF_BID_ASK_SPREAD"] = p["ETF_SPREAD"]
        globals()["ETF_SLIPPAGE"] = p["ETF_SLIPPAGE"]
        globals()["ETF_STRESS_SPREAD"] = p["ETF_STRESS_SPREAD"]
        globals()["N_ETF_ISSUERS_TOTAL"] = p["N_ETF_ISSUERS"]

        # 2. Run Analysis for this scenario
        res = run_full_analysis()
        all_experiments[scen] = res
        
        # 3. Print Summary for this scenario
        print(f"\n--- RISULTATI SCENARIO: {scen} ---")
        summarize_and_plot_combined(res)
        
        # Save CSV specific for this scenario
        # summarize_and_plot_combined already saves 'results.csv', we rename it
        import os
        if os.path.exists(f'results_{scen}.csv'):
            os.remove(f'results_{scen}.csv')
        if os.path.exists('results.csv'):
             os.rename('results.csv', f'results_{scen}.csv')
    
    print("\n\nALL EXPERIMENTS COMPLETED.")

