import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    het_arch
)

# ___________________________________ Préparation des données ___________________________________ 


# Chargement des séries de prix quotidiennes
df_cvx = pd.read_csv('DATA/Chevron Stock Price History.csv', index_col='Date', parse_dates=True, thousands=',')[['Price']].rename(columns={'Price': 'CVX'})
df_vix = pd.read_csv('DATA/CBOE Volatility Index Historical Data.csv', index_col='Date', parse_dates=True, thousands=',')[['Price']].rename(columns={'Price': 'VIX'})
df_sp500 = pd.read_csv('DATA/S&P 500 Historical Data.csv', index_col='Date', parse_dates=True, thousands=',')[['Price']].rename(columns={'Price': 'SP500'})
df_gold = pd.read_csv('DATA/XAU_USD Historical Data.csv', index_col='Date', parse_dates=True, thousands=',')[['Price']].rename(columns={'Price': 'Gold'})
df_oil = pd.read_excel('DATA/DCOILBRENTEU.xlsx', sheet_name='Daily', index_col='observation_date', parse_dates=True)[['DCOILBRENTEU']].rename(columns={'DCOILBRENTEU': 'Oil'})
df_us10y = pd.read_excel('DATA/T10Y2Y.xlsx', sheet_name='Daily', index_col='observation_date', parse_dates=True)[['T10Y2Y']].rename(columns={'T10Y2Y': 'UST10'})

# Chargement des séries macroéconomiques (périodicités trimestrielle et mensuelle)
df_pib = pd.read_excel('DATA/PIB.xlsx', sheet_name='Quarterly', index_col='observation_date', parse_dates=True)[['GDP']].rename(columns={'GDP': 'PIB'})
df_cpi = pd.read_excel('DATA/CPIAUCSL.xlsx', sheet_name='Monthly', index_col='observation_date', parse_dates=True)[['CPIAUCSL']].rename(columns={'CPIAUCSL': 'CPI'})

# Traitement des valeurs manquantes ponctuelles de l'inflation par interpolation linéaire
df_cpi = df_cpi.interpolate()

# Alignement des données de marché quotidiennes (calendrier boursier commun)
dfs_quotidiens = [df_cvx, df_gold, df_oil, df_sp500, df_us10y, df_vix]
df_global_quotidien = pd.concat(dfs_quotidiens, axis=1, join='inner').drop_duplicates().dropna()

# Ré-échantillonnage trimestriel (Quarter Start pour s'aligner sur les publications FRED)
df_trimestriel = df_global_quotidien.resample('QS').mean()
df_trimestriel_cpi = df_cpi.resample('QS').mean()

# Fusion finale de l'ensemble des variables macro-financières
df_final_trimestre = pd.concat([df_trimestriel, df_pib, df_trimestriel_cpi], axis=1, join='inner').dropna()

print("\n--- Statistiques descriptives de l'échantillon brut ---")
print(df_final_trimestre.describe())

# Visualisation des distributions
df_final_trimestre.plot(kind='box', subplots=True, layout=(2, 4), figsize=(15, 8))
plt.suptitle("Détection des valeurs atypiques (Boxplots)", fontsize=14)
plt.tight_layout()
plt.show()

# Export de la base consolidée
df_final_trimestre.to_excel('GRUPO_06_Base_Propre.xlsx')


# ___________________________________ Transformations économétriques  ___________________________________ 

# Déflation des séries nominales par l'indice CPI (conversion en termes réels)
variables_a_deflater = ['CVX', 'Gold', 'Oil', 'SP500', 'PIB']
for col in variables_a_deflater:
    df_final_trimestre[col] = (df_final_trimestre[col] / df_final_trimestre['CPI']) * 100

df_final_trimestre = df_final_trimestre.drop(columns=['CPI'])

# Test de stationnarité : Dickey-Fuller Augmenté (ADF) sur séries en niveaux
print("\n--- Tests de stationnarité ADF (Séries en niveaux) ---")
for col in df_final_trimestre.columns:
    p_value = adfuller(df_final_trimestre[col])[1]
    statut = "Stationnaire" if p_value < 0.05 else "Non-Stationnaire"
    print(f"{col:<10} | p-value: {p_value:.4f} | {statut}")

# Transformation en rendements (variations relatives) pour stationnariser les séries
# Note : Le VIX étant déjà un indice de volatilité stationnaire en niveau, il est conservé tel quel
df_stationnaire = df_final_trimestre.copy()
for col in df_stationnaire.columns:
    if col != "VIX":
        df_stationnaire[col] = df_stationnaire[col].pct_change()

df_stationnaire = df_stationnaire.dropna()



# ___________________________________ Test de saisonnalité déterministe  ___________________________________ 

df_stationnaire['Trimestre'] = df_stationnaire.index.quarter
dummies = pd.get_dummies(df_stationnaire['Trimestre'], prefix='Q', drop_first=True).astype(int)
X_saisonnier = sm.add_constant(dummies)

print("\n--- Test F de saisonnalité déterministe ---")
for col in [c for c in df_stationnaire.columns if c != 'Trimestre']:
    f_pvalue = sm.OLS(df_stationnaire[col], X_saisonnier).fit().f_pvalue
    effet = "Saisonnalité détectée" if f_pvalue < 0.05 else "Pas de saisonnalité"
    print(f"{col:<10} | p-value (Test F): {f_pvalue:.4f} | {effet}")

df_stationnaire = df_stationnaire.drop(columns=['Trimestre'])


# ___________________________________ Estimation du modèle de régression multiple  ___________________________________ 

Y = df_stationnaire['CVX']
X = sm.add_constant(df_stationnaire.drop(columns=['CVX']))

# Estimation OLS avec correction robuste de White (HC3) contre l'hétéroscédasticité
modele_final = sm.OLS(Y, X).fit(cov_type='HC3')
print("\n" + "=" * 78)
print("RÉSULTATS DE LA RÉGRESSION OLS (AVEC ERREURS STANDARDS ROBUSTES HC3)")
print("=" * 78)
print(modele_final.summary())


# ___________________________________ Diagnostic des résultats  ___________________________________ 


# A. Multicolinéarité (Corrélation & VIF)
print("\n--- Matrice de Corrélation des Explicatives ---")
print(X.drop(columns=['const']).corr().round(3))

vif_data = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\n--- Facteurs d'Inflation de la Variance (VIF) ---")
print(vif_data.round(2))

# B. Autocorrélation des résidus (Breusch-Godfrey)
bg_pvalue = acorr_breusch_godfrey(modele_final, nlags=4)[1]
print(f"\nTest de Breusch-Godfrey (Autocorrélation) : p-value = {bg_pvalue:.4f}")
print("-> Présence d'autocorrélation" if bg_pvalue < 0.05 else "-> Absence d'autocorrélation (H0 acceptée)")

# C. Hétéroscédasticité (Breusch-Pagan & White)
bp_pvalue = het_breuschpagan(modele_final.resid, modele_final.model.exog)[1]
white_pvalue = het_white(modele_final.resid, modele_final.model.exog)[1]
print(f"Test de Breusch-Pagan (Hétéroscédasticité)  : p-value = {bp_pvalue:.4f}")
print(f"Test de White (Non-linéarités)            : p-value = {white_pvalue:.4f}")

# D. Hétéroscédasticité conditionnelle autorégressive (ARCH)
arch_pvalue = het_arch(modele_final.resid, nlags=4)[1]
print(f"Test ARCH-LM (Clusters de volatilité)     : p-value = {arch_pvalue:.4f}")


# ___________________________________ Visualisation du modèle  ___________________________________ 

plt.figure(figsize=(12, 6))
plt.plot(df_stationnaire.index, Y, label='Rendement réel CVX', color='#1f77b4', linewidth=1.5)
plt.plot(df_stationnaire.index, modele_final.fittedvalues, label='Valeurs ajustées (Modèle OLS)', color='#d62728', linestyle='--', linewidth=1.5)
plt.title("Chevron (CVX) : Rendements Réels vs Prédictions du Modèle", fontsize=13, fontweight='bold')
plt.xlabel("Date", fontsize=11)
plt.ylabel("Variation Trimestrielle", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('resultat_modele_cvx.png', dpi=300)
plt.show()










