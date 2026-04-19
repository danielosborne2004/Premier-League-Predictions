# Premier League Season Simulation

This project estimates the probability of each Premier League team finishing in every possible league position by the end of the current season. It combines an XGBoost expected-goals model with a Poisson match simulator run across thousands of season simulations.

---

## Methodology

The full methodology is documented in **`PremierLeague_Model.ipynb`** — open this file on github for a complete walkthrough covering:

- Target vector construction (deriving expected goals from betting market odds)
- Feature engineering (rolling match stats, attack/defence strength, enrichment flags)
- XGBoost model training and optimisation
- Season simulation framework and output visualisations

This is the best starting point for understanding how the model works.

---

## Using the Model

### Step 1 — Update the data

Open and run **`UpdateData.ipynb`**. This will:

1. Download the latest match results from football-data.co.uk
2. Download the latest fixture schedule
3. Reconcile any manually patched results
4. Display any fixtures that are missing from the dataset

Run this notebook before every simulation to ensure the model uses current data.

### Step 2 — Run a simulation

Two simulation notebooks are available:

#### `RunLeagueSimulation.ipynb`
Runs a standard league simulation (10,000 seasons) and produces a position probability matrix showing the likelihood of each team finishing in each position.

#### `RunConditionalLeagueSimulation.ipynb`
Runs a conditional simulation that splits results based on the outcome of a specific upcoming fixture. Produces three separate probability matrices — one for each possible result (home win, draw, away win) — so you can see how a single match affects the final table.

To change the fixture being analysed, edit the `home_team` and `away_team` parameters in the notebook.

---

## File Structure

```
├── PremierLeague_Model.html          # Full methodology document (read this first)
├── PremierLeague_Model.ipynb         # Source notebook for the methodology document
├── RunLeagueSimulation.ipynb         # Standard season simulation
├── RunConditionalLeagueSimulation.ipynb  # Conditional simulation by fixture result
├── UpdateData.ipynb                  # Data update pipeline
├── AssistingFunctions.py             # All modelling and simulation functions
├── UpdateFunctions.py                # Data download and patch management functions
└── Data/
    ├── PL2020.csv – PL2025.csv       # Historical season data
    ├── PL2025_patch.csv              # Manual patches for missing results
    └── schedule.csv                  # Current season fixture list
```
