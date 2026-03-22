<h1 align="center">⚡ Electricity Bill Prediction using Physics-Informed Machine Learning</h1>

<div align="center">
  A production-ready web application that combines physics-based energy calculations, state-wise electricity slab pricing, and an ensemble machine learning model to accurately predict household electricity bills.
</div>

<p align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD.svg?style=flat&logo=XGBoost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Stacking%20Ensemble-green" alt="Machine Learning" />
</p>

## 📋 Table of Contents
- [📖 Problem Statement](#-problem-statement)
- [💡 Solution Approach](#-solution-approach)
- [🚀 Why This Project Stands Out](#-why-this-project-stands-out)
- [🏗️ System Architecture](#️-system-architecture)
- [✨ Core Features](#-core-features)
- [📉 Dataset & Statistical Insights](#-dataset--statistical-insights)
- [📊 Model Performance](#-model-performance)
- [⚙️ In-Depth Execution Guide (How to Run)](#️-in-depth-execution-guide-how-to-run)
- [🔮 Future Improvements](#-future-improvements)

---

## 📖 Problem Statement

Predicting household electricity bills using only machine learning often leads to unrealistic results. Standard ML models might predict an energy bill that violates the strict constraints of actual appliance physics or ignores the non-linear, tier-based slab pricing structures used by state utility companies. This causes standard tabular ML models to struggle when exposed to extreme values, leading to poor generalization in real-world engineering scenarios. 

## 💡 Solution Approach

This project bridges the gap between pure data science and real-world engineering by implementing a hybrid **Physics-Informed Machine Learning** approach:

1. **Physics-Based Energy Calculation:** Computes the baseline Kilowatt-hours (kWh) using actual appliance wattage, usage hours, and energy limits. 
2. **State-Wise Slab Billing Logic:** Applies real Indian electricity board pricing rules (covering 10 major states) to compute the exact cost based on physical energy consumption.
3. **Ratio-Based ML Correction:** Instead of predicting the final bill directly or predicting the absolute residual, the model predicts the **Ratio** (`Actual Bill / Physics Bill`). This dynamically corrects for behavioral inefficiencies.

## 🚀 Why This Project Stands Out

Unlike generic ML projects that simply fit an algorithm to a CSV, this project solves a complex, real-world engineering problem:

* **Domain Knowledge Integration:** Incorporates complex tier-based slab logic instead of purely relying on the model to "guess" the mathematical relationship.
* **Physics over Blackbox:** Ensures predictions are grounded in the laws of physics (Watt-hours). Safety constraints prevent the ML model from generating physically impossible electricity bills.
* **Ratio-Based Target over Residuals:** Modeling the ratio creates a scale-invariant learning objective, completely eliminating the drastic overprediction bias commonly found in standard residual-based learning.

## 🏗️ System Architecture

1. **User Input:** User provides appliance usage (Fan, TV, Refrigerator, Computer, AC, Motor), State, and Month.
2. **Feature Engineering:** Pipeline generates interaction features mapping months to seasons (Winter, Summer, Monsoon) and checks for high usage logic.
3. **Physics Engine:** Calculates theoretical kWh and base cost using state-wise slab logic.
4. **Machine Learning Model:** Stacking Ensemble (Ridge + Random Forest + XGBoost) predicts the expected usage inefficiency (Ratio).
5. **Final Computation:** `Final Predicted Bill = Physics Base Cost × Predicted Ratio`.
6. **Frontend:** Serves predictions and energy-saving tips via a Streamlit application.

## ✨ Core Features

* **Appliance-Level Modeling:** Supports detailed daily usage inputs for Fans, TVs, Fridges, Computers, ACs, and Water Motors.
* **Multi-State Slab Support:** Dynamic pricing for 10 Indian States: Gujarat, Maharashtra, Delhi, Karnataka, Tamil Nadu, Rajasthan, Uttar Pradesh, West Bengal, Madhya Pradesh, and Telangana.
* **Actionable Energy Tips:** Suggests specific cost-saving actions (e.g., reducing AC time, considering solar power for bills > ₹5,000).
* **Stacking Ensemble Architecture:** Hyper-tuned hybrid model combining linear robustness and tree-based non-linearity using `RandomizedSearchCV`.

## 📉 Dataset & Statistical Insights

The project utilizes a rich, custom dataset simulating realistic household behaviors overlapping with physics laws:

* **Dataset Size:** `45,345 records` × `12 features`
* **Average Actual Bill:** `₹2,948` per household simulation.
* **Target Ratio Variance:** The model targets a multiplier ratio ranging up to `1.30` (representing up to ~30% physical inefficiency due to old appliances, extreme heat loss, etc.).
* **Data Consistency:** Quality checks during the pipeline ensure the ratio gap between physics and actuals stays bounded to realistic margins (Mean Ratio standard deviation: ~`0.06`).

## 📊 Model Performance

Evaluated rigorously using an industry-standard Train / Validation / Test split over the 45k+ samples. The stacking ensemble provides massive resistance to overfitting.

| Metric | Validation Set | Test Set |
|--------|----------------|----------|
| **R² Score** | `0.9563` (95.6%) | `0.9515` (95.1%) |
| **RMSE** | ₹108 | ₹111 |
| **MAE** | ₹68 | ₹71 |

### 🤔 Why Ratio Model > Residual Model?
When predicting residuals (`Actual - Physics`), standard models struggle with high-variance spikes. By predicting the **ratio** (`Actual / Physics`), the model scales dynamically. If the model predicts a `1.15` ratio, the household is essentially 15% less efficient than the baseline calculation—keeping predictions strictly bounded.

## ⚙️ In-Depth Execution Guide (How to Run)

The repository provides modular scripts to reproduce the ML pipeline from scratch and launch the interactive UI.

### 1. Environment Setup

* **Clone the repository:**
  ```bash
  git clone https://github.com/sahil-vasani/electricity-bill-prediction-ml.git
  cd electricity-bill-prediction-ml
  ```

* **Create & Activate a Virtual Environment (Highly Recommended):**
  ```bash
  # Windows
  python -m venv venv
  .\venv\Scripts\activate
  
  # macOS/Linux
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

### 2. Running the ML Pipeline (Data Processing & Training)

If you want to re-train the model or regenerate the `processed.csv`, execute the main pipeline orchestrator in your terminal:

```bash
python main.py
```
**What happens under the hood:**
1. Loads raw dataset from `data/electricity_bill_dataset.csv`.
2. Cleans constraints and encodes features (`src/preprocess.py`).
3. Computes the `physics_bill` using individual state slab rates (`src/slab_rates.py` & `src/feature_engineering.py`).
4. Performs sanity checks and exports `data/processed.csv`.
5. Trains the Stacking Model (Ridge, RF, XGBoost) and performs cross-validation (`src/train_model.py`).
6. Saves `models/electricity_model.pkl`.

### 3. Launching the Web Application

To use the interactive dashboard for predictions, run the following command to start the Streamlit server:

```bash
streamlit run app.py
```
* **Local URL:** `http://localhost:8501`
* **Usage:** Select your state, current month, and slide the appliance hours. Click "Predict Electricity Bill" to see your estimated monthly cost, units consumed (kWh), and intelligent energy tips.

## 🔮 Future Improvements

* ☀️ **Solar Panel Integration:** Account for net-metering laws and solar generation offsets.
* 💰 **Subsidy Mechanisms:** Add dynamic logic for government electricity subsidies based on consumption limits.
* 🕰️ **Time-of-Use (ToU) Pricing:** Factor in smart-meter dynamic pricing structures (peak vs. off-peak hours).