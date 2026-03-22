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
  <img src="https://img.shields.io/badge/UI-Glassmorphism-blue" alt="Glassmorphism" />
</p>

## 📖 Problem Statement

Predicting household electricity bills using only machine learning often leads to unrealistic results. Standard ML models might predict an energy bill that violates the strict constraints of actual appliance physics or ignores the non-linear, tier-based slab pricing structures used by state utility companies. This causes standard tabular ML models to struggle when exposed to extreme values, leading to poor generalization in real-world engineering scenarios. 

## 💡 Solution Approach

This project bridges the gap between pure data science and real-world engineering by implementing a hybrid **Physics-Informed Machine Learning** approach:

1. **Physics-Based Energy Calculation:** Computes the baseline Kilowatt-hours (kWh) using actual appliance wattage, usage hours, and energy limits. 
2. **State-Wise Slab Billing Logic:** Applies real Indian electricity board pricing rules (e.g., Gujarat, Maharashtra, Delhi) to compute the exact cost based on physical energy consumption.
3. **Ratio-Based ML Correction:** Instead of predicting the final bill directly or predicting the absolute residual, the model predicts the **Ratio** (`Actual Bill / Physics Bill`). This dynamically corrects for behavioral inefficiencies while strictly grounding the prediction in mathematical reality.

## 🚀 Why This Project Stands Out

Unlike generic ML projects that simply fit an algorithm to a CSV (e.g., standard regression datasets, Titanic survival), this project solves a complex, real-world engineering problem:

* **Domain Knowledge Integration:** Incorporates complex tier-based slab logic instead of purely relying on the model to "guess" the mathematical relationship.
* **Physics over Blackbox:** Ensures predictions are grounded in the laws of physics (Watt-hours). Safety constraints prevent the ML model from generating physically impossible electricity bills (e.g., predicting ₹50,000 for standard fan usage).
* **Ratio-Based Target over Residuals:** Modeling the ratio creates a scale-invariant learning objective, completely eliminating the drastic overprediction bias commonly found in standard residual-based learning.
* **Production Ready:** An end-to-end pipeline handling feature alignment, missing values, and seasonal interaction features, beautifully wrapped in a front-end application.

## 🏗️ System Architecture

1. **User Input:** User provides appliance usage (AC, Fridge, etc.), region (state/city), and season.
2. **Feature Engineering:** Pipeline generates interaction features like Energy Usage Intensity, seasonal impact, and temperature factors.
3. **Physics Engine:** Calculates theoretical kWh and base cost using state-wise slab logic.
4. **Machine Learning Model:** Stacking Ensemble (Ridge + Random Forest + XGBoost) predicts the expected usage inefficiency (Ratio).
5. **Final Computation:** `Final Predicted Bill = Physics Base Cost × Predicted Ratio`.
6. **Frontend:** Serves predictions via a modern Streamlit application designed with a sleek Glassmorphism aesthetic.

## ✨ Features

* **Real-World Slab Billing:** Accurately mimics Indian electricity boards' calculation matrices.
* **Seasonal & Temperature Awareness:** Captures high AC usage spikes during summer conditions.
* **Energy Usage Intensity Modeled:** Correlates overall appliance load interaction.
* **Safety Constraints:** Built-in programmatic safeguards preventing extreme or negative predictions.
* **Stacking Ensemble Architecture:** Hyper-tuned hybrid model combining linear robustness and tree-based non-linearity using `RandomizedSearchCV`.
* **Modern UI:** Glassmorphism Streamlit UI offering a premium user experience.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Model Serialization:** Joblib
* **Frontend:** Streamlit

## 📊 Model Performance

Evaluated rigorously using an industry-standard Train / Validation / Test split.

* **Metric:** Coefficient of Determination ($R^2$)
* **Validation $R^2$:** ~95%
* **Test $R^2$:** ~95%
* **Overfitting Check:** Robust performance cross-validated across all sets, confirming the stacking ensemble successfully generalizes without data leakage.

### 🤔 Why Ratio Model > Residual Model?
When predicting residuals (`Actual - Physics`), standard models struggle with high-variance spikes (like heavy AC usage vs. mild winter usage), resulting in severe over-predictions for low-energy households. By predicting the **ratio** (`Actual / Physics`), the model scales dynamically. If the model predicts a 1.2 ratio, it simply means the household is 20% less efficient than the baseline physics calculation, keeping predictions mathematically bounded and highly accurate.

## 📸 Screenshots

*(Replace these placeholders with actual screenshots of your application)*

| Dashboard View | Prediction Results |
| :---: | :---: |
| `![Dashboard Screenshot](path/to/dashboard.png)` | `![Prediction Screenshot](path/to/prediction.png)` |

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/electricity-bill-prediction.git
   cd electricity-bill-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the Streamlit web application:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your web browser. 

### Example Input/Output

**Input Parameters:**
* **State:** Maharashtra
* **Season:** Summer
* **AC Usage:** 6 Hours/Day
* **Fridge Usage:** 24 Hours/Day
* **Fans:** 3 (8 Hours/Day each)

**Output Engine:** 
* *Physics Calculation:* 400 kWh ➡️ ₹3,500 (Base estimation)
* *ML Ratio Correction:* 1.15 (Accounts for peak summer heat inefficiency and appliance depreciation)
* *Final Prediction:* **₹4,025**

## 🔮 Future Improvements

* ☀️ **Solar Panel Integration:** Account for net-metering laws and solar generation offsets.
* 💰 **Subsidy Mechanisms:** Add dynamic logic for government electricity subsidies based on consumption limits.
* 🕰️ **Time-of-Use (ToU) Pricing:** Factor in smart-meter dynamic pricing structures (peak vs. off-peak hours).