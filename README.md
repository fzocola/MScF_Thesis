# Credit Rating Transition Modeling

## Can We Predict Whether Issuers Will Become Fallen Angels or Rising Stars?

### Abstract
Credit-rating transitions, especially downgrades from investment-grade to high-yield ("fallen angels") and upgrades from high-yield to investment-grade ("rising stars"), can have a significant impact on the value of corporate bonds.
This thesis develops a parsimonious yet economically grounded framework to estimate one-year credit rating transition probabilities for U.S. non-financial corporations. Adapting a discrete‑time hazard‑rate approach used in default modeling, rating transitions are modeled as Bernoulli events, with transition hazard rates estimated through panel logistic regressions.
Three specifications are compared. The first uses accounting ratios only (EBIT/TA, TD/TA, SIZE), the second market information only (distance‑to‑default derived from a Merton–KMV structural model). Finally, the third combines both sets of variables.
The data set covers 238 issuers and 32,321 monthly observations (2011‑2023), yielding 66 downgrades and 63 upgrades.
Results show that distance‑to‑default is a powerful single predictor, and that combining market and accounting variables materially improves model goodness-of-fit.
The model produces insightful in-sample probability estimates for identifying potential fallen angels and rising stars.
These findings confirm that a straightforward hazard-rate model can deliver timely and actionable early warning signals, helping analysts allocate surveillance efforts toward issuers with the highest risk of crossing the investment-grade/high-yield boundary.


---

### Contact
|            |                        |
|------------|------------------------|
| **Author** | Florian Zocola         |
| **Email**  | florian.zocola@unil.ch |

---

### Prerequisites
This project has been developed with the following Python interpreter:
- **Python 3.11**

> **Note:** All required third‑party packages are pinned in **`requirements.txt`**.  
> Install them with  
> ```bash
> pip install -r requirements.txt
> ```

---

