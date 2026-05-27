Would like to share I've been working through a Kaggle Playground problem that asks a simple but strategy-heavy question: will this car pit on the next lap?

The dataset is lap-level Formula 1 data (tyre life, stint, race progress, lap times, position changes, and more) with a binary target called `PitNextLap`. Roughly **80%** of laps are "no pit" and only about **20%** are pit next lap (~4:1). That imbalance shaped almost every modeling decision, not just the final metric I reported.

────────────────────────────────────────

Problem statement

This is an imbalanced binary classification problem. A dummy model that **always predicts "no pit"** would already hit ~**80% accuracy** while catching **zero** real pit stops, so headline accuracy was never a fair way to judge success.

I cared most about the **minority class** (pit next lap): **recall** (how many true pits I catch), **F1** (balance of precision and recall on that class), and **ROC-AUC** (how well the model ranks pit vs non-pit laps). High accuracy alone could still mean the model was mostly ignoring the events strategists actually care about.

────────────────────────────────────────

How I managed the imbalance

I deliberately **did not** start with oversampling (SMOTE) or undersampling. With ~439K rows, I wanted to keep all real laps and avoid synthetic pit samples skewing the distribution.

Instead I used **class weighting** so the minority class counted more during training:

• **Random Forest**: `class_weight="balanced"` (sklearn reweights classes inversely to frequency)
• **XGBoost / LightGBM**: `scale_pos_weight` set from the fold's class ratio (~4:1, matching the ~80/20 split)

That nudges trees to pay attention to pit stop patterns without throwing away data or generating fake minority rows. The tradeoff showed up clearly in results: models with stronger weighting toward the minority class (boosters) pushed **pit recall toward ~0.90**, while Random Forest stayed more conservative (~0.67 recall) but with higher **pit precision** (~0.75).

I also used a **stratified** train/validation split so both sets kept the same ~80/20 mix, so validation scores weren't inflated by a lucky split with even fewer pit laps than reality.

────────────────────────────────────────

Models & experimentation

Before trusting a single validation score, I added 5-fold OOF evaluation so each training lap got a prediction from a model that never saw it in that fold. The encouraging part: OOF and holdout pit class F1 were almost identical. Gaps were only about 0.002 to 0.005, which made me feel the holdout split wasn't telling a totally different story than cross-validation.

I compared Random Forest (`class_weight="balanced"`), XGBoost, and LightGBM (both with `scale_pos_weight` ≈ 4 per fold):

• Random Forest → Holdout accuracy: 0.89 | Pit Recall: 0.67 | Pit F1: 0.71 | OOF Pit F1: 0.706
• XGBoost → Holdout accuracy: 0.86 | Pit Recall: 0.90 | Pit F1: 0.73 | ROC-AUC: 0.944 | OOF Pit F1: 0.723
• LightGBM → Holdout accuracy: 0.86 | Pit Recall: 0.91 | Pit F1: 0.72 | ROC-AUC: 0.944 | OOF Pit F1: 0.721

Weighting alone didn't make them behave the same: boosters pushed pit recall toward ~0.90, while Random Forest stayed more conservative on recall but stronger on pit precision.

────────────────────────────────────────

Results & model comparison

• Random Forest → Holdout accuracy: 0.89 | Pit Recall: 0.67 | Pit F1: 0.71 | OOF Pit F1: 0.706
• XGBoost → Holdout accuracy: 0.86 | Pit Recall: 0.90 | Pit F1: 0.73 | ROC-AUC: 0.944 | OOF Pit F1: 0.723
• LightGBM → Holdout accuracy: 0.86 | Pit Recall: 0.91 | Pit F1: 0.72 | ROC-AUC: 0.944 | OOF Pit F1: 0.721

The main lesson from imbalance handling: **the "best" model depends on which error you can afford.** Random Forest wins if you only look at accuracy and want cleaner pit predictions (higher precision). XGBoost and LightGBM are better if the goal is to **flag most real pit windows**, accepting more false positives (lower pit precision, higher pit recall).

Class weighting got me there without resampling, but it didn't remove the precision/recall tradeoff. It just made it visible: boosters behave like a recall focused pit detector; Random Forest behaves like a more cautious one.

OOF vs holdout staying aligned gave me confidence those patterns were stable, not a one off split artifact.

Feature importance from the boosting models lined up with intuition: `RaceProgress`, `TyreLife`, and `LapTime_Delta` were among the strongest drivers, which makes sense for pit timing.

────────────────────────────────────────

Competition progress

After tuning and experimentation, my current notebook achieved a competition accuracy of 85.81%, placing me at #1874 globally on the leaderboard so far. There are still 11 days left in the competition, so this definitely isn’t the final version of the notebook yet. Over the next few days, I’ll explore feature engineering, ensembles, and validation improvements to further boost performance before the competition ends.

#MachineLearning #DataScience #Formula1 #Kaggle #TabularML #ImbalancedClassification #XGBoost #LightGBM #RandomForest #CrossValidation
