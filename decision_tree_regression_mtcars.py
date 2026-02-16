"""
Decision Tree Regression on the mtcars dataset.

Predicts miles per gallon (mpg) from vehicle characteristics using
both single-feature and multi-feature decision tree regressors,
with tree visualisation and feature importance analysis.
"""

import io
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── mtcars dataset (embedded so no external download is needed) ──────────────

MTCARS_CSV = """\
name,mpg,cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb
Mazda RX4,21.0,6,160.0,110,3.90,2.620,16.46,0,1,4,4
Mazda RX4 Wag,21.0,6,160.0,110,3.90,2.875,17.02,0,1,4,4
Datsun 710,22.8,4,108.0,93,3.85,2.320,18.61,1,1,4,1
Hornet 4 Drive,21.4,6,258.0,110,3.08,3.215,19.44,1,0,3,1
Hornet Sportabout,18.7,8,360.0,175,3.15,3.440,17.02,0,0,3,2
Valiant,18.1,6,225.0,105,2.76,3.460,20.22,1,0,3,1
Duster 360,14.3,8,360.0,245,3.21,3.570,15.84,0,0,3,4
Merc 240D,24.4,4,146.7,62,3.69,3.190,20.00,1,0,4,2
Merc 230,22.8,4,140.8,95,3.92,3.150,22.90,1,0,4,2
Merc 280,19.2,6,167.6,123,3.92,3.440,18.30,1,0,4,4
Merc 280C,17.8,6,167.6,123,3.92,3.440,18.90,1,0,4,4
Merc 450SE,16.4,8,275.8,180,3.07,4.070,17.40,0,0,3,3
Merc 450SL,17.3,8,275.8,180,3.07,3.730,17.60,0,0,3,3
Merc 450SLC,15.2,8,275.8,180,3.07,3.780,18.00,0,0,3,3
Cadillac Fleetwood,10.4,8,472.0,205,2.93,5.250,17.98,0,0,3,4
Lincoln Continental,10.4,8,460.0,215,3.00,5.424,17.82,0,0,3,4
Chrysler Imperial,14.7,8,440.0,230,3.23,5.345,17.42,0,0,3,4
Fiat 128,32.4,4,78.7,66,4.08,2.200,19.47,1,1,4,1
Honda Civic,30.4,4,75.7,52,4.93,1.615,18.52,1,1,4,2
Toyota Corolla,33.9,4,71.1,65,4.22,1.835,19.90,1,1,4,1
Toyota Corona,21.5,4,120.1,97,3.70,2.465,20.01,1,0,3,1
Dodge Challenger,15.5,8,318.0,150,2.76,3.520,16.87,0,0,3,2
AMC Javelin,15.2,8,304.0,150,3.15,3.435,17.30,0,0,3,2
Camaro Z28,13.3,8,350.0,245,3.73,3.840,15.41,0,0,3,4
Pontiac Firebird,19.2,8,400.0,175,3.08,3.845,17.05,0,0,3,2
Fiat X1-9,27.3,4,79.0,66,4.08,1.935,18.90,1,1,4,1
Porsche 914-2,26.0,4,120.3,91,4.43,2.140,16.70,0,1,5,2
Lotus Europa,30.4,4,95.1,113,3.77,1.513,16.90,1,1,5,2
Ford Pantera L,15.8,8,351.0,264,4.22,3.170,14.50,0,1,5,4
Ferrari Dino,19.7,6,145.0,175,3.62,2.770,15.50,0,1,5,6
Maserati Bora,15.0,8,301.0,335,3.54,3.570,14.60,0,1,5,8
Volvo 142E,21.4,4,121.0,109,4.11,2.780,18.60,1,1,4,2
"""


def load_data() -> pd.DataFrame:
    """Load the mtcars dataset into a DataFrame."""
    df = pd.read_csv(io.StringIO(MTCARS_CSV))
    df = df.set_index("name")
    return df


def single_feature_tree(df: pd.DataFrame) -> None:
    """Decision tree regression using only weight (wt) to predict mpg."""
    print("=" * 60)
    print("Decision Tree Regression: mpg ~ wt")
    print("=" * 60)

    X = df[["wt"]].values
    y = df["mpg"].values

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    print(f"  Tree depth  : {model.get_depth()}")
    print(f"  Leaf nodes  : {model.get_n_leaves()}")
    print(f"  R-squared   : {r2_score(y, y_pred):.4f}")
    print(f"  RMSE        : {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print()

    # Scatter plot with step-function prediction
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["wt"], df["mpg"], color="steelblue", edgecolors="white", s=60, label="Data")
    wt_range = np.linspace(df["wt"].min() - 0.3, df["wt"].max() + 0.3, 500).reshape(-1, 1)
    ax.plot(wt_range, model.predict(wt_range), color="tomato", linewidth=2, label="Tree prediction")
    ax.set_xlabel("Weight (1000 lbs)")
    ax.set_ylabel("Miles per Gallon")
    ax.set_title("Decision Tree Regression: mpg ~ wt (max_depth=3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("dt_simple_regression.png", dpi=150)
    plt.close(fig)
    print("  Plot saved to dt_simple_regression.png\n")

    # Tree diagram
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(model, feature_names=["wt"], filled=True, rounded=True, ax=ax, fontsize=9)
    ax.set_title("Decision Tree: mpg ~ wt", fontsize=14)
    fig.tight_layout()
    fig.savefig("dt_simple_tree.png", dpi=150)
    plt.close(fig)
    print("  Tree diagram saved to dt_simple_tree.png\n")


def multiple_feature_tree(df: pd.DataFrame) -> None:
    """Decision tree regression using wt, hp, and disp to predict mpg."""
    print("=" * 60)
    print("Decision Tree Regression: mpg ~ wt + hp + disp")
    print("=" * 60)

    features = ["wt", "hp", "disp"]
    X = df[features].values
    y = df["mpg"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"  Tree depth  : {model.get_depth()}")
    print(f"  Leaf nodes  : {model.get_n_leaves()}")
    print()
    print("  Training set:")
    print(f"    R-squared : {r2_score(y_train, y_pred_train):.4f}")
    print(f"    RMSE      : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print()
    print("  Test set:")
    print(f"    R-squared : {r2_score(y_test, y_pred_test):.4f}")
    print(f"    RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print()

    # Cross-validation
    cv_scores = cross_val_score(
        DecisionTreeRegressor(max_depth=4, random_state=42),
        X, y, cv=5, scoring="r2"
    )
    print(f"  5-fold CV R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()

    # Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_train, y_pred_train, label="Train", color="steelblue", edgecolors="white", s=60)
    ax.scatter(y_test, y_pred_test, label="Test", color="tomato", edgecolors="white", s=60)
    lims = [min(y.min(), y_pred_test.min()) - 1, max(y.max(), y_pred_test.max()) + 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlabel("Actual mpg")
    ax.set_ylabel("Predicted mpg")
    ax.set_title("Decision Tree: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig("dt_multiple_regression.png", dpi=150)
    plt.close(fig)
    print("  Plot saved to dt_multiple_regression.png\n")

    # Tree diagram
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(model, feature_names=features, filled=True, rounded=True, ax=ax, fontsize=8)
    ax.set_title("Decision Tree: mpg ~ wt + hp + disp", fontsize=14)
    fig.tight_layout()
    fig.savefig("dt_multiple_tree.png", dpi=150)
    plt.close(fig)
    print("  Tree diagram saved to dt_multiple_tree.png\n")


def feature_importance(df: pd.DataFrame) -> None:
    """Train on all features and display feature importances."""
    print("=" * 60)
    print("Feature Importance (all predictors, max_depth=4)")
    print("=" * 60)

    features = [c for c in df.columns if c != "mpg"]
    X = df[features].values
    y = df["mpg"].values

    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 40)
        print(f"  {feat:>5s}: {imp:.4f}  {bar}")
    print()

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Decision Tree Feature Importance for mpg")
    fig.tight_layout()
    fig.savefig("dt_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Plot saved to dt_feature_importance.png\n")


def main() -> None:
    df = load_data()

    print(f"Dataset: {df.shape[0]} observations, {df.shape[1]} variables\n")
    print(df.describe().round(2))
    print()

    single_feature_tree(df)
    multiple_feature_tree(df)
    feature_importance(df)

    print("Done.")


if __name__ == "__main__":
    main()
