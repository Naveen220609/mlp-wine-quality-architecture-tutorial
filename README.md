# mlp-wine-quality-architecture-tutorial
Exploring how Multilayer Perceptron depth and width affect wine quality classification on the UCI red wine dataset, with a reproducible Jupyter notebook, figures, and report.archive.ics.uci+1​

# Exploring Multilayer Perceptron Architectures on the Wine Quality Dataset

This repository contains the code, notebook, and report for a coursework project that studies how the **architecture** of a Multilayer Perceptron (MLP) affects performance on the **red Wine Quality dataset** from the UCI Machine Learning Repository.

The main question is:

> Is a bigger, deeper neural network always better, or can a simpler MLP work just as well on a small tabular dataset?

## Project overview

The project uses the **red wine quality** dataset (1,599 wines, 11 physicochemical features and a sensory quality score from 3 to 8). Quality is turned into a binary label:

- `good_quality = 1` if quality ≥ 7  
- `good_quality = 0` otherwise

The workflow is:

1. Load and explore the dataset (summary statistics and correlation heatmap).
2. Create a binary target and perform a stratified 80/20 train–test split.
3. Scale features with `StandardScaler`.
4. Train a **baseline MLP** with one hidden layer of 50 neurons.
5. Run a grid search over several architectures:
   - `(50,)`, `(100,)`, `(100, 100)`, `(100, 100, 100)`  
   - `alpha ∈ {0.0001, 0.001}` (L2 regularisation strength) 
6. Compare models using cross‑validation accuracy, test accuracy, confusion matrices and learning curves.
7. Summarise the results in a short, accessible report.

Key finding: on this dataset, a **shallow MLP with a single wide hidden layer** performs as well as or better than deeper networks, illustrating that more layers do not automatically mean better generalisation.

## Installation

Tested with Python 3.10+.

clone: git clone https://github.com/Naveen220609/mlp-wine-quality-architecture-tutorial.git
cd mlp-wine-quality-architecture-tutorial

The required : dependencies 
numpy
pandas
scikit-learn
matplotlib
seaborn
text

## Data

The project uses the **red wine quality** CSV from UCI:

- UCI dataset page: https://archive.ics.uci.edu/dataset/186/wine+quality 
- File: `winequality-red.csv`

Place the CSV file in the `data/` folder so that the notebook can load it via:


data_path = "data/winequality-red.csv"
df = pd.read_csv(data_path, sep=";")
text

If you cannot include the CSV directly in the repo, add these instructions to help users download it themselves.

---

## How to run the notebook

1. Start Jupyter:
jupyter notebook
text

2. Open `mlp_wine_quality.ipynb`.

3. Run all cells from top to bottom. This will:

- Load and preprocess the data.
- Train the baseline MLP.
- Run the grid search over architectures.
- Generate:
  - Confusion matrices for baseline and best models.
  - Cross‑validation accuracy plot over architectures.
  - Learning curve for the best architecture.
  - Correlation heatmap.
- Save the result tables in the `results/` folder.

All plots use a colour‑blind‑friendly seaborn palette and larger fonts for better readability. 

---

## Results summary

- **Baseline model**: 1 hidden layer with 50 neurons, `alpha = 0.0001`
- Test accuracy ≈ **0.925**
- Strong performance on the majority class (not good wines).
- **Best CV model**: 1 hidden layer with 100 neurons, `alpha = 0.001`
- Mean cross‑validation accuracy ≈ **0.886**
- Test accuracy ≈ **0.916** (very similar to baseline).
- Deeper models with 2 or 3 hidden layers reach almost perfect training accuracy but **do not** improve validation or test accuracy, showing mild overfitting.

The report in `mlp_wine_quality_report.pdf` explains these findings in plain language, including simple descriptions of **ReLU**, **Adam**, and **alpha** for readers new to neural networks.

---

## Accessibility

To support accessibility:

- All plots use `sns.set_palette("colorblind")` and increased font sizes for axis labels and titles.
- Figures in the report include short alt‑text descriptions so that their key message can be understood with a screen reader.
- The notebook uses clear section headings and avoids colour as the only way to distinguish information where possible.

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgements

- **Dataset**: UCI Machine Learning Repository – *Wine Quality Data Set: https://archive.ics.uci.edu/dataset/186/wine+quality* 
- **MLP implementation**: `sklearn.neural_network.MLPClassifier` from scikit‑learn 
- Course guidance and assignment specification provided by the module tutor.
