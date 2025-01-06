# Salary Prediction

This project demonstrates a basic implementation of **Simple Linear Regression** to predict salaries based on years of experience. It uses the [Salary Dataset](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression/data) from Kaggle.

## Features
- **Data Preprocessing**: The dataset is loaded and converted into NumPy arrays.
- **Model Implementation**: Simple Linear Regression is implemented from scratch using Python.
- **Gradient Descent**: The model uses gradient descent to optimize weights.
- **Visualization**: Graphs of actual vs. predicted salaries are included.
- **Model Evaluation**: Key metrics such as cost function and \( R^2 \) score are calculated.

## Project Structure
The project consists of a single Jupyter Notebook file:
- `main.ipynb`: Contains the entire implementation of the project, including loading data, model training, visualization, and evaluation.

## Dataset
The dataset is sourced from Kaggle: [Salary Dataset](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression/data)  
It contains two columns:
1. `YearsExperience`: Years of work experience.
2. `Salary`: Corresponding salary.

## Requirements
- Python 3.7+
- Jupyter Notebook
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/KartikAg13/simple_linear_regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd simple_linear_regression
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Follow the cells to run the project step by step.

## Results
- **Optimal Parameters**: The model finds the best values for weight (\( w \)) and bias (\( b \)) using gradient descent.
- **Prediction Visualization**: A graph is plotted showing the regression line (predicted) vs. actual data.
- **Evaluation**: The cost function and \( R^2 \) score are calculated to assess the model's performance.

## Example Output
![image](https://github.com/user-attachments/assets/51857b53-58a7-460f-9844-500670735c68)

## Contributing
This is a basic project designed for educational purposes. Feel free to fork and extend it! Contributions are welcome.

## Acknowledgments
- [Kaggle](https://www.kaggle.com) for providing the dataset.
- [Coursera](https://www.coursera.org/learn/machine-learning/) for inspiration through the Machine Learning course.

