# Life_Expectancy_Project

This project builds and compares several regression models attempting to predict life expectancy based on the variables of GDP per capita, CO2 emissions per capita, and year. It includes data preprocessing, exploratory data analysis, model training, evaluation, and visualization of results through a Quarto website.

## Data Collection
The life expectancy data was acquired from the World Bank API, which contains life expectancy data for various countries over several years. The GDP per capita data was also sourced from the World Bank API, providing economic context for each country. The CO2 emissions per capita data was obtained from the Global Carbon Project, agithub repository, which tracks carbon emissions globally.

### Project Overview
The goals of this project are:
1. To preprocess, clean, and merge the 3 datasets we found.
2. To perform exploratory data analysis to find underlying trends in the data
3. Train multiple machine learning algorithms using flaml to select the best model to predict life expectancy. Based on the year, GDP per capita, and CO2 emissions per capita.
4. Evaluate and compare the performance of these models using RMSE, MAE and R^2.
5. Visualize the results and findings in a Quarto website.

## Getting Started

### Dependencies

* Python 3.13+
* Quarto
* All required packages in requirements.txt
* An API Key is needed to access the life expectancy dataset and the GDP dataset from the World Bank. You can obtain an API key by signing up on the World Bank's data portal.

### Project Structure
Life_Expectancy_Project/
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── notebooks/
├── src/
│   ├── models/
│   ├── data/
│   └── visualizations/
├── website/
│   ├── index.qmd
│   ├── methodology.qmd
│   ├── analysis.qmd
│   ├── data.qmd
│   ├── visualizations.qmd
│   ├── automl_life_expectancy.qmd
│   └── _quarto.yml
├── requirements.txt
├── README.md
└── .gitignore

### Executing program

1. Pull the external co2 dataset from https://github.com/owid/co2-data
2. run the files in /src/data/ in order to pull the csv's from the API's and merge datasets
3. run the /notebooks/data_cleaning_and_exploration.ipynb file to clean the data
4. run the /notebooks/visualizations.ipynb to perform EDA and make visuals
5. run the /src/models/train_life_expectancy_model.py to train all four models
6. run the /notebooks/Regression_visualizations.ipynb to get the regression visualizations 

## Authors

Jack Zocco
Anthony Sacchetti

## Acknowledgments
* Some visualization inspirations were drawn from the following dashboards:

* [Mortgage Dashboard by Isabel Velásquez](https://ivelasq.github.io/mortgage-dashboard/)
* [Gapminder Dashboard by J.J. Allaire](https://jjallaire.github.io/gapminder-dashboard/)
* [Quarto — Data Display & iTables Documentation](https://quarto.org/docs/dashboards/data-display.html#itables)

