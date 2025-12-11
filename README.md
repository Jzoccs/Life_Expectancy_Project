# Life_Expectancy_Project

This project builds and compares several regression models attempting to predict life expectancy based on the variables of GDP per capita, CO2 emissions per capita, and year. It includes data preprocessing, model training, evaluation, and visualization of results through a Quarto website.

## Data Collection
The life expectancy data was acquired from the World Bank API, which contains life expectancy data for various countries over several years. The GDP per capita data was also sourced from the World Bank API, providing economic context for each country. The CO2 emissions per capita data was obtained from the Global Carbon Project, agithub repository, which tracks carbon emissions globally.

### Project Overview
The goals of this project are:
1. To preprocess, clean, and merge the 3 datasets we found.
2. Train a flaml machine learning algorithm to select the best model to predict life expectancy. Based on the year, GDP per capita, and CO2 emissions per capita.
3. Evaluate and compare the performance of these models using RMSE, MAE and R^2.
4. Visualize the results and findings in a Quarto website.

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
│   └── processed/
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
│   └── automl.qmd
├── requirements.txt
├── README.md
└── _quarto.yml

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets

## Help

Any advise for common problems or issues.

## Authors

Jack Zocco
Anthony Sacchetti

## Acknowledgments
* Some visualization inspirations were drawn from the following dashboards:

* [Mortgage Dashboard by Isabel Velásquez](https://ivelasq.github.io/mortgage-dashboard/)
* [Gapminder Dashboard by J.J. Allaire](https://jjallaire.github.io/gapminder-dashboard/)
* [Quarto — Data Display & iTables Documentation](https://quarto.org/docs/dashboards/data-display.html#itables)

