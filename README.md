# Meteo Bakery - Bakery Sales Forecasting using Weather Data

In this project, we performed sales forecasting for bakery products using weather data.
The data was kindly provided by our stakeholder [meteolytix](https://meteolytix.de).

## Project Summary

Overproduction and food waste are a huge problem in food industry. This is particularly true for bakeries, which often face the problem of wasting large amounts of food due to overproduction. On the other hand, problems can also arise from underproduction, such as sellout, customer dissatisfaction ultimately loss of customers. Thus, there is a need for generating precise forecasting solutions for bakery product sales.

Together with our stakeholder meteolytix, an AI and IT company specialized in using weather data to improve sales forecasting, we worked on a project where we used weather data to improve sales forecasting for different bakery products. We established a solution that generates separate 7-day forecasts for different bakery products at different bakery branches based on time and weather data using LightGBM, a decision-tree based gradient boosting ensemble technique. 

We found that our solution outperforms a Naive Seasonal drift baseline model, which uses the sales of the preceding 7 days as a proxy for the sales on the upcoming 7 days. Including weather data into our model made up about 10% of the total improvement compared to our baseline drift model. Thus, using weather data can improve sales forecasting for bakery products, thereby reducing costs due to over- or underproduction and possibly also increased sustainability.

## Environment

Below you will find instructions to rebuild the project.

### Requirements:

- pyenv with Python: 3.9.8

### Setup

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the commands below. Note that the lightgbm package is not available as a python package, but instead needs to be installed using brew. However, the package can be imported into python files as usual using the `import` statement.

```BASH
make setup

#or

pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
brew install lightgbm
pip install --upgrade pip
pip install -r requirements_dev.txt
```
