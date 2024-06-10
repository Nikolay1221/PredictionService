import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import timedelta
import holidays
import time
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import itertools 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics
from tqdm import tqdm  
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.optimize import minimize
from datetime import datetime
from flask import jsonify
import traceback

def missing_data(input_data):
    '''
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.
    
    input: pandas df
    output: pandas df
    
    '''
    
    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)
    table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    types = []
    for col in input_data.columns: 
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return(pd.DataFrame(table))

def mape(actual, pred): 
    '''
    Mean Absolute Percentage Error (MAPE) Function
    
    input: list/series for actual values and predicted values
    output: mape value 
    '''
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def read_sales_data(json_data):
    try:
        # Считывание данных из JSON, предоставленного как строка или словарь
        # Если json_data представляет собой строку:
        if isinstance(json_data, str):
            df = pd.read_json(json_data, lines=True)
        else:
            # Если json_data представляет собой словарь или список словарей:
            df = pd.DataFrame(json_data)

        # Приведение названий столбцов к нижнему регистру
        df.columns = df.columns.str.lower()
        print(df.columns)


        # Преобразование столбца 'date' в формат datetime
        df['yyyy_mm'] = pd.to_datetime(df['yyyy_mm'], format="%Y-%m-%d")
        df['yyyy_mm'] = df['yyyy_mm'].dt.to_period('D')

        return df
    except ValueError as e:
        print(f"Ошибка при чтении JSON: {e}")
        return None

def create_total_sales_df(df):
    agg_df = df.groupby(['yyyy_mm', 'productcategory']).agg({'sales': 'sum'}).reset_index().sort_values(['productcategory', 'yyyy_mm'])
    total_sales_df = agg_df.pivot(index='yyyy_mm', columns='productcategory', values='sales')
    return total_sales_df


def remove_outliers(total_sales_df):
    prediction_df_list = []
    
    for column in total_sales_df.columns:
        df_clean = total_sales_df[[column]].dropna()
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        prediction_df_list.append(filtered_df)
        
    return prediction_df_list


# Пример функции, которая будет передавать прогресс
def optimize_prophet_params(prediction_df_list, total_sales_df, socketio=None):
    changepoint_prior_scale_range = np.linspace(0.001, 0.5, num=5).tolist()
    seasonality_prior_scale_range = np.linspace(0.01, 10, num=5).tolist()
    start_time = time.time()

    dicts = {}

    for index, filtered_df in enumerate(prediction_df_list):
        feature = total_sales_df.columns[index]
        category_df = filtered_df.copy().reset_index()
        category_df.columns = ["ds", "y"]

        category_df["y"] = category_df["y"].apply(pd.to_numeric)
        category_df["ds"] = category_df["ds"].dt.to_timestamp()

        param_grid = {
            "changepoint_prior_scale": changepoint_prior_scale_range,
            "seasonality_prior_scale": seasonality_prior_scale_range
        }

        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        mapes = []

        for i, params in enumerate(all_params):
            m = Prophet(**params).fit(category_df)
            df_cv = cross_validation(m, initial="180 days", period="30 days", horizon="30 days")
            df_p = performance_metrics(df_cv, rolling_window=1)
            mape = np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y'])) * 100
            mapes.append(mape)

            if socketio:
                socketio.emit('progress', {'step': 'param_optimization', 'index': i, 'total': len(all_params)})

        tuning_results = pd.DataFrame(all_params)
        tuning_results["mape"] = mapes

        params_dict = dict(tuning_results.sort_values("mape").reset_index(drop=True).iloc[0])
        params_dict["column"] = feature

        dicts[feature] = params_dict

    print("--- %s seconds ---" % (time.time() - start_time))
    return dicts

def generate_holiday_data(country_code='MX', years=[2017, 2018]):
    # Создаём пустой DataFrame для хранения данных о праздниках
    holiday_df = pd.DataFrame([])

    # Получаем объект праздников для заданной страны
    country_holidays = holidays.country_holidays(country_code)

    # Проходим по годам и записываем информацию о праздниках в DataFrame
    for year in years:
        for date, name in sorted(country_holidays(year=year).items()):
            temp_df = pd.DataFrame({
                'ds': [date],
                'holiday': [name],
                'lower_window': [-2],
                'upper_window': [1]
            })
            holiday_df = pd.concat([holiday_df, temp_df], ignore_index=True)

    # Преобразуем столбец 'ds' в формат datetime, обрабатывая исключения
    try:
        holiday_df['ds'] = pd.to_datetime(holiday_df['ds'], format='%Y-%m-%d')
    except ValueError as e:
        print(f"Error converting date: {e}")
        # Optionally, handle or log the error as needed

    return holiday_df


def evaluate_sales_predictions(daily_aggregated, second_dict):
    evaluation_results = {}  
    evaluation_summary = {}
    fit_models = {}
    predictions = {}
    evaluation_metrics = {}

    holiday = pd.DataFrame([])

    mexico_holidays = holidays.MX()  # this is a dict

    mexico_holidays = holidays.country_holidays('MX')  # this is a dict


    for date_, name in sorted(holidays.MX(years=[2017,2018]).items()):
        holiday = pd.concat([holiday, pd.DataFrame({'ds': date_, 'holiday': "MX-Holidays", 'lower_window': -2, 'upper_window': 1}, index=[0])], ignore_index=True) 

    holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')
    
    for product in daily_aggregated['product_id'].unique():
        frame = daily_aggregated[daily_aggregated['product_id'] == product].copy()
        frame.rename(columns={'yyyy_mm': 'ds', 'sales': 'y'}, inplace=True)
        frame['ds'] = frame['ds'].dt.to_timestamp()

        train_frame = frame[frame['ds'] < '2018-07-01']
        test_frame = frame[frame['ds'] >= '2018-07-01']

        if len(train_frame['y']) > 1 and len(train_frame['y'].unique()) > 1:
            train_frame = train_frame.copy()
            train_frame['y'], lambda_prophet = boxcox(train_frame['y'] + 1)
        else:
            print("Недостаточно данных для преобразования Бокса-Кокса или данные не вариативны.")

        category = frame['productcategory'].iloc[0]

        if train_frame.shape[0] >= 2:
            feature = 'category'  # Замените 'category' на имя соответствующего столбца категории
            m = Prophet(
                changepoint_prior_scale=second_dict[category]['changepoint_prior_scale'],
                seasonality_prior_scale=second_dict[category]['seasonality_prior_scale'],
                seasonality_mode='multiplicative'
                ,holidays=holiday
            )

            m.fit(train_frame[['ds', 'y']])
            fit_models[product] = m

            try:
                cv_results = cross_validation(m, horizon='30 days', period='30 days', initial='180 days')
                metrics = performance_metrics(cv_results)
                evaluation_results[product] = metrics
            except Exception as e:
                print(f"Ошибка при кросс-валидации для продукта {product}: {e}")

            future_dates = m.make_future_dataframe(periods=365, freq='D')    
            forecast = m.predict(future_dates)
            predictions[product] = forecast
            forecast['yhat'] = inv_boxcox(forecast['yhat'], lambda_prophet) - 1

            if test_frame.empty or forecast[['ds', 'yhat']].empty:
                evaluation_summary[product] = "One of the daily_aggregated frames is empty."
            else:
                test_forecast_merged = test_frame.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
                if test_forecast_merged.empty:
                    evaluation_summary[product] = "No overlapping dates found."
                else:
                    mae = mean_absolute_error(test_forecast_merged['y'], test_forecast_merged['yhat'])
                    rmse = np.sqrt(mean_squared_error(test_forecast_merged['y'], test_forecast_merged['yhat']))
                    mape = np.mean(np.abs((test_forecast_merged['y'] - test_forecast_merged['yhat']) / 
                              (test_forecast_merged['y'] + np.finfo(float).eps))) * 100

                    evaluation_metrics[product] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                    evaluation_summary[product] = evaluation_metrics[product]

                    print(f'Product: {product}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    return predictions


def aggregate_final_data(daily_aggregated, predictions):
    if daily_aggregated['yyyy_mm'].dtype.name == 'period[D]':
        daily_aggregated['yyyy_mm'] = daily_aggregated['yyyy_mm'].dt.to_timestamp()

    final_df_2018 = pd.DataFrame()

    for product in daily_aggregated['product_id'].unique():
        forecast_2018 = predictions.get(product)
        if forecast_2018 is None:
            continue
        forecast_2018 = forecast_2018[forecast_2018['ds'].dt.year == 2018]

        actual_sales_2018 = daily_aggregated[daily_aggregated['product_id'] == product]
        actual_sales_2018 = actual_sales_2018[actual_sales_2018['yyyy_mm'].dt.year == 2018]

        merged_data_2018 = forecast_2018.merge(actual_sales_2018, left_on='ds', right_on='yyyy_mm', how='left')
        merged_data_2018['product_id'] = product
        merged_data_2018['min'] = merged_data_2018['min'].ffill()
        merged_data_2018['max'] = merged_data_2018['max'].ffill()

        final_df_2018 = pd.concat([final_df_2018, merged_data_2018], ignore_index=True)

    final_df_2018.drop(columns=['yyyy_mm'], inplace=True)
    final_df_2018.rename(columns={'yhat': 'Predicted Sales'}, inplace=True)
    final_df_2018['month'] = final_df_2018['ds'].dt.to_period('M').dt.to_timestamp()

    # Заполнение пропущенных категорий модой
    category_mode = final_df_2018.groupby('product_id')['productcategory'].agg(lambda x: x.mode().iloc[0])
    final_df_2018['productcategory'] = final_df_2018.apply(
        lambda row: category_mode[row['product_id']] if pd.isna(row['productcategory']) else row['productcategory'],
        axis=1
    )

    final_df_2018['min'] = final_df_2018['min'].fillna(final_df_2018['min'].mean())
    final_df_2018['max'] = final_df_2018['max'].fillna(final_df_2018['max'].mean())

    monthly_aggregated = final_df_2018.groupby(['product_id', 'month', 'productcategory', 'min', 'max']).agg({
        'Predicted Sales': 'sum',  
        'sales': 'sum',
        'balancestart': 'sum',
        'transit': 'sum',
        'backorder': 'sum'
    }).reset_index()

    return monthly_aggregated


def calculate_turnover(result):

    result['arrival'] = result['transit'] + result['backorder'] + result['orders']
    
    # Инициализируем `balance start` для первого месяца, если он еще не установлен
    if pd.isna(result.loc[0, 'balancestart']):
        result.loc[0, 'balancestart'] = 0  # или другое подходящее начальное значение

    # Вычисляем `expenditure` и `balance end` на основе `Predicted Sales`
    for i in range(len(result)):
        if i > 0:
            # Обновляем `balance start` для текущего месяца на основе `balance end` предыдущего месяца
            result.loc[i, 'balancestart'] = result.loc[i - 1, 'balance end']

        # `expenditure` равен минимуму между начальным балансом и предсказанными продажами
        result.loc[i, 'expenditure'] = min(result.loc[i, 'balancestart'], result.loc[i, 'Predicted Sales'])

        # Вычисляем `balance end` как начальный баланс плюс поступления минус расход
        result.loc[i, 'balance end'] = result.loc[i, 'balancestart'] + result.loc[i, 'arrival'] - result.loc[i, 'expenditure']


    average_predicted_sales = result['Predicted Sales'].mean()
    result['next_month_sales'] = result['Predicted Sales'].shift(-1).fillna(average_predicted_sales)
    result['after_next_month_sales'] = result['Predicted Sales'].shift(-2).fillna(average_predicted_sales)

    # Рассчитываем оборачиваемость, используя среднее предсказанных продаж следующего и после следующего месяца
    # и остаток на конец текущего месяца
    result['turnover'] = result['balance end'] / ((result['next_month_sales'] + result['after_next_month_sales']) / 2)

    return result

def optimize_orders(result, calculate_turnover):
    if 'orders' not in result.columns:
        result['orders'] = 0
    
    is_future = result['month'] >= '2018-09-01'

    initial_orders = np.where(is_future, result['Predicted Sales'] * 0.1, result['orders'])
    initial_guess = initial_orders[is_future] 

    lower_bounds = np.where(is_future, 0, result['orders'])[is_future]  # Нижняя граница
    quantile_sales = result['sales'].quantile(0.95)  # 95-й квантиль
    upper_bounds = np.where(is_future, quantile_sales, result['orders'])[is_future]
    bounds = list(zip(lower_bounds, upper_bounds))

    def turnover_diff_function(orders, result):
        temp_result = result.copy()
        temp_result.loc[temp_result['month'] >= '2018-09-01', 'orders'] = orders.astype(int)
        temp_result = calculate_turnover(temp_result)
        temp_result['average_target'] = (temp_result['min'] + temp_result['max']) / 2
        temp_result['turnover_diff'] = np.sqrt((temp_result['average_target'] - temp_result['turnover']) ** 2)
        return temp_result['turnover_diff'].sum()

    optimization_result = minimize(
        fun=turnover_diff_function,
        x0=initial_guess,
        args=(result,),
        method='SLSQP',
        bounds=bounds
    )

    if len(optimization_result.x) == is_future.sum():
        result.loc[is_future, 'orders'] = optimization_result.x.astype(int)
    else:
        print(f"Ошибка: размер оптимизированных значений {len(optimization_result.x)} не соответствует ожидаемому {is_future.sum()}.")

    result = calculate_turnover(result)
    result['average_target'] = (result['min'] + result['max']) / 2
    result['turnover_diff'] = np.sqrt((result['average_target'] - result['turnover']) ** 2).round(3)

    return result, optimization_result


def main_process(json_data, socketio=None):
    try:
        df = read_sales_data(json_data)

        total_sales_df = create_total_sales_df(df)
        prediction_df_list = remove_outliers(total_sales_df)
        optimized_params = optimize_prophet_params(prediction_df_list, total_sales_df, socketio)
        predictions = evaluate_sales_predictions(df, optimized_params)
        result = aggregate_final_data(df, predictions)
        optimized_result, optimization_details = optimize_orders(result, calculate_turnover)

        columns_order = [
            'product_id', 'productcategory', 'month', 'balancestart', 'arrival',
            'expenditure', 'balance end', 'turnover', 'turnover_diff', 'orders',
            'Predicted Sales', 'transit', 'backorder'
        ]
        result_ordered = optimized_result[columns_order]

        return result_ordered

    except Exception as e:
        print(f"Ошибка при обработке файла: {str(e)}")
        traceback.print_exc()
        return None