# :warning::minibus: Прогнозирование серьезности ДТП 

Прогнозирование серьезности дорожно-транспортных проишествий в Новгородской области с 2015 по 2024 года методами машинного обучения и разработка API-сервера для обработки запросов на предсказание.

## Цели проекта

* Первая цель проекта - выявить ключевые факторы, влияющие на тяжесть аварии. 

* Вторая цель - разработать модель, которая может точно предсказать тяжесть аварии. Это может быть только что произошедшая авария или потенциальная авария, предсказанная другими моделями. 

* Третья цель - предоставить API-сервер для обращения к модели с целью предсказания в режиме реального времени.

## :books: Технологии использованы
* Python 3.12
* Fast API, Pydantic
* Numpy, pandas, seaborn, matplotlib
* Scikit-learn

## Данные

Датасет получен через платформу сбора данных о дорожно-транспортных проишествиях [карта ДТП](https://dtp-stat.ru/).