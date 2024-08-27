# MARL MSA - Multi-Agent Reinforcement Learning Microservice Architecture

![image](https://github.com/user-attachments/assets/8ebdc814-8eed-48ef-97f6-c0e05f8b0c3d)

MARL_MSA предлагает новый способ интеграции мультиагентного обучения в миксросервисную систему. Проект представляет собой сервис выбора дорожного маршрута, который предлагает пользователю оптимальное туристическое направление на основе текущих погодных условий, а также загруженности траффика. Сервис использует подключенную глубокую нейронную сеть IQN, которая независимо обучает агентов управлять ресурсами внутри системы и выбирать наилучший маршрут с учетом динамически изменяющихся условий.

## Используемые технологии

- **numpy**, **torch**: создание нейронной сети
- **Flask**: основной фреймворк для создания UI

## Установка
```
git clone https://github.com/egormorgunov/smart_city_road.git
cd smart_city_road
pip install -e .
```

## Версии

![image](https://github.com/user-attachments/assets/67b99a0c-cfe0-4dea-ae0a-cfbe8d8a2dd9)

Проект включает в себя два типа сервисов:
- Одноагентный сервис, в котором агент обучается с помощью алгоритма глубокого Q-обучения (см. [Одноагентная среда](single_agent/MicroserviceEnvironment.py))
- Мультиагентный сервис, в котором агенты обучаются с помощью алгоритма независимого Q-обучения (см. [Мультиагентная среда](multi_agent/MicroserviceEnvironment.py))
