import random
import time
from Tools import psycopg2_cursor, PSQL_CONN


class WeatherInfo:
    names = ("Sun", "Rain")
    coefficients = {"Sun": {"Car": 5, "Train": 2, "Plane": 0.5}, "Rain": {"Car": 2, "Train": 0.5, "Plane": 5}}

    def __init__(self):
        self.name = random.choice(self.names)
        self.rewards = self.coefficients[self.name]

    def __str__(self):
        return f" Имя = {self.name}, Награды = {self.rewards}"


@psycopg2_cursor(PSQL_CONN)
def get_info(cursor):
    cursor.execute(
        "SELECT weather_type, car_coef, train_coef, plane_coef "
        "FROM public.weather_info "
        "ORDER BY time "
        "DESC LIMIT 1"
    )
    last_row = cursor.fetchone()
    name, coef1, coef2, coef3 = last_row
    return name, coef1, coef2, coef3
    # return cursor.fetchall()


@psycopg2_cursor(PSQL_CONN)
def upload_weather_data(cursor, weather_object):
    query = """
        INSERT INTO public.weather_info (weather_type, car_coef, train_coef, plane_coef)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(
        query,
        (
            weather_object.name,
            weather_object.rewards["Car"],
            weather_object.rewards["Train"],
            weather_object.rewards["Plane"],
        ),
    )


while True:
    current_info = WeatherInfo()
    upload_weather_data(current_info)
    print(current_info)
    time.sleep(10)

# a, b, c = get_info()
# print(a, b, c)
