from Tools import psycopg2_cursor, PSQL_CONN


@psycopg2_cursor(PSQL_CONN)
def insert_weather_info(cursor, weather_type, car_coef, train_coef, plane_coef):
    query = "INSERT INTO public.weather_info (weather_type, car_coef, train_coef, plane_coef) VALUES (%s, %s, %s, s%)"
    cursor.execute(query, (weather_type, car_coef, train_coef, plane_coef))


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


@psycopg2_cursor(PSQL_CONN)
def insert_agent_info(cursor, reward, stage, action, episode, table_name):
    query = f"INSERT INTO {table_name} (reward, stage, action, episode) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (reward, stage, int(action), episode))
    # return cursor.fetchone()


@psycopg2_cursor(PSQL_CONN)
def insert_booking_info(cursor, transport, status, table_name):
    query = f"INSERT INTO {table_name} (transport, status) VALUES (%s, %s)"
    cursor.execute(query, (transport, status))


@psycopg2_cursor(PSQL_CONN)
def get_booking_info(cursor, table_name):
    cursor.execute(
        f"SELECT transport, status "
        f"FROM {table_name} "
        f"ORDER BY time "
        f"DESC LIMIT 1"
    )
    last_row = cursor.fetchone()
    if last_row is None:
        return None, None
    transport, status = last_row
    return transport, status


@psycopg2_cursor(PSQL_CONN)
def get_table_info(cursor, table_name):
    query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1"
    cursor.execute(query)
    return cursor.fetchone()


@psycopg2_cursor(PSQL_CONN)
def clear_table_info(cursor, table_name):
    query = f"DELETE FROM {table_name}"
    cursor.execute(query)


if __name__ == "__main__":
    get_table_info("public.agent_info")
    get_table_info("public.weather_info")
