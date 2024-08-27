from functools import wraps

import psycopg2


def psycopg2_cursor(conn_info):
    """Wrap function to set up and tear down a Postgres connection while
    providing a cursor object to make queries with.
    """
    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                connection = psycopg2.connect(**conn_info)
                cursor = connection.cursor()

                return_val = f(cursor, *args, **kwargs)

            finally:
                connection.commit()
                connection.close()

            return return_val
        return wrapper
    return wrap


PSQL_CONN = {
    'host': '127.0.0.1',
    'port': '5432',
    'user': 'postgres',
    'password': '2659',
    'dbname': 'postgres'
}


# @psycopg2_cursor(PSQL_CONN)
# def tester(cursor):
#     """Test function that uses our psycopg2 decorator
#     """
#     cursor.execute('SELECT 1 + 1')
#     return cursor.fetchall()
