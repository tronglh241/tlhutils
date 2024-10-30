import sqlite3


class SQLiteCursor:
    def __init__(self, connection):
        '''
        Initializes the SQLiteCursor class.

        Args:
            connection (sqlite3.Connection): SQLite connection object.
        '''
        self.connection = connection
        self.cursor = self.connection.cursor()

    def execute(self, query, parameters=None):
        '''
        Executes a given SQL query.

        Args:
            query (str): The SQL query to execute.
            parameters (tuple or dict, optional): Parameters for the query.
        '''
        if parameters:
            self.cursor.execute(query, parameters)
        else:
            self.cursor.execute(query)

    def fetchall(self):
        '''Fetches all rows of the query result.'''
        return self.cursor.fetchall()

    def fetchone(self):
        '''Fetches the next row of a query result.'''
        return self.cursor.fetchone()

    def commit(self):
        '''Commits the current transaction.'''
        self.connection.commit()

    def close(self):
        '''Closes the cursor.'''
        self.cursor.close()


class SQLiteDatabase:
    def __init__(self, db_path):
        '''
        Initializes the SQLiteDatabase class.

        Args:
            db_path (str): Path to the SQLite database file.
        '''
        self.connection = sqlite3.connect(db_path)

    def table_exists(self, table_name):
        '''
        Checks if a table exists in the database.

        Args:
            table_name (str): Name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        '''
        query = 'SELECT name FROM sqlite_master WHERE type="table" AND name=?;'
        cursor = self.query(query, (table_name,))
        result = cursor.fetchone() is not None
        cursor.close()
        return result

    def query(self, query, parameters=None):
        '''
        Executes a given SQL query and returns the results.

        Args:
            query (str): The SQL query to execute.
            parameters (tuple or dict, optional): Parameters for the query.

        Returns:
            list: A list of rows returned by the query.
        '''
        cursor = SQLiteCursor(self.connection)
        cursor.execute(query, parameters)
        return cursor

    def close(self):
        '''Closes the database connection.'''
        self.connection.close()
