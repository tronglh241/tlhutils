import sqlite3
from typing import Any, List, Optional, Tuple, Union


class SQLiteCursor:
    def __init__(self, connection: sqlite3.Connection) -> None:
        '''
        Initializes the SQLiteCursor class.

        Args:
            connection (sqlite3.Connection): SQLite connection object.
        '''
        self.connection: sqlite3.Connection = connection
        self.cursor: sqlite3.Cursor = self.connection.cursor()

    def execute(self, query: str, parameters: Optional[Union[Tuple[Any, ...], dict[str, Any]]] = None) -> None:
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

    def fetchall(self) -> List[Tuple[Any, ...]]:
        '''Fetches all rows of the query result.'''
        return self.cursor.fetchall()

    def fetchone(self) -> Optional[Any]:
        '''Fetches the next row of a query result.'''
        return self.cursor.fetchone()

    def commit(self) -> None:
        '''Commits the current transaction.'''
        self.connection.commit()

    def close(self) -> None:
        '''Closes the cursor.'''
        self.cursor.close()


class SQLiteDatabase:
    def __init__(self, db_path: str) -> None:
        '''
        Initializes the SQLiteDatabase class.

        Args:
            db_path (str): Path to the SQLite database file.
        '''
        self.connection: sqlite3.Connection = sqlite3.connect(db_path)

    def table_exists(self, table_name: str) -> bool:
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

    def query(self, query: str, parameters: Optional[Union[Tuple[Any, ...], dict[str, Any]]] = None) -> SQLiteCursor:
        '''
        Executes a given SQL query and returns the results.

        Args:
            query (str): The SQL query to execute.
            parameters (tuple or dict, optional): Parameters for the query.

        Returns:
            SQLiteCursor: A cursor with the results of the query.
        '''
        cursor = SQLiteCursor(self.connection)
        cursor.execute(query, parameters)
        return cursor

    def close(self) -> None:
        '''Closes the database connection.'''
        self.connection.close()
