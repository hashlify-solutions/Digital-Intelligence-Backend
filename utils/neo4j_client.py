from neo4j import GraphDatabase
from typing import Optional, Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify that a working connection can be established."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.consume()
            logger.info("Neo4j connection verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify Neo4j connection: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j: {e}")

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        """Run a Cypher query and return the records as a list."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params or {})
                # Convert result to list immediately to avoid consumption issues
                return list(result)
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise

    def execute_query(self, cypher: str, params: Optional[Dict[str, Any]] = None, 
                     database_: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Any, List[str]]:
        """
        Execute a Cypher query and return records, summary, and keys.
        This follows the neo4j driver's execute_query pattern.
        """
        try:
            db = database_ or self.database
            with self.driver.session(database=db) as session:
                result = session.run(cypher, params or {})
                records = []
                
                # Get keys before consuming records
                keys = result.keys()
                
                # Collect all records
                for record in result:
                    records.append(record.data())
                
                # Get summary after consuming records
                summary = result.consume()
                
                return records, summary, keys
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise


CREATE_CONSTRAINTS = """
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT email_addr IF NOT EXISTS FOR (e:Email) REQUIRE e.address IS UNIQUE;
CREATE CONSTRAINT phone_num IF NOT EXISTS FOR (n:PhoneNumber) REQUIRE n.number IS UNIQUE;
CREATE CONSTRAINT chat_id IF NOT EXISTS FOR (m:ChatMessage) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT call_id IF NOT EXISTS FOR (c:Call) REQUIRE c.id IS UNIQUE;
"""


def init_constraints(neo: "Neo4jClient") -> None:
    """Initialize database constraints."""
    constraints = [
        "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT email_addr IF NOT EXISTS FOR (e:Email) REQUIRE e.address IS UNIQUE",
        "CREATE CONSTRAINT phone_num IF NOT EXISTS FOR (n:PhoneNumber) REQUIRE n.number IS UNIQUE",
        "CREATE CONSTRAINT chat_id IF NOT EXISTS FOR (m:ChatMessage) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT call_id IF NOT EXISTS FOR (c:Call) REQUIRE c.id IS UNIQUE"
    ]
    
    try:
        logger.info("Creating Neo4j constraints...")
        for constraint in constraints:
            try:
                neo.run(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as constraint_error:
                # Constraint might already exist, log but don't fail
                logger.debug(f"Constraint might already exist or failed: {constraint_error}")
        logger.info("Neo4j constraints created successfully")
    except Exception as e:
        logger.error(f"Error creating constraints: {e}")
        # Don't raise the exception - constraints might already exist
        # or the database might not be available yet