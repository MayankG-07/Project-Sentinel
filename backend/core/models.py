from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base

# Instantiate the base class for declarative models
Base = declarative_base()

class Employee(Base):
    """
    SQLAlchemy model for an 'employees' table.
    Represents employee data within the organization.
    """
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    department = Column(String)
    salary = Column(Integer)
    clearance_level = Column(String)
