from pydantic import BaseModel
from enum import Enum

class Geography(str, Enum):
    France = "France"
    Spain = "Spain"
    Germany = "Germany"

class Gender(str, Enum):
    Male = "Male"
    Female = "Female"


class CustomerData(BaseModel):
    CreditScore: float
    Geography: Geography
    Gender: Gender
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float