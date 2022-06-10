from typing import Optional
from pydantic import BaseModel 


class Characteristic(BaseModel):
    """ Define the parameters what using for method post

    Args:
        BaseModel (class): Parameters sepal_length, sepal_width, petal_lenght, petal_width.
    """
    sepal_length:Optional[float]
    sepal_width:Optional[float]
    petal_length:Optional[float]
    petal_width:Optional[float]
