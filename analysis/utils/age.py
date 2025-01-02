from enum import IntEnum
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class AgeCategory(IntEnum):
    """Age classification with integrated numeric values"""

    infantil = 0
    adolescente = 1
    adulto = 2
    idoso = 3

    @property
    def range(self) -> Tuple[float, float]:
        """Returns the age range for each category"""
        RANGES = {
            self.infantil: (0, 12),
            self.adolescente: (13, 19),
            self.adulto: (20, 59),
            self.idoso: (60, float("inf")),
        }
        return RANGES[self]


class AgeLevel(BaseModel):
    """Model for age classification"""

    value: Optional[float] = Field(
        ..., description="Age level value", examples=[0, 12, 60]
    )

    @property
    def category(self) -> AgeCategory:
        """Determines Age category based on value"""
        for category in AgeCategory:
            min_val, max_val = category.range
            if min_val <= self.value <= max_val:
                return category.name
        raise ValueError(f"Invalid Age: {self.value}")

    @property
    def label(self) -> int:
        """Returns the numeric label of the category"""
        return self.category.value


def age_str_to_label(category: str) -> int:
    """Converts category string to numeric label"""
    try:
        return AgeCategory[category.lower()].value
    except KeyError:
        valid = ", ".join(c.name.lower() for c in AgeCategory)
        raise ValueError(f"Invalid category. Use: {valid}")
