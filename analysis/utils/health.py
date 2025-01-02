from enum import IntEnum
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class BMICategory(IntEnum):
    """BMI classification with integrated numeric values"""

    muito_abaixo_peso = 0
    abaixo_do_peso = 1
    peso_normal = 2
    acima_do_peso = 3
    obesidade_lvl_1 = 4
    obesidade_lvl_2 = 5
    obesidade_lvl_3 = 6

    @property
    def range(self) -> Tuple[float, float]:
        """Returns the BMI range for each category"""
        RANGES = {
            self.muito_abaixo_peso: (0.0, 16.9),
            self.abaixo_do_peso: (17.0, 18.49),
            self.peso_normal: (18.5, 24.99),
            self.acima_do_peso: (25.0, 29.99),
            self.obesidade_lvl_1: (30.0, 34.99),
            self.obesidade_lvl_2: (35.0, 40),
            self.obesidade_lvl_3: (40.1, float("inf")),
        }
        return RANGES[self]


class BMI(BaseModel):
    """Model for BMI calculation and classification"""

    value: Optional[float] = Field(
        ..., description="BMI value", examples=[23.567, 18.923, 25.102]
    )

    @property
    def category(self) -> BMICategory:
        """Determines BMI category based on value"""
        for category in BMICategory:
            min_val, max_val = category.range
            if min_val <= self.value <= max_val:
                return category.name
        raise ValueError(f"Invalid BMI: {self.value}")

    @property
    def label(self) -> int:
        """Returns the numeric label of the category"""
        return self.category.value


def bmi_str_to_label(category: str) -> int:
    """Converts category string to numeric label"""
    try:
        return BMICategory[category.lower()].value
    except KeyError:
        valid = ", ".join(c.name.lower() for c in BMICategory)
        raise ValueError(f"Invalid category. Use: {valid}")
