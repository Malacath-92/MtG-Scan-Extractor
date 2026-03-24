from dataclasses import dataclass
from typing import Type
import math


@dataclass
class Vec:
    x: int | float
    y: int | float

    def __neg__(self):
        return Vec(-self.x, -self.y)

    def __add__(lhs, rhs: Type["Vec"]):
        return Vec(lhs.x + rhs.x, lhs.y + rhs.y)

    def __sub__(lhs, rhs: Type["Vec"]):
        return Vec(lhs.x - rhs.x, lhs.y - rhs.y)

    def __mul__(lhs, rhs: int):
        return Vec(lhs.x * rhs, lhs.y * rhs)

    def __truediv__(lhs, rhs: int):
        return Vec(lhs.x / rhs, lhs.y / rhs)

    def dot(lhs, rhs: Type["Vec"]):
        return lhs.x * rhs.x + lhs.y * rhs.y

    def cross(lhs, rhs: Type["Vec"]):
        return lhs.x * rhs.y - lhs.y * rhs.x

    def normalize(self):
        return self / self.len

    def offset(self, offset: Type["Vec"]):
        return Vec(self.x + offset.x, self.y + offset.y)

    def rotate(self, angle: float, pivot: Type["Vec"]):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        off = self - pivot

        return Vec(
            off.x * cos_a + off.y * sin_a + pivot.x,
            -off.x * sin_a + off.y * cos_a + pivot.y,
        )

    @property
    def len(self):
        return math.sqrt(self.len_sqr)

    @property
    def len_sqr(self):
        return self.x * self.x + self.y * self.y


@dataclass
class Line:
    from_pos: Vec
    to_pos: Vec

    @property
    def delta(self) -> Vec:
        return self.to_pos - self.from_pos

    @property
    def dir(self) -> Vec:
        return self.delta.normalize()

    @property
    def vertical(self) -> bool:
        return abs(self.delta.x) < abs(self.delta.y)

    @property
    def horizontal(self) -> bool:
        return not self.vertical

    @property
    def slope(self) -> int:
        if self.vertical:
            return self.delta.x / self.delta.y
        return self.delta.y / self.delta.x

    @property
    def intercept(self) -> int:
        if self.vertical:
            return self.to_pos.x - self.slope * self.to_pos.y
        return self.to_pos.y - self.slope * self.to_pos.x

    def offset(self, offset: Type["Vec"]):
        return Line(
            self.from_pos.offset(offset),
            self.to_pos.offset(offset),
        )

    def rotate(self, angle: float, pivot: Type["Vec"]):
        return Line(
            self.from_pos.rotate(angle, pivot),
            self.to_pos.rotate(angle, pivot),
        )

    def parallel(lhs, rhs: Type["Line"]):
        slope_diff = abs(lhs.slope - rhs.slope)
        return lhs.vertical == rhs.vertical and slope_diff < 0.01

    def dist(lhs, rhs: Type["Line"]):
        assert lhs.parallel(rhs)

        lhs_slope_factor = math.sqrt(lhs.slope * lhs.slope + 1)
        rhs_slope_factor = math.sqrt(rhs.slope * rhs.slope + 1)
        slope_factor = (lhs_slope_factor + rhs_slope_factor) / 2

        return abs(rhs.intercept - lhs.intercept) / slope_factor


@dataclass
class Transform:
    rotation: float
    translation: Vec
