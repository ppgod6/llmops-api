import math
from dataclasses import dataclass
from typing import Any

from flask_wtf import FlaskForm
from wtforms import IntegerField
from wtforms.validators import Optional, NumberRange

from pkg.sqlalchemy import SQLAlchemy


class PaginatorReq(FlaskForm):
    current_page = IntegerField("current_page", default=1, validators=[
        Optional(),
        NumberRange(min=1, max=9999, message="当前页面的范围在1-9999")
    ])

    page_size = IntegerField("page_size", default=20, validators=[
        Optional(),
        NumberRange(min=1, max=50, message="每页数据的条数范围在1-50")
    ])


@dataclass
class Paginator:
    total_page: int = 0
    total_record: int = 0
    current_page: int = 1
    page_size: int = 20

    def __init__(self, db: SQLAlchemy, req: PaginatorReq = None):
        if req is not None:
            self.current_page = req.current_page.data
            self.page_size = req.page_size.data

        self.db = db

    def paginate(self, select) -> list[Any]:
        p = self.db.paginate(select, page=self.current_page, per_page=self.page_size, error_out=False)

        self.total_record = p.total
        self.total_page = math.ceil(p.total / self.page_size)

        return p.items


@dataclass
class PageModel:
    list: list[Any]
    paginator: Paginator
