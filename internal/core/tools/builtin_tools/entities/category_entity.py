from pydantic import BaseModel, field_validator

from internal.exception import NotFoundException


class CategoryEntity(BaseModel):
    category: str
    name: str
    icon: str

    @field_validator("icon")
    def check_icon_extension(cls, value: str):
        if not value.endswith(".svg"):
            raise NotFoundException(f"该分类的icon图标并不是.svg格式")
        return value
