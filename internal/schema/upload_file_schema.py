from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileField, FileAllowed, FileSize
from marshmallow import Schema, fields, pre_dump

from internal.entity.upload_file_entity import ALLOWED_DOCUMENT_EXTENSION, ALLOWED_IMAGE_EXTENSION
from internal.model import UploadFile


class UploadFileReq(FlaskForm):
    file = FileField("file", validators=[
        FileRequired("上传文件不能为空"),
        FileSize(max_size=15 * 1024 * 1024, message="上传文件最大不能超过15MB"),
        FileAllowed(ALLOWED_DOCUMENT_EXTENSION, message=f"仅允许上传{'/'.join(ALLOWED_DOCUMENT_EXTENSION)}格式文件")
    ])


class UploadFileResp(Schema):
    id = fields.UUID(dump_default="")
    account_id = fields.UUID(dump_default="")
    name = fields.String(dump_default="")
    key = fields.String(dump_default="")
    size = fields.Integer(dump_default=0)
    extension = fields.String(dump_default="")
    mime_type = fields.String(dump_default="")
    created_at = fields.Integer(dump_default=0)

    @pre_dump
    def process_data(self, data: UploadFile, **kwargs):
        return {
            "id": data.id,
            "account_id": data.account_id,
            "name": data.name,
            "key": data.key,
            "size": data.size,
            "extension": data.extension,
            "mime_type": data.mime_type,
            "created_at": int(data.created_at.timestamp()),
        }


class UploadImageReq(FlaskForm):
    file = FileField("file", validators=[
        FileRequired("上传图片不能为空"),
        FileSize(max_size=15 * 1024 * 1024, message="上传图片最大不能超过15MB"),
        FileAllowed(ALLOWED_IMAGE_EXTENSION, message=f"仅允许上传{'/'.join(ALLOWED_IMAGE_EXTENSION)}格式文件")
    ])
