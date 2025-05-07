from dataclasses import dataclass

from flask_login import login_required, current_user
from injector import inject

from internal.schema.upload_file_schema import UploadFileReq, UploadFileResp, UploadImageReq
from internal.service import CosService
from pkg.response import validate_error_json, success_json


@inject
@dataclass
class UploadFileHandler:
    cos_service: CosService

    @login_required
    def upload_file(self):
        req = UploadFileReq()
        if not req.validate():
            return validate_error_json(req.errors)

        upload_file = self.cos_service.upload_file(req.file.data, True, current_user)

        resp = UploadFileResp()
        return success_json(resp.dump(upload_file))

    @login_required
    def upload_image(self):
        req = UploadImageReq()
        if not req.validate():
            return validate_error_json(req.errors)

        upload_file = self.cos_service.upload_file(req.file.data, True, current_user)

        image_url = self.cos_service.get_file_url(upload_file.key)

        return success_json({"image_url": image_url})
