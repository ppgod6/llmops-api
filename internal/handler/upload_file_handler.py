from dataclasses import dataclass

from injector import inject

from internal.schema.upload_file_schema import UploadFileReq, UploadFileResp, UploadImageReq
from internal.service import CosService
from pkg.response import validate_error_json, success_json


@inject
@dataclass
class UploadFileHandler:
    cos_service: CosService

    def upload_file(self):
        req = UploadFileReq()
        if not req.validate():
            return validate_error_json(req.errors)

        upload_file = self.cos_service.upload_file(req.file.data)

        resp = UploadFileResp()
        return success_json(resp.dump(upload_file))

    def upload_image(self):
        req = UploadImageReq()
        if not req.validate():
            return validate_error_json(req.errors)

        upload_file = self.cos_service.upload_file(req.file.data, True)

        image_url = self.cos_service.get_file_url(upload_file.key)

        return success_json({"image_url": image_url})
