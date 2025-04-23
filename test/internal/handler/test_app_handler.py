import pytest

from pkg.response import HttpCode


class TestAppHandler:
    """app控制器的测试类"""

    @pytest.mark.parametrize(
        "app_id,query",
        [
            ("e60c2b0c-2c74-4d6d-b1ee-b87e0379d19e", None),
            ("e60c2b0c-2c74-4d6d-b1ee-b87e0379d19e", "你好，你是？"),
        ]
    )
    def test_completion(self, app_id, query, client):
        resp = client.post(f"/apps/{app_id}/debug", json={"query": query})
        assert resp.status_code == 200
        if query is None:
            assert resp.json.get("code") == HttpCode.VALIDATE_ERROR
        else:
            assert resp.json.get("code") == HttpCode.SUCCESS
