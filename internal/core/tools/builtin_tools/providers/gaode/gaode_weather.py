import json
import os
from typing import Any, Type

from langchain import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from internal.lib.helper import add_attribute


class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


class GaodeWeatherTool(BaseTool):
    name: str = "GaodeWeather"
    description: str = "当你想查询天气或者与天气相关的问题时可以使用的工具"
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema

    def _run(self, *args: Any, **kwargs: Any) -> str:
        try:
            gaode_api_key = os.environ.get("GAODE_API_KEY")
            if not gaode_api_key:
                return f"高德开放平台API未配置"

            city = kwargs.get("city", "")
            api_domain = "https://restapi.amap.com/v3"
            session = requests.session()

            city_response = session.get(
                method="GET",
                url=f"{api_domain}/config/district?key={gaode_api_key}&keywords={city}&subdistrict=0",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )

            city_response.raise_for_status()
            city_data = city_response.json()
            if city_data.get("info") == "OK":
                ad_code = city_data["district"][0]["adcode"]

                weather_response = session.request(
                    method="GET",
                    url=f"{api_domain}/weather/weatherInfo?key={gaode_api_key}&city={ad_code}&extensions=all",
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                weather_response.raise_for_status()
                weather_data = weather_response.json()
                if weather_data.get("info") == "OK":
                    return json.dumps(weather_data)
            return f"获取{city}天气预报信息失败"
        except Exception as e:
            return f"获取{kwargs.get('city', '')}天气预报信息失败"


@add_attribute("args_schema", GaodeWeatherArgsSchema)
def gaode_weather(**kwargs) -> BaseTool:
    return GaodeWeatherTool()
