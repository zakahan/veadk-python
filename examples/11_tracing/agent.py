# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A VeADK agent whose weather tool emits a custom business span."""

import os

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from veadk import Agent
from veadk.tracing.telemetry.exporters.base_exporter import BaseExporter
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

# This tracer uses the global OpenTelemetry TracerProvider initialized by
# OpentelemetryTracer below. A span started while the tool is running is
# automatically attached to the current agent/tool trace.
weather_tracer = trace.get_tracer("veadk.examples.tracing.weather")


def _enabled(env_name: str) -> bool:
    return os.getenv(env_name, "").lower() == "true"


def build_exporters() -> list[BaseExporter]:
    """Build cloud exporters enabled through runtime environment variables."""
    exporters: list[BaseExporter] = []
    if _enabled("ENABLE_APMPLUS"):
        from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter

        exporters.append(APMPlusExporter())
    if _enabled("ENABLE_COZELOOP"):
        from veadk.tracing.telemetry.exporters.cozeloop_exporter import (
            CozeloopExporter,
        )

        exporters.append(CozeloopExporter())
    if _enabled("ENABLE_TLS"):
        from veadk.tracing.telemetry.exporters.tls_exporter import TLSExporter

        exporters.append(TLSExporter())
    return exporters


def get_city_weather(city: str) -> dict[str, str]:
    """Get the current weather for a city with a custom business span.

    Args:
        city: The English name of the city, e.g. "Beijing".
    """
    normalized_city = city.lower().strip()

    with weather_tracer.start_as_current_span(
        "weather.lookup",
        # The exception path below records these explicitly for demonstration.
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        # Attributes are searchable metadata. Do not put secrets or sensitive
        # user content in them.
        span.set_attribute("weather.city", normalized_city)
        span.set_attribute("weather.provider", "demo-fixed-data")
        span.add_event("weather.lookup.started")

        try:
            if normalized_city == "error":
                # A deterministic input for trying the error trace locally.
                raise RuntimeError("The demo weather provider is unavailable")

            weather_by_city = {
                "beijing": "Sunny, 25°C",
                "shanghai": "Cloudy, 22°C",
                "shenzhen": "Partly cloudy, 29°C",
            }
            weather = weather_by_city.get(normalized_city)
            span.set_attribute("weather.found", weather is not None)
            span.add_event(
                "weather.lookup.completed",
                {"weather.result": "found" if weather else "not_found"},
            )
            return {"result": weather or f"No data for {city}"}
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def create_agent(*, exporters: list[BaseExporter] | None = None) -> Agent:
    """Create the traced agent with explicitly selected exporters.

    AgentKit already manages its APMPlus span processor, so ``app.py`` passes no
    exporters here. The local runner passes ``build_exporters()`` instead.
    Keeping those paths separate prevents the same span from being exported by
    both AgentKit and this example.
    """
    agent_tracer = OpentelemetryTracer(exporters=list(exporters or []))
    return Agent(
        name="traced_agent",
        description="A weather assistant whose business tool emits custom spans.",
        instruction=(
            "Help with weather. Always use get_city_weather when asked about a "
            "city's weather. Pass the English city name to the tool."
        ),
        tools=[get_city_weather],
        tracers=[agent_tracer],
    )
