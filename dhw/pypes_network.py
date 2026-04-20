"""
pype_schema-based DHW network with only water draws (no recirculation).

Defines a :class:`DHWDrawNetwork` that builds a JSON configuration in the
same style as ``desal_sample.json`` and loads it into a ``pype_schema``
``Network`` object via :class:`pype_schema.parse_json.JSONParser`.

Topology
--------
::

    ColdWaterSupply (Reservoir)
        └── ColdInlet ──▶ DHWSystem (Facility)
                               │
                               ├── HotWaterHeater (Tank)
                               │       └── HeaterToJunctionF1
                               ├── Junction_Floor1 (Junction)
                               │       ├── JunctionF1_toBath1Shower ─▶ Bathroom1_Shower (Tank)
                               │       ├── JunctionF1_toBath1Faucet ─▶ Bathroom1_Faucet (Tank)
                               │       └── JunctionF1_toJunctionF2  ─▶ Junction_Floor2 (Junction)
                               │                                          ├── JunctionF2_toBath2Shower ─▶ Bathroom2_Shower (Tank)
                               │                                          ├── JunctionF2_toBath2Faucet ─▶ Bathroom2_Faucet (Tank)
                               │                                          └── JunctionF2_toKitchenSink ─▶ Kitchen_Sink (Tank)

Usage
-----
>>> from dhw.pypes_network import DHWDrawNetwork
>>> net = DHWDrawNetwork()
>>> network_obj = net.build(json_path="output/dhw_draw_network.json")
>>> print(network_obj)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pype_schema.parse_json import JSONParser


class DHWDrawNetwork:
    """Builder for a draw-only Domestic Hot Water ``pype_schema`` network.

    The configuration is assembled in memory (no external file is read) and
    written to disk only so that :class:`pype_schema.parse_json.JSONParser`
    can consume it via its file-based constructor.

    Parameters
    ----------
    name :
        Identifier used for the top-level :class:`pype_schema.node.Facility`.
    heater_volume_m3 :
        Volume of the hot water heater tank, in cubic metres.
    draw_volume_m3 :
        Nominal volume assigned to each draw-point tank, in cubic metres.
    shower_elevation_m :
        Elevation of shower/faucet draw points, in metres.
    kitchen_elevation_m :
        Elevation of kitchen draw point, in metres.

    Attributes
    ----------
    config : dict
        The JSON-compatible network description.
    network : pype_schema.node.Network | None
        Populated after :meth:`build` is invoked.
    """

    def __init__(
        self,
        name: str = "DHWSystem",
        heater_volume_m3: float = 0.3,
        draw_volume_m3: float = 0.002,
        shower_elevation_m: float = 3.0,
        kitchen_elevation_m: float = 0.0,
    ) -> None:
        self.name = name
        self.heater_volume_m3 = heater_volume_m3
        self.draw_volume_m3 = draw_volume_m3
        self.shower_elevation_m = shower_elevation_m
        self.kitchen_elevation_m = kitchen_elevation_m

        self.config: dict[str, Any] = self._build_config()
        self.network = None

    # ------------------------------------------------------------------
    # JSON construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flowrate(units: str = "cubic meters / day") -> dict[str, Any]:
        return {"min": None, "max": None, "design": None, "units": units}

    @staticmethod
    def _pipe(source: str, destination: str, **extra: Any) -> dict[str, Any]:
        pipe = {
            "type": "Pipe",
            "source": source,
            "destination": destination,
            "contents": "DrinkingWater",
            "tags": {},
        }
        pipe.update(extra)
        return pipe

    def _draw_tank(
        self,
        elevation_m: float,
        volume_m3: float | None = None,
        temp_tag_id: str | None = None,
    ) -> dict[str, Any]:
        tank: dict[str, Any] = {
            "type": "Tank",
            "num_units": 1,
            "elevation": {"value": elevation_m, "units": "m"},
            "volume": {
                "value": volume_m3 if volume_m3 is not None else self.draw_volume_m3,
                "units": "m^3",
            },
            "contents": "DrinkingWater",
            "tags": {},
        }
        if temp_tag_id is not None:
            tank["tags"][temp_tag_id] = {
                "type": "Temperature",
                "units": "celsius",
                "contents": "DrinkingWater",
                "totalized": False,
            }
        return tank

    def _build_config(self) -> dict[str, Any]:
        facility_nodes = [
            "HotWaterHeater",
            "Junction_Floor1",
            "Junction_Floor2",
            "Bathroom1_Shower",
            "Bathroom1_Faucet",
            "Bathroom2_Shower",
            "Bathroom2_Faucet",
            "Kitchen_Sink",
        ]
        facility_connections = [
            "HeaterToJunctionF1",
            "JunctionF1_toBath1Shower",
            "JunctionF1_toBath1Faucet",
            "JunctionF1_toJunctionF2",
            "JunctionF2_toBath2Shower",
            "JunctionF2_toBath2Faucet",
            "JunctionF2_toKitchenSink",
        ]

        return {
            "nodes": ["ColdWaterSupply", self.name],
            "connections": ["ColdInlet"],
            "virtual_tags": {},

            "ColdWaterSupply": {
                "type": "Reservoir",
                "contents": "DrinkingWater",
                "tags": {},
            },

            self.name: {
                "type": "Facility",
                "input_contents": "DrinkingWater",
                "output_contents": "DrinkingWater",
                "elevation": {"value": 0, "units": "m"},
                "flowrate": self._flowrate(),
                "tags": {},
                "nodes": facility_nodes,
                "connections": facility_connections,
            },

            "HotWaterHeater": {
                "type": "Tank",
                "num_units": 1,
                "elevation": {"value": 0, "units": "m"},
                "volume": {"value": self.heater_volume_m3, "units": "m^3"},
                "contents": "DrinkingWater",
                "tags": {
                    "HeaterOutletTemp": {
                        "type": "Temperature",
                        "units": "celsius",
                        "contents": "DrinkingWater",
                        "totalized": False,
                    }
                },
            },

            "Junction_Floor1": {
                "type": "Junction",
                "contents": "DrinkingWater",
                "tags": {},
            },
            "Junction_Floor2": {
                "type": "Junction",
                "contents": "DrinkingWater",
                "tags": {},
            },

            "Bathroom1_Shower": self._draw_tank(
                self.shower_elevation_m, temp_tag_id="B1ShowerTemp"
            ),
            "Bathroom1_Faucet": self._draw_tank(
                self.shower_elevation_m, volume_m3=0.001
            ),
            "Bathroom2_Shower": self._draw_tank(
                self.shower_elevation_m + 3.0, temp_tag_id="B2ShowerTemp"
            ),
            "Bathroom2_Faucet": self._draw_tank(
                self.shower_elevation_m + 3.0, volume_m3=0.001
            ),
            "Kitchen_Sink": self._draw_tank(
                self.kitchen_elevation_m, volume_m3=0.001
            ),

            "ColdInlet": self._pipe(
                source="ColdWaterSupply",
                destination=self.name,
                entry_point="HotWaterHeater",
                flowrate=self._flowrate(),
            ),

            "HeaterToJunctionF1": self._pipe("HotWaterHeater", "Junction_Floor1"),
            "JunctionF1_toBath1Shower": self._pipe(
                "Junction_Floor1", "Bathroom1_Shower"
            ),
            "JunctionF1_toBath1Faucet": self._pipe(
                "Junction_Floor1", "Bathroom1_Faucet"
            ),
            "JunctionF1_toJunctionF2": self._pipe(
                "Junction_Floor1", "Junction_Floor2"
            ),
            "JunctionF2_toBath2Shower": self._pipe(
                "Junction_Floor2", "Bathroom2_Shower"
            ),
            "JunctionF2_toBath2Faucet": self._pipe(
                "Junction_Floor2", "Bathroom2_Faucet"
            ),
            "JunctionF2_toKitchenSink": self._pipe(
                "Junction_Floor2", "Kitchen_Sink"
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a deep-ish copy of the JSON-compatible configuration."""
        return json.loads(json.dumps(self.config))

    def write_json(self, path: str | Path, indent: int = 2) -> Path:
        """Serialize the configuration to ``path`` and return the ``Path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.config, f, indent=indent)
        return path

    def build(
        self,
        json_path: str | Path | None = None,
        verbose: bool = False,
    ):
        """Materialize the network using ``pype_schema.JSONParser``.

        Parameters
        ----------
        json_path :
            Destination for the generated JSON file. If ``None``, a temporary
            file is created and removed after parsing.
        verbose :
            Forwarded to :meth:`JSONParser.initialize_network`.

        Returns
        -------
        pype_schema.node.Network
            The parsed parent network.
        """
        if json_path is None:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            tmp.close()
            json_path = Path(tmp.name)
            cleanup = True
        else:
            cleanup = False

        json_path = self.write_json(json_path)
        try:
            parser = JSONParser(str(json_path))
            self.network = parser.initialize_network(verbose=verbose)
        finally:
            if cleanup:
                try:
                    json_path.unlink()
                except OSError:
                    pass

        return self.network


__all__ = ["DHWDrawNetwork"]
