from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, List, Union

import yaml
from yacs.config import CfgNode as BaseCfgNode


class CfgNode(BaseCfgNode):  # type: ignore
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CfgNode:
        '''
        Loads a configuration from a YAML file and converts dictionary-like objects
        recursively to instances of CfgNode.

        Args:
            filepath (str or Path): Path to the YAML configuration file.

        Returns:
            CfgNode: A CfgNode instance with loaded configuration.
        '''
        # Load the YAML file
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        # Recursively convert dict-like objects to CfgNode instances
        config = cls._convert_to_cfg_node(config_data)

        assert isinstance(config, CfgNode)
        return config

    @classmethod
    def _convert_to_cfg_node(cls, data: Any) -> Union[CfgNode, List[Any], Any]:
        '''
        Recursively converts dictionary-like objects to CfgNode instances.

        Args:
            data (dict or list): The configuration data to convert.

        Returns:
            CfgNode or list: CfgNode if data is a dictionary, otherwise list of converted elements.
        '''
        if isinstance(data, dict):
            # Convert dictionary to CfgNode and recursively process each item
            cfg_node = cls()
            for key, value in data.items():
                cfg_node[key] = cls._convert_to_cfg_node(value)
            return cfg_node
        elif isinstance(data, list):
            # Recursively process each item in a list
            return [cls._convert_to_cfg_node(item) for item in data]
        else:
            # Return the item if it is not a dict or list
            return data

    @staticmethod
    def argparse():
        caller_file = Path(inspect.stack()[1].filename)

        parser = argparse.ArgumentParser()
        parser.add_argument('--config-file', default=str(caller_file.parent.joinpath('configs', 'config.yml')))
        args = parser.parse_args()

        return args.config_file
