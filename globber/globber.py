from pathlib import Path
from typing import List, Optional

from natsort import natsorted


class FileGlobber:
    def __init__(
        self,
        root_dir: str,
        pattern: Optional[str] = None,
        dir_only: bool = False,
        natsort: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.pattern = pattern
        self.dir_only = dir_only
        self.natsort = natsort

    def glob(self) -> List[Path]:
        # If no pattern is provided, return the root directory
        if not self.pattern:
            return [self.root_dir]

        if not self.root_dir.is_dir():
            raise ValueError(f"The provided root directory '{self.root_dir}' is not valid.")

        # Perform the glob operation
        paths = list(self.root_dir.glob(self.pattern))

        # Filter for directories only if specified
        if self.dir_only:
            paths = [p for p in paths if p.is_dir()]

        if self.natsort:
            paths = natsorted(paths, key=lambda p: str(p))

        return paths
