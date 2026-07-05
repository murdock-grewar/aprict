
import logging
import sys

import tqdm
import tqdm.contrib.logging

from ..lib import gpu_tools


class ProgressBar:

    _cm: object
    _pbar: tqdm.tqdm
    _gpu_mon: gpu_tools.UtilisationAveragePercMon

    _maxval: float | int
    _desc: str
    _valunit: str
    _persist: bool
    _unit_SI_scale: bool  # noqa: N815
    _smoothing: float

    _enable: bool

    def __init__(
        self,
        *,
        desc: str,
        logger: logging.Logger | None = None,  # This argument is needed to ensure that printing to the output goes through tqdm.write(...) before the logger. This ensures compatibility, so the progress bar is not reprinted when there are log lines in the middle of the progress bar progress.
        maxval: float | int,
        valunit: str,
        persist: bool = False,
        unit_SI_scale: bool = False,  # noqa: N803
        smoothing: float = 0.01,
        enable: bool = True,
    ):
        self._enable = enable
        if self._enable:
            logger = logging.getLogger() if logger is None else logger
            self._cm = tqdm.contrib.logging.logging_redirect_tqdm(loggers=[logger])
            self._gpu_mon = gpu_tools.UtilisationAveragePercMon(gpu_indices=gpu_tools.all_gpu_indices(), interval_seconds=0.5)
            self._pbar = tqdm.tqdm(
                total = maxval,
                desc = desc,
                leave = persist,
                ascii = False,
                disable = None, # Disables on non-TTY
                unit = valunit,
                unit_scale = unit_SI_scale, # Uses SI abbreviations for powers of thousands, e.g. K, M, G, T, P
                file = sys.stderr,
                bar_format = '{desc} {percentage:1.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]',
                smoothing=smoothing,
                postfix="",
            )

    def __enter__(self) -> ProgressBar:  # noqa: D105
        if self._enable:
            self._cm.__enter__()  # ty:ignore[unresolved-attribute]
            self.update_to(0)
        return self

    def update_to(self, val: float | int) -> None:
        """Update the progress bar value."""
        if self._enable:
            pbar = self._pbar
            gpu_perc = self._gpu_mon.gpu_perc  # ty:ignore[unresolved-attribute]
            gpu_perc_str = "--%" if gpu_perc is None else f"{int(round(gpu_perc)): 3d}%"
            pbar.set_postfix_str(f'GPUs: {gpu_perc_str:s}', refresh=False)
            pbar.n = val
            pbar.update(0)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D105
        if self._enable:
            self._pbar.close()
            self._cm.__exit__(exc_type, exc_val, exc_tb)  # ty:ignore[unresolved-attribute]
