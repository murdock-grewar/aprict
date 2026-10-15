# Copyright (C) 2026  Murdock Grewar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tools to apply a no-operation (noop) filter operations to projection data."""

import typing

import cupy as cp

# There is a pesky bug that I can't solve. This flag will apply a hotfix.
_HOTFIX_DISABLE_ASYNCHRONICITY: bool = True


class NoOpFilterer:
    """A class to facilitate repetitive noop-fitrations of a cupy projection array."""

    chunk_shape: tuple[int, int, int]
    stream: cp.cuda.Stream

    def __init__(
        self,
        *,
        proj_2d_shape: tuple[int, int],
        projs_per_batch: int = 2,  # not much benefit going above 3, I think. And it uses a lot of VRAM.
        stream: cp.cuda.Stream | None = None,
    ) -> None:
        self.chunk_shape = (projs_per_batch, *proj_2d_shape)
        self.stream = (
            cp.cuda.get_current_stream() if stream is None else stream
        )

    @property
    def num_kerns(self) -> int:
        """The number of kernels applied by this object."""
        return 1

    def _filter_chunk_axis1(
        self,
        cparr_view: cp.ndarray,
    ) -> tuple[list[cp.ndarray], cp.cuda.Event]:
        with self.stream as stream:
            ev = cp.cuda.Event(
                block=True,  # calling event.synchronize() will cause the CPU thread to wait for the event to complete.
                disable_timing=True,
                interprocess=False,
            )
            out = cparr_view.copy()
            stream.record(ev)
            return ev, out

    def filter_bulk_axis1_stream(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> typing.Generator[tuple[list[cp.ndarray], cp.cuda.Event]]:
        """Create a generator of ramp-filtered projection chunks, from an input array."""
        num_slices: int = (
            cparr_view.shape[0] + (self.chunk_shape[0] - 1)
        ) // self.chunk_shape[0]
        for i in range(num_slices):
            sl = slice(
                i * self.chunk_shape[0],
                min((i + 1) * self.chunk_shape[0], cparr_view.shape[0]),
            )
            v = cparr_view[sl]
            out, completion_event = self._filter_chunk_axis1(v)
            if block:
                # self.stream.synchronize()
                completion_event.synchronize()
            yield out, completion_event

    def filter_bulk_axis1_concat(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> tuple[list[cp.ndarray], cp.cuda.Event]:
        """Create an output array of [list of filtered projections, one item for each filter], from an input array."""
        out_gen = self.filter_bulk_axis1_stream(
            cparr_view, block=_HOTFIX_DISABLE_ASYNCHRONICITY
        )
        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            outputs = [
                cp.zeros_like(cparr_view) for _ in range(self.num_kerns)
            ]
            outputs_created_event = stream.record()
        sl_start: int = 0
        events = []
        for index, (out_arrs, e) in enumerate(out_gen):
            sl = slice(sl_start, sl_start + out_arrs[0].shape[0])
            with cp.cuda.Stream(
                non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
            ) as stream:
                stream.wait_event(outputs_created_event)
                stream.wait_event(e)
                for out_index in range(self.num_kerns):
                    cp.multiply(
                        out_arrs[out_index], 1, out=outputs[out_index][sl]
                    )
                ev = cp.cuda.Event(
                    block=_HOTFIX_DISABLE_ASYNCHRONICITY,
                    disable_timing=True,
                    interprocess=False,
                )
                stream.record(ev)
                events.append(ev)
            sl_start += out_arrs[0].shape[0]

        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            for e in events:
                stream.wait_event(e)
            completed_event = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(completed_event)

        if block:
            completed_event.synchronize()

        return outputs, completed_event
