from __future__ import annotations

import collections
from queue import PriorityQueue
from typing import List, Optional, Tuple, Union

Number = Union[float, int]


class EventQueue:
    def __init__(self):
        self.events: list[Event] = []
        self.q = PriorityQueue()
        self.time = 0

    def put(self, event: Event) -> None:
        self.q.put(event)

    def empty(self) -> bool:
        return self.q.empty()

    def pop(self) -> Optional[Event]:
        event = self.q.get()
        if event is not None:
            self.time = event.time
        return event


class Gpu:
    def __init__(
        self,
        id: int,
        queue: EventQueue,
        graph: PipelineGraph,
        max_breadth: int,
    ):
        self.next_free = 0
        self.queue = queue
        self.graph = graph
        self.id = id
        self.forward_active = 0
        self.max_breadth = max_breadth
        self.forward_pending: list[Computation] = []
        self.order: list[Computation] = []

    def add(self, comp: Computation) -> None:
        if (
            comp.layer == 0
            and not comp.backward
            and self.forward_active >= self.max_breadth
        ):
            self.forward_pending.append(comp)
            self.forward_pending.sort()
        else:
            self.schedule(comp)

    def schedule(self, comp: Computation) -> None:
        if comp.layer == 0 and not comp.backward:
            self.forward_active += 1

        self.order.append(comp)

        time = max(self.queue.time, self.next_free)
        self.next_free = time + comp.cost
        event = Event(self.queue, self.next_free, comp, self.graph)
        self.queue.put(event)

        if comp.backward and comp.layer == 0:
            self.forward_active -= 1
            if self.forward_active >= self.max_breadth:
                raise Exception("too many forward passes were active")
            if self.forward_pending:
                pending = self.forward_pending.pop(0)
                self.schedule(pending)


class Computation:
    def __init__(
        self,
        cost: Number,
        layer: int,
        microbatch: int,
        backward: bool,
        gpu: Gpu,
    ):
        self.cost = cost
        self.layer = layer
        self.microbatch = microbatch
        self.backward = backward
        self.dependencies = set()
        self.gpu = gpu

    def __lt__(self, other: Computation) -> bool:
        if self.backward != other.backward:
            return self.backward > other.backward

        return self.layer > other.layer

    @property
    def id(self) -> Tuple[int, int, int]:
        return (self.layer, self.microbatch, self.backward)

    def add_dependency(self, dependency) -> None:
        self.dependencies.add(dependency.id)

    def remove_dependency(self, dependency) -> bool:
        self.dependencies.remove(dependency.id)
        return len(self.dependencies) == 0


class PipelineGraph:
    def __init__(self):
        self.ready: list[Computation] = []
        self.pending = collections.defaultdict(list)

    def add_ready(self, computation: Computation):
        self.ready.append(computation)

    def add_precursor(self, next: Computation, prev: Computation):
        next.add_dependency(prev)
        self.pending[prev.id].append(next)

    def pop(self) -> Optional[Computation]:
        self.ready.sort()
        return self.ready.pop() if self.ready else None

    def done(self, computation: Computation):
        for consumer in self.pending[computation.id]:
            if consumer.remove_dependency(computation):
                self.add_ready(consumer)
        del self.pending[computation.id]


class Event:
    def __init__(
        self,
        queue: EventQueue,
        time: Number,
        computation: Computation,
        graph: PipelineGraph,
    ):
        self.queue = queue
        self.time = time
        self.computation = computation
        self.graph = graph

    def __lt__(self, other: Event):
        return self.time < other.time

    def run(self):
        self.graph.done(self.computation)
        while comp := self.graph.pop():
            comp.gpu.add(comp)


class Scheduler:
    def __init__(
        self,
        num_microbatches: int,
        num_layers: int,
        num_gpus: int,
        forward_cost: Number,
        backward_cost: Number,
        max_breadth: int,
    ):
        self.num_microbatches = num_microbatches
        self.num_layers = num_layers
        self.num_gpus = num_gpus
        self.forward_cost = forward_cost
        self.backward_cost = backward_cost
        self.max_breadth = max_breadth
        self.queue = EventQueue()
        self.graph = PipelineGraph()
        self.num_interleave = num_layers // num_gpus

        self.gpus = [
            Gpu(gpu, self.queue, self.graph, max_breadth)
            for gpu in range(num_gpus)
        ]

        for mb in range(num_microbatches):
            prev = None
            for layer in range(num_layers):
                gpu = self.gpus[layer % num_gpus]
                comp = Computation(
                    cost=forward_cost,
                    layer=layer,
                    microbatch=mb,
                    backward=False,
                    gpu=gpu,
                )
                if prev:
                    self.graph.add_precursor(comp, prev)
                else:
                    self.graph.add_ready(comp)
                prev = comp
            for layer in reversed(range(num_layers)):
                gpu = self.gpus[layer % num_gpus]
                comp = Computation(
                    cost=backward_cost,
                    layer=layer,
                    microbatch=mb,
                    backward=True,
                    gpu=gpu,
                )
                self.graph.add_precursor(comp, prev)
                prev = comp

    def compute(self) -> Tuple[Number, List[Gpu]]:
        start = Event(
            self.queue, 0, Computation(0, None, None, False, 0), self.graph
        )
        start.run()
        total_order = []
        while not self.queue.empty():
            event = self.queue.pop()
            event.run()
            total_order.append(event.computation)
        return self.queue.time, total_order
