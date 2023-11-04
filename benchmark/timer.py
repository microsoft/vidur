from time import perf_counter


class Timer:
    def __init__(name, self):
        self.name = name
        self.time = None

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, *_):
        time = perf_counter() - self.time
        print(f"{self.name} took {time} seconds")
