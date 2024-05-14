from simulator.config import Config
from simulator.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from simulator.simulator import Simulator
from simulator.utils.random import set_seeds


def main():
    config = Config()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    main()
